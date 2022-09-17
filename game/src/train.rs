use bytemuck::cast_slice;
use bytes::Bytes;
use ndarray::Array;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;
use std::{cell::RefCell, rc::Rc};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::sync::watch::Sender;

use crate::game::*;
use crate::model::*;
use crate::player::*;

#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub struct TrainDataItem(
    StateTensor,   /* 4x15x15 state */
    SquaredMatrix, /* 15*15 prob per position */
    f32,           /* winner */
);

impl Into<Bytes> for TrainDataItem {
    fn into(self) -> Bytes {
        let buffer: Vec<u8> = bincode::serialize(&self).unwrap();
        Bytes::from(buffer)
    }
}

impl From<Bytes> for TrainDataItem {
    fn from(bytes: Bytes) -> Self {
        let instance: Self = bincode::deserialize(&bytes).expect("Unable to deserialize");
        instance
    }
}

pub struct DataProducer {
    tx: UnboundedSender<TrainDataItem>,
    latest_parameters: Arc<Mutex<Option<Bytes>>>,
}

impl DataProducer {
    pub fn new() -> (
        Self,
        UnboundedReceiver<TrainDataItem>,
        Arc<Mutex<Option<Bytes>>>,
    ) {
        let latest_parameters = Arc::new(Mutex::new(None));
        let (tx, rx) = mpsc::unbounded_channel::<TrainDataItem>();
        (
            Self {
                tx: tx,
                latest_parameters: latest_parameters.clone(),
            },
            rx,
            latest_parameters,
        )
    }
    pub fn run(self: &mut Self) {
        let model = PolicyValueModel::get_best();
        let model = Rc::new(RefCell::new(model));

        loop {
            let option = self.latest_parameters.lock().unwrap().take();
            if let Some(latest_parameter) = option {
                if latest_parameter.len() > 0 {
                    model
                        .borrow()
                        .import(latest_parameter)
                        .expect("Unable to import parameter");
                }
            }

            let (mut black_player, mut white_player) = SelfPlayer::new_pair(model.clone());
            let state = Match::new(&mut black_player, &mut white_player).play_to_end();
            let score = match &state {
                TerminalState::BlackWon => 1f32,
                TerminalState::WhiteWon => -1f32,
                TerminalState::Draw => 0f32,
                _ => unreachable!(),
            };
            black_player.consume(|state_tensor: StateTensor, prob_matrix: SquaredMatrix| {
                self.tx
                    .send(TrainDataItem(state_tensor, prob_matrix, score))
                    .expect("Failed to send");
            });
            let score = match &state {
                TerminalState::BlackWon => -1f32,
                TerminalState::WhiteWon => 1f32,
                TerminalState::Draw => 0f32,
                _ => unreachable!(),
            };
            white_player.consume(|state_tensor: StateTensor, prob_matrix: SquaredMatrix| {
                self.tx
                    .send(TrainDataItem(state_tensor, prob_matrix, score))
                    .expect("Failed to send");
            });
        }
    }
}

pub struct Trainer {
    payload_sender: Sender<Bytes>, // broadcast latest parameters
    data_receiver: UnboundedReceiver<TrainDataItem>, // reveice train data set
    model: PolicyValueModel,
    epochs: usize, // num of train_steps for each update
    kl_targ: f32,
    learn_rate: f32,
    lr_multiplier: f32, // adaptively adjust the learning rate based on KL
    batch_size: usize,
}

impl Trainer {
    pub fn new(payload_sender: Sender<Bytes>, rx: UnboundedReceiver<TrainDataItem>) -> Self {
        let m = PolicyValueModel::get_best();

        let payload = m.export().expect("Unable to export parameters");
        payload_sender.send(payload).expect("Failed to set payload");

        Self {
            payload_sender: payload_sender,
            data_receiver: rx,
            model: m,
            epochs: 5,
            learn_rate: 2e-3,
            lr_multiplier: 1f32,
            kl_targ: 0.02f32,
            batch_size: 500,
        }
    }

    pub async fn run(self: &mut Self) {
        let mut data_buffer = Vec::<TrainDataItem>::with_capacity(5000);

        loop {
            while let Ok(item) = self.data_receiver.try_recv() {
                data_buffer.push(item);
            }

            if data_buffer.len() > self.batch_size * 2 {
                let mut state_tensor_batch = vec![StateTensor::<f32>::default(); self.batch_size];
                let mut mcts_probs_batch = vec![SquaredMatrix::<f32>::default(); self.batch_size];
                let mut winner_batch = vec![0f32; self.batch_size];

                // randomly pick a batch
                let mut count = 0;
                while count < self.batch_size {
                    let index = rand::random::<usize>() % data_buffer.len();
                    let TrainDataItem(state_tensor, prob_tensor, winner) =
                        data_buffer.swap_remove(index);

                    state_tensor_batch[count] = state_tensor;
                    mcts_probs_batch[count] = prob_tensor;
                    winner_batch[count] = winner;

                    count += 1;
                }

                let now = Instant::now();
                let (old_log_prob_matrix, old_value) = self
                    .model
                    .predict(&state_tensor_batch, true)
                    .expect("Failed to predict");

                let slice: &[f32] = cast_slice(&old_log_prob_matrix);
                let old_log_probs = Array::from_vec(slice.to_vec());
                let old_probs = old_log_probs.mapv(f32::exp);

                let mut kl: f32 = 0f32;
                let mut loss: f32 = 0f32;
                let mut entropy: f32 = 0f32;
                for _ in 0..self.epochs {
                    (loss, entropy) = self
                        .model
                        .train(
                            &state_tensor_batch,
                            &mcts_probs_batch,
                            &winner_batch,
                            self.learn_rate * self.lr_multiplier,
                        )
                        .expect("Failed to train");

                    let (new_log_prob_matrix, new_value) = self
                        .model
                        .predict(&state_tensor_batch, true)
                        .expect("Failed to predict");

                    let slice: &[f32] = cast_slice(&new_log_prob_matrix);
                    let new_log_probs = Array::from_vec(slice.to_vec());

                    // https://docs.rs/ndarray/latest/ndarray/doc/ndarray_for_numpy_users/index.html
                    // kl = np.mean(np.sum(old_probs * (old_log_probs - new_log_probs), axis=1))
                    let diff = &old_probs * (&old_log_probs - &new_log_probs);
                    kl = diff.sum();
                    //kl = x.sum_axis(Axis(2)).mean().unwrap();

                    if kl > self.kl_targ * 4f32 {
                        break; // stopping early if D_KL diverges badly
                    }
                }

                let elapsed = now.elapsed();

                println!(
                    "lr={}; loss={}; entropy={}; kl={}; elapsed={:.2?}",
                    self.learn_rate * self.lr_multiplier,
                    loss,
                    entropy,
                    kl,
                    elapsed
                );

                // adaptively adjust the learning rate
                if kl > self.kl_targ * 2f32 && self.lr_multiplier > 0.1f32 {
                    self.lr_multiplier /= 1.5f32;
                } else if kl < self.kl_targ / 2f32 && self.lr_multiplier < 10f32 {
                    self.lr_multiplier *= 1.5f32;
                }

                let parameters = self.model.export().expect("Unable to export");

                {
                    let mut file = OpenOptions::new()
                        .read(false)
                        .write(true)
                        .truncate(true)
                        .create(true)
                        .open("best.ckpt")
                        .expect("Unable to open file");
                    file.write_all(&parameters).expect("Unable to write file");
                }

                self.payload_sender
                    .send(parameters)
                    .expect("Unable to send");
            } else {
                match self.data_receiver.recv().await {
                    Some(item) => {
                        data_buffer.push(item);
                    }
                    None => {
                        println!("data_receiver is closed");
                        return;
                    }
                }
            }
        }
    }
}
