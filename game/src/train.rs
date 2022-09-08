use crate::*;
use ndarray::{prelude::*, Data};
use std::sync::Mutex;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::task;

use std::sync::Arc;
use std::{cell::RefCell, fs, rc::Rc};
const BATCH_SIZE: usize = 500;

#[derive(Debug)]
pub struct TrainDataItem(
    StateTensor,   /* 4x15x15 state */
    SquaredMatrix, /* 15*15 prob per position */
    f32,           /* winner */
);

pub struct DataProducer {
    tx: UnboundedSender<TrainDataItem>,
    encoded_params: Arc<Mutex<String>>,
}

pub struct Trainer {
    model: Rc<RefCell<PolicyValueModel>>,
    epochs: usize, // num of train_steps for each update
    kl_targ: f32,
    learn_rate: f32,
    lr_multiplier: f32, // adaptively adjust the learning rate based on KL
}

const export_dir: &str = "/Users/jerry/projects/renju/renju.git/game/renju_15x15_model/";
impl DataProducer {
    fn run(self: &mut Self) {
        let model = get_best_model();
        let model = Rc::new(RefCell::new(model));

        let mut loaded_encoded_params = String::new();
        loop {
            let current_encoded_params = {
                let data = self.encoded_params.lock().unwrap();
                data.clone()
            };
            if !loaded_encoded_params.eq_ignore_ascii_case(&current_encoded_params) {
                model
                    .borrow()
                    .import(&current_encoded_params)
                    .expect("Unable to import");
                loaded_encoded_params = current_encoded_params;
                //println!("Loaded new parameters");
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

impl Trainer {
    pub fn new() -> Self {
        let m = Rc::new(RefCell::new(get_best_model()));

        Self {
            model: m,
            epochs: 5,
            learn_rate: 2e-3,
            lr_multiplier: 1f32,
            kl_targ: 0.02f32,
        }
    }

    pub fn execute(self: &mut Self) {
        let (tx, mut rx) = mpsc::unbounded_channel::<TrainDataItem>();

        let mut params = Arc::new(Mutex::new(String::new()));
        let _join_handles: Vec<_> = (0..5)
            .into_iter()
            .map(|_| (tx.clone(), params.clone()))
            .map(|(cloned_tx, cloned_params)| {
                task::spawn_blocking(move || {
                    let mut producer = DataProducer {
                        tx: cloned_tx,
                        encoded_params: cloned_params,
                    };
                    producer.run();
                })
            })
            .collect();

        let mut data_buffer = Vec::<TrainDataItem>::with_capacity(5000);

        loop {
            while let Ok(item) = rx.try_recv() {
                data_buffer.push(item);
            }

            if data_buffer.len() > BATCH_SIZE * 2 {
                let mut state_tensor_batch = [StateTensor::<f32>::default(); BATCH_SIZE];
                let mut mcts_probs_batch = [SquaredMatrix::<f32>::default(); BATCH_SIZE];
                let mut winner_batch = [0f32; BATCH_SIZE];

                // randomly pick a batch
                let mut count = 0;
                while count < BATCH_SIZE {
                    let index = rand::random::<usize>() % data_buffer.len();
                    let TrainDataItem(state_tensor, prob_tensor, winner) =
                        data_buffer.swap_remove(index);

                    state_tensor_batch[count] = state_tensor;
                    mcts_probs_batch[count] = prob_tensor;
                    winner_batch[count] = winner;

                    count += 1;
                }

                let (old_log_probs, old_value) = self
                    .model
                    .borrow()
                    .predict(&state_tensor_batch)
                    .expect("Failed to predict");

                let old_probs = old_log_probs.mapv(f32::exp);

                let mut kl: f32 = 0f32;
                let mut loss: f32 = 0f32;
                let mut entropy: f32 = 0f32;
                for _ in 0..self.epochs {
                    (loss, entropy) = self
                        .model
                        .borrow()
                        .train(
                            &state_tensor_batch,
                            &mcts_probs_batch,
                            &winner_batch,
                            self.learn_rate * self.lr_multiplier,
                        )
                        .expect("Failed to train");

                    let (new_log_probs, new_value) = self
                        .model
                        .borrow()
                        .predict(&state_tensor_batch)
                        .expect("Failed to predict");

                    // https://docs.rs/ndarray/latest/ndarray/doc/ndarray_for_numpy_users/index.html
                    // kl = np.mean(np.sum(old_probs * (old_log_probs - new_log_probs), axis=1))
                    let x = &old_probs * (&old_log_probs - new_log_probs);
                    kl = x.index_axis(Axis(0), 0).sum_axis(Axis(1)).mean().unwrap();
                    //kl = x.sum_axis(Axis(2)).mean().unwrap();

                    if kl > self.kl_targ * 4f32 {
                        break; // stopping early if D_KL diverges badly
                    }
                }

                println!(
                    "lr={}; loss={}; entropy={}; kl={}",
                    self.learn_rate * self.lr_multiplier,
                    loss,
                    entropy,
                    kl
                );

                // adaptively adjust the learning rate
                if kl > self.kl_targ * 2f32 && self.lr_multiplier > 0.1f32 {
                    self.lr_multiplier /= 1.5f32;
                } else if kl < self.kl_targ / 2f32 && self.lr_multiplier < 10f32 {
                    self.lr_multiplier *= 1.5f32;
                }

                let updated_params = self
                    .model
                    .borrow()
                    .save("best.ckpt")
                    .expect("Unable to save checkpoint");

                {
                    let data = &mut params.lock().unwrap();
                    data.replace_range(.., &updated_params);
                }
            } else {
                let ten_millis = std::time::Duration::from_millis(10);
                std::thread::sleep(ten_millis);
            }
        }
    }
}

fn get_best_model() -> PolicyValueModel {
    let ai_model = PolicyValueModel::load(export_dir).expect("Unable to load model");
    let checkpoint_filename = "best.ckpt";
    if fs::metadata(checkpoint_filename).is_ok() {
        match ai_model.restore(checkpoint_filename) {
            Err(e) => {
                println!(
                    "WARNING : Unable to restore checkpoint {}. {}",
                    checkpoint_filename, e
                );
            }
            _ => {
                println!("Successfully loaded checkpoint {}", checkpoint_filename);
            }
        }
    }

    ai_model
}
