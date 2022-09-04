use crate::*;

use ndarray::prelude::*;
use ndarray::Array;

use std::{cell::RefCell, fs, rc::Rc};

pub struct TrainDataItem(
    StateTensor,   /* 4x15x15 state */
    SquaredMatrix, /* 15*15 prob per position */
    f32,           /* winner */
);

pub struct Trainer {
    model: Rc<RefCell<PolicyValueModel>>,
    data_buffer: Vec<TrainDataItem>,
    epochs: usize, // num of train_steps for each update
    kl_targ: f32,
    learn_rate: f32,
    lr_multiplier: f32, // adaptively adjust the learning rate based on KL
}

const BATCH_SIZE: usize = 3;

impl Trainer {
    pub fn new() -> Self {
        let m = Rc::new(RefCell::new(get_best_model()));

        Self {
            model: m,
            data_buffer: Vec::with_capacity(1000),
            epochs: 5,
            learn_rate: 2e-3,
            lr_multiplier: 1f32,
            kl_targ: 0.02f32,
        }
    }

    pub fn execute(self: &mut Self) {
        fs::create_dir_all("checkpoints").expect("Unable to create directory");
        let mut round = 0;
        loop {
            self.collect_data();

            if self.data_buffer.len() > BATCH_SIZE as usize {
                let mut state_tensor_batch = [StateTensor::<f32>::default(); BATCH_SIZE];
                let mut mcts_probs_batch = [SquaredMatrix::<f32>::default(); BATCH_SIZE];
                let mut winner_batch = [0f32; BATCH_SIZE];

                // randomly pick a batch
                let mut count = 0;
                while count < BATCH_SIZE {
                    let index = rand::random::<usize>() % self.data_buffer.len();
                    let TrainDataItem(state_tensor, prob_tensor, winner) =
                        self.data_buffer.swap_remove(index);

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

                let old_log_probs = Array::from(old_log_probs);
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

                    let new_log_probs = Array::from(new_log_probs);

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
                /*
                let filename = format!("checkpoints/{}.ckpt", round);
                round += 1;

                self.model
                    .borrow()
                    .save(&filename)
                    .expect("Unable to save checkpoint");*/
                self.model
                    .borrow()
                    .save("best.ckpt")
                    .expect("Unable to save checkpoint");
            }
        }
    }

    fn collect_data(self: &mut Self) {
        let (mut black_player, mut white_player) = SelfPlayer::new_pair(self.model.clone());
        let state = Match::new(&mut black_player, &mut white_player).play_to_end();
        let score = match &state {
            TerminalState::BlackWon => 1f32,
            TerminalState::WhiteWon => -1f32,
            TerminalState::Draw => 0f32,
            _ => unreachable!(),
        };
        black_player.consume(|state_tensor: StateTensor, prob_matrix: SquaredMatrix| {
            self.data_buffer
                .push(TrainDataItem(state_tensor, prob_matrix, score));
        });
        let score = match &state {
            TerminalState::BlackWon => -1f32,
            TerminalState::WhiteWon => 1f32,
            TerminalState::Draw => 0f32,
            _ => unreachable!(),
        };
        white_player.consume(|state_tensor: StateTensor, prob_matrix: SquaredMatrix| {
            self.data_buffer
                .push(TrainDataItem(state_tensor, prob_matrix, score));
        });
    }
}

fn get_best_model() -> PolicyValueModel {
    let export_dir = "/Users/jerry/projects/renju/renju.git/game/renju_15x15_model/";

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
