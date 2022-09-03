use crate::*;
use bytemuck::cast_slice;
use ndarray::prelude::*;
use ndarray::Array;
use std::borrow::Cow;
use std::{cell::RefCell, fs, rc::Rc};

pub struct SelfPlayerOutcome(
    Vec<RenjuBoard>,
    Vec<Vec<((usize, usize), f32)>>, /*mcts probs */
    Vec<f32>,                        /*winner */
);

pub struct TrainDataItem(
    StateTensor,   /* 4x15x15 state */
    SquaredMatrix, /* 15*15 prob per position */
    f32,           /* winner */
);

pub struct Trainer {
    // temperature parameter in (0, 1] controls the level of exploration
    temperature: f32,
    model: Rc<RefCell<PolicyValueModel>>,
    data_buffer: Vec<TrainDataItem>,
    epochs: usize, // num of train_steps for each update
    kl_targ: f32,
    learn_rate: f32,
    lr_multiplier: f32, // adaptively adjust the learning rate based on KL
}

const BATCH_SIZE: usize = 500;

impl Trainer {
    pub fn new() -> Self {
        let m = Rc::new(RefCell::new(get_best_model()));

        Self {
            temperature: 1e-3,
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

                let tensor1 = Tensor::<f32>::new(&[
                    BATCH_SIZE as u64,
                    4,
                    game::BOARD_SIZE as u64,
                    game::BOARD_SIZE as u64,
                ])
                .with_values(cast_slice(&state_tensor_batch))
                .expect("Unable to create state batch tensor");

                let tensor2 = Tensor::<f32>::new(&[
                    BATCH_SIZE as u64,
                    game::BOARD_SIZE as u64 * game::BOARD_SIZE as u64,
                ])
                .with_values(cast_slice(&mcts_probs_batch))
                .expect("Unable to create MCTS probability batch tensor");

                let tensor3 = Tensor::<f32>::new(&[BATCH_SIZE as u64])
                    .with_values(cast_slice(&winner_batch))
                    .expect("Unable to create winner batch tensor");

                let (old_log_probs, old_value) = self
                    .model
                    .borrow()
                    .predict(&tensor1)
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
                            &tensor1,
                            &tensor2,
                            &tensor3,
                            self.learn_rate * self.lr_multiplier,
                        )
                        .expect("Failed to train");

                    let (new_log_probs, new_value) = self
                        .model
                        .borrow()
                        .predict(&tensor1)
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

                let filename = format!("checkpoints/{}.ckpt", round);
                round += 1;

                self.model
                    .borrow()
                    .save(&filename)
                    .expect("Unable to save checkpoint");
            }
        }
    }

    fn collect_data(self: &mut Self) {
        let outcome = self.self_play();
        assert_eq!(outcome.0.len(), outcome.1.len());
        assert_eq!(outcome.0.len(), outcome.2.len());

        for index in 0..outcome.0.len() {
            let board = &outcome.0[index];
            let mcts_prob_pairs = &outcome.1[index];
            let winner = outcome.2[index];

            // 15x15 tensor records the probability of each move
            let mut mcts_probs_matrix = SquaredMatrix::default();
            mcts_prob_pairs
                .iter()
                .for_each(|((row, col), probability)| {
                    mcts_probs_matrix[*row][*col] = *probability;
                });

            let state_tensor = board.get_state_tensor();

            get_equivalents(&state_tensor, &mcts_probs_matrix)
                .into_iter()
                .for_each(|(equi_state_tensor, equi_prob_matrix)| {
                    self.data_buffer.push(TrainDataItem(
                        equi_state_tensor,
                        equi_prob_matrix,
                        winner,
                    ));
                });
            self.data_buffer
                .push(TrainDataItem(state_tensor, mcts_probs_matrix, winner));
        }
    }

    fn pick_move(
        self: &Self,
        board: &mut RenjuBoard,
        move_prob_pairs: &Vec<((usize, usize), f32)>,
    ) -> (usize, usize) {
        assert!(!move_prob_pairs.is_empty());

        let mut pairs = Cow::from(move_prob_pairs);
        while !pairs.is_empty() {
            // determine the position to move
            // For self-player, choose a position by probabilities with dirichlet noice
            let mut probability_tensor = Tensor::<f32>::new(&[pairs.len() as u64]);
            pairs
                .iter()
                .enumerate()
                .for_each(|(index, (_, probability))| {
                    probability_tensor[index] = *probability;
                });
            let index = self
                .model
                .borrow()
                .random_choose_with_dirichlet_noice(&probability_tensor)
                .expect("random_choose_with_dirichlet_noice failed");

            let pos = pairs[index].0;
            if !board.is_forbidden(pos) {
                return pos;
            }
            pairs.to_mut().swap_remove(index); // remove the forbidden position and try again
        }
        move_prob_pairs[0].0
    }

    pub fn self_play(self: &mut Self) -> SelfPlayerOutcome {
        let mut board = RenjuBoard::default();
        let mut searcher = TreeSearcher::new(5f32, 1000u32, self.model.clone());

        let capacity = board.width() * board.height();
        let mut board_state_list = Vec::<RenjuBoard>::with_capacity(capacity);
        let mut mcts_probs_list = Vec::<_>::with_capacity(capacity);
        let mut player_list = Vec::<bool>::with_capacity(capacity); // true if black turn

        let mut ts = TerminalState::default();
        loop {
            let move_prob_pairs: Vec<((usize, usize), f32)> =
                searcher.get_move_probability(&board, ts, self.temperature);

            let pos = self.pick_move(&mut board, &move_prob_pairs);
            searcher.update_with_position(pos);

            // save board state
            board_state_list.push(board.clone());
            mcts_probs_list.push(move_prob_pairs);
            player_list.push(board.is_black_turn());

            match board.do_move(pos) {
                TerminalState::AvailableMoves(vector) => {
                    ts = TerminalState::AvailableMoves(vector);
                    continue;
                }
                s => {
                    board.print();
                    let winner_list: Vec<f32> = player_list
                        .iter()
                        .map(|is_black_turn| match s {
                            TerminalState::BlackWon if *is_black_turn => 1f32,
                            TerminalState::WhiteWon if !*is_black_turn => 1f32,
                            TerminalState::WhiteWon if *is_black_turn => -1f32,
                            TerminalState::BlackWon if !*is_black_turn => -1f32,
                            _ => 0f32,
                        })
                        .collect();

                    return SelfPlayerOutcome(board_state_list, mcts_probs_list, winner_list);
                }
            }
        }
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

fn get_equivalents(
    state_tensor: &[[[f32; game::BOARD_SIZE]; game::BOARD_SIZE]; 4],
    probability_tensor: &SquaredMatrix,
) -> Vec<([[[f32; 15]; 15]; 4], SquaredMatrix)> {
    let mut state_tensor1 = state_tensor.clone();
    let mut probability_tensor1 = probability_tensor.clone();

    state_tensor1[0].flip_over_main_diagonal();
    state_tensor1[1].flip_over_main_diagonal();
    state_tensor1[2].flip_over_main_diagonal();
    probability_tensor1.flip_over_main_diagonal();

    let mut state_tensor2 = state_tensor1.clone();
    let mut probability_tensor2 = probability_tensor1.clone();
    state_tensor2[0].flip_top_bottom();
    state_tensor2[1].flip_top_bottom();
    state_tensor2[2].flip_top_bottom();
    probability_tensor2.flip_top_bottom();

    let mut state_tensor3 = state_tensor.clone();
    let mut probability_tensor3 = probability_tensor.clone();
    state_tensor3[0].flip_over_anti_diagonal();
    state_tensor3[1].flip_over_anti_diagonal();
    state_tensor3[2].flip_over_anti_diagonal();
    probability_tensor3.flip_over_anti_diagonal();

    let mut state_tensor4 = state_tensor3.clone();
    let mut probability_tensor4 = probability_tensor3.clone();
    state_tensor4[0].flip_top_bottom();
    state_tensor4[1].flip_top_bottom();
    state_tensor4[2].flip_top_bottom();
    probability_tensor4.flip_top_bottom();

    let mut state_tensor5 = state_tensor.clone();
    let mut probability_tensor5 = probability_tensor.clone();
    state_tensor5[0].flip_left_right();
    state_tensor5[1].flip_left_right();
    state_tensor5[2].flip_left_right();
    probability_tensor5.flip_left_right();

    let mut state_tensor6 = state_tensor.clone();
    let mut probability_tensor6 = probability_tensor.clone();
    state_tensor6[0].flip_top_bottom();
    state_tensor6[1].flip_top_bottom();
    state_tensor6[2].flip_top_bottom();
    probability_tensor6.flip_top_bottom();

    let mut state_tensor7 = state_tensor6.clone();
    let mut probability_tensor7 = probability_tensor6.clone();
    state_tensor7[0].flip_left_right();
    state_tensor7[1].flip_left_right();
    state_tensor7[2].flip_left_right();
    probability_tensor7.flip_left_right();

    vec![
        (state_tensor1, probability_tensor1),
        (state_tensor2, probability_tensor2),
        (state_tensor3, probability_tensor3),
        (state_tensor4, probability_tensor4),
        (state_tensor5, probability_tensor5),
        (state_tensor6, probability_tensor6),
        (state_tensor7, probability_tensor7),
    ]
}
