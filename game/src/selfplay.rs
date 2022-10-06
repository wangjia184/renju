use bytemuck::cast_slice;
use bytes::Bytes;
use ndarray::Array;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::time::Instant;
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;

use futures::future::OptionFuture;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand_distr::Dirichlet;
use tokio::task::JoinHandle;

use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};

use crate::game::*;
use crate::mcts::{MonteCarloTree, PredictionPromise};
use crate::model::PolicyValueModel;

pub struct Trainer {
    parallel_num: u32, // parallel self-play matches for a single open pattern
    batch_size: usize,
    mcts_c_puct: f32, // https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5
    mcts_iterations: usize, // rollout iterations in each move
    epochs: usize,    // num of train_steps for each update
    kl_targ: f32,
    learn_rate: f32,
    lr_multiplier: f32, // adaptively adjust the learning rate based on KL
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DataSet {
    state_tensor_batch: Vec<StateTensor>,
    prob_matrix_batch: Vec<SquareMatrix>,
    score_batch: Vec<f32>,
}

impl Trainer {
    pub fn new() -> Self {
        Self {
            batch_size: 500,
            parallel_num: 10, // parallel self-play matches for a single open pattern
            mcts_c_puct: 1f32,
            mcts_iterations: 10,
            epochs: 5,
            learn_rate: 1e-3,
            lr_multiplier: 1f32,
            kl_targ: 0.02f32,
        }
    }

    pub async fn run(self: &mut Self) {
        let mut option = Some(PolicyValueModel::get_latest());
        loop {
            assert!(option.is_some());
            let (handles, predict_rx) = self.start_groups();

            // handle predictions and wait all complection
            let mut model = option.take().unwrap();
            model = self.handle_prediction(model, predict_rx).await;

            // collect data
            let mut match_outputs: Vec<Vec<(StateTensor, SquareMatrix, f32)>> =
                Vec::with_capacity(1000);
            for handle in handles {
                let mut vectors: Vec<Vec<(StateTensor, SquareMatrix, f32)>> =
                    handle.await.expect("Error in TrainGroup");
                match_outputs.append(&mut vectors);
            }
            println!("Total matches = {}", match_outputs.len());
            let data_sets = self.collect_data_set(match_outputs);
            println!(
                "{} batches. each size is {}",
                data_sets.len(),
                self.batch_size
            );

            option = Some(self.train(model, data_sets).await);
        }
    }

    fn start_groups(
        self: &Self,
    ) -> (
        Vec<JoinHandle<Vec<Vec<(StateTensor, SquareMatrix, f32)>>>>,
        UnboundedReceiver<PredictionPromise>,
    ) {
        // https://www.wuziqi123.com/jiangzuo/dingshiyanjiu/156.html
        // for each open pattern, start a group

        let (predict_tx, predict_rx) = mpsc::unbounded_channel();

        let mut handles = Vec::with_capacity(100);

        let mut open_patterns = Vec::with_capacity(26);

        let first_black = (BOARD_SIZE / 2, BOARD_SIZE / 2);
        let first_white = (first_black.0 - 1, first_black.1);
        for row in (BOARD_SIZE / 2 - 2)..=(BOARD_SIZE / 2 + 2) {
            for col in (BOARD_SIZE / 2)..=(BOARD_SIZE / 2 + 2) {
                let second_black = (row, col);
                if second_black != first_black && second_black != first_white {
                    open_patterns.push([first_black, first_white, second_black]);
                }
            }
        }

        let first_black = (BOARD_SIZE / 2, BOARD_SIZE / 2);
        let first_white = (first_black.0 - 1, first_black.1 + 1);
        for row in (BOARD_SIZE / 2 - 2)..=(BOARD_SIZE / 2 + 2) {
            for col in (BOARD_SIZE / 2 + 2 - (row - (BOARD_SIZE / 2 - 2)))..=(BOARD_SIZE / 2 + 2) {
                let second_black = (row, col);
                if second_black != first_black && second_black != first_white {
                    open_patterns.push([first_black, first_white, second_black]);
                }
            }
        }

        for open_pattern in open_patterns {
            let cloned_tx = predict_tx.clone();
            let group = SelfPlayGroup::new(
                open_pattern,
                self.mcts_c_puct,
                self.mcts_iterations,
                cloned_tx,
            );
            let parallel_num = self.parallel_num;
            let handle = tokio::spawn(async move { group.run(parallel_num).await });
            handles.push(handle);
        }

        (handles, predict_rx)
    }

    fn predict_batch(
        model: PolicyValueModel,
        promises: Vec<PredictionPromise>,
    ) -> JoinHandle<PolicyValueModel> {
        let state_tensor_batch: Vec<StateTensor> = promises
            .iter()
            .map(|promise| promise.get_state_tensor().clone())
            .collect();

        tokio::task::spawn_blocking(move || {
            let now = Instant::now();

            let pairs = model
                .predict_batch(state_tensor_batch)
                .expect("Unable to predict batch");

            let elapsed = now.elapsed().as_millis();
            let batch_size = promises.len();

            promises
                .into_iter()
                .enumerate()
                .for_each(|(index, promise)| promise.resolve(pairs[index].0, pairs[index].1));

            tokio::task::spawn_blocking(move || {
                println!(
                    "avg {:.2} ms batch size {}",
                    batch_size as f32 / elapsed as f32,
                    batch_size,
                );
            });
            model
        })
    }

    async fn handle_prediction(
        self: &Self,
        m: PolicyValueModel,
        mut predict_rx: UnboundedReceiver<PredictionPromise>,
    ) -> PolicyValueModel {
        let mut promises = Some(Vec::with_capacity(1000));

        let mut model = Some(m);
        let mut join_handle: OptionFuture<_> = None.into();

        let mut channel_disconnected = false;
        while !channel_disconnected || !promises.as_ref().unwrap().is_empty() || model.is_none() {
            tokio::select! {
                //biased; //  branches are picked in order of code

                option = &mut join_handle => {
                    if let Some(result) = option {
                        match result {
                            Ok(returned_model) => {
                                model = Some(returned_model);
                                join_handle = if !promises.as_ref().unwrap().is_empty() {
                                    Some(Self::predict_batch( model.take().unwrap(), promises.replace(Vec::with_capacity(1000)).unwrap())).into()
                                } else {
                                    None.into()
                                };
                            },
                            Err(e) => {
                                panic!("Unable to predict batch {}", e);
                            }
                        }

                    }
                }

                result = predict_rx.recv() => {
                    match result {
                        Some(promise) => {
                            assert!(promises.is_some());

                            promises.as_mut().unwrap().push(promise);
                            if model.is_some() { // if model is taken, there is already a predction running
                                join_handle = Some(Self::predict_batch( model.take().unwrap(), promises.replace(Vec::with_capacity(1000)).unwrap())).into();
                            }
                        }
                        None => {
                            // completed
                            channel_disconnected = true;
                        }
                    }
                },


            };
        }

        model.unwrap()
    }

    fn collect_data_set(
        self: &Self,
        mut matches: Vec<Vec<(StateTensor, SquareMatrix, f32)>>,
    ) -> Vec<DataSet> {
        // store data set into temp files
        let mut data_sets = Vec::with_capacity(1000);

        while !matches.is_empty() {
            let mut state_tensor_batch = Vec::with_capacity(self.batch_size);
            let mut prob_matrix_batch = Vec::with_capacity(self.batch_size);
            let mut score_batch = Vec::with_capacity(self.batch_size);

            // randomly pick a batch
            while !matches.is_empty() && state_tensor_batch.len() < self.batch_size {
                let match_index = rand::random::<usize>() % matches.len();
                let random_match = &mut matches[match_index];

                let index = rand::random::<usize>() % random_match.len();
                let (state_tensor, prob_matrix, score) = random_match.swap_remove(index);

                state_tensor_batch.push(state_tensor);
                prob_matrix_batch.push(prob_matrix);
                score_batch.push(score);

                if random_match.is_empty() {
                    matches.swap_remove(match_index);
                }
            }

            let data_set = DataSet {
                state_tensor_batch: state_tensor_batch,
                prob_matrix_batch: prob_matrix_batch,
                score_batch: score_batch,
            };
            assert_eq!(
                data_set.state_tensor_batch.len(),
                data_set.prob_matrix_batch.len()
            );
            assert_eq!(
                data_set.state_tensor_batch.len(),
                data_set.score_batch.len()
            );

            data_sets.push(data_set);
        }
        data_sets
    }

    async fn train(
        self: &mut Self,
        model: PolicyValueModel,
        data_sets: Vec<DataSet>,
    ) -> PolicyValueModel {
        for data_set in data_sets {
            let now = Instant::now();
            let (old_log_prob_matrix, old_value) = model
                .predict(&[data_set.state_tensor_batch[0].clone()], true)
                .expect("Failed to predict");

            let slice: &[f32] = cast_slice(&old_log_prob_matrix);
            let old_log_probs = Array::from_vec(slice.to_vec());
            let old_probs = old_log_probs.mapv(f32::exp);

            let mut kl: f32 = 0f32;
            let mut loss: f32 = 0f32;
            let mut entropy: f32 = 0f32;
            for _ in 0..self.epochs {
                (loss, entropy) = model
                    .train(
                        &data_set.state_tensor_batch,
                        &data_set.prob_matrix_batch,
                        &data_set.score_batch,
                        self.learn_rate * self.lr_multiplier,
                    )
                    .expect("Failed to train");

                let (new_log_prob_matrix, new_value) = model
                    .predict(&[data_set.state_tensor_batch[0].clone()], true)
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

            let parameters = model.export().expect("Unable to export");

            {
                let mut file = OpenOptions::new()
                    .read(false)
                    .write(true)
                    .truncate(true)
                    .create(true)
                    .open("latest.weights")
                    .await
                    .expect("Unable to open file");
                file.write_all(&parameters)
                    .await
                    .expect("Unable to write file");
            }
        }
        model
    }
}

// each group handles a specific open pattern
struct SelfPlayGroup {
    open_pattern: [(usize, usize); 3],
    mcts_c_puct: f32,
    mcts_iterations: usize,
    predict_tx: UnboundedSender<PredictionPromise>,
}

impl SelfPlayGroup {
    fn new(
        open_pattern: [(usize, usize); 3],
        mcts_c_puct: f32,
        mcts_iterations: usize,
        predict_tx: UnboundedSender<PredictionPromise>,
    ) -> Self {
        Self {
            open_pattern: open_pattern,
            mcts_c_puct: mcts_c_puct,
            mcts_iterations: mcts_iterations,
            predict_tx: predict_tx,
        }
    }

    async fn run(self: &Self, parallel_num: u32) -> Vec<Vec<(StateTensor, SquareMatrix, f32)>> {
        let mut handles = Vec::with_capacity(parallel_num as usize);
        for _ in 0..parallel_num {
            let worker = SelfPlayMatch::new(
                self.open_pattern,
                self.mcts_c_puct,
                self.mcts_iterations,
                self.predict_tx.clone(),
            );
            let handle = tokio::spawn(async move { worker.run().await });
            handles.push(handle);
        }
        let mut vector = Vec::with_capacity(parallel_num as usize);
        for handle in handles {
            vector.push(handle.await.expect("Panic in SelfPlayMatch"));
        }
        vector
    }
}

// each train worker work on a single self-play match
struct SelfPlayMatch {
    board: RenjuBoard,
    self_player: SelfPlayer,
    open_pattern: [(usize, usize); 3],
    black_pairs: Vec<(StateTensor, SquareMatrix)>, // state tensor and probability matrix of MCTS for black
    white_pairs: Vec<(StateTensor, SquareMatrix)>,
    mcts_iterations: usize,
}

impl SelfPlayMatch {
    pub fn new(
        open_pattern: [(usize, usize); 3],
        mcts_c_puct: f32,
        mcts_iterations: usize,
        predict_tx: UnboundedSender<PredictionPromise>,
    ) -> Self {
        Self {
            board: RenjuBoard::default(),
            self_player: SelfPlayer::new(mcts_c_puct, predict_tx),
            open_pattern: open_pattern,
            black_pairs: Vec::with_capacity(50),
            white_pairs: Vec::with_capacity(50),
            mcts_iterations: mcts_iterations,
        }
    }
    async fn run(mut self: Self) -> Vec<(StateTensor, SquareMatrix, f32)> {
        // first x moves
        let mut state: TerminalState = TerminalState::default();
        for pos in self.open_pattern {
            state = self.direct_move(pos);
            assert!(!state.is_over());
        }
        self.self_player
            .reset_tree(&self.board, self.board.get_last_move().unwrap());

        let state = self.play_to_end(state).await;
        assert!(state.is_over());

        // collect result
        let mut vector = Vec::with_capacity((self.black_pairs.len() + self.white_pairs.len()) * 8);

        let score = match &state {
            TerminalState::BlackWon => 1f32,
            TerminalState::WhiteWon => -1f32,
            TerminalState::Draw => 0f32,
            _ => unreachable!(),
        };
        self.black_pairs
            .iter()
            .for_each(|(state_tensor, prob_matrix)| {
                get_equivalents(state_tensor, prob_matrix)
                    .into_iter()
                    .for_each(|(equi_state_tensor, equi_prob_matrix)| {
                        vector.push((equi_state_tensor, equi_prob_matrix, score));
                    });
                vector.push((*state_tensor, *prob_matrix, score));
            });
        let score = match &state {
            TerminalState::BlackWon => -1f32,
            TerminalState::WhiteWon => 1f32,
            TerminalState::Draw => 0f32,
            _ => unreachable!(),
        };
        self.white_pairs
            .iter()
            .for_each(|(state_tensor, prob_matrix)| {
                get_equivalents(state_tensor, prob_matrix)
                    .into_iter()
                    .for_each(|(equi_state_tensor, equi_prob_matrix)| {
                        vector.push((equi_state_tensor, equi_prob_matrix, score));
                    });
                vector.push((*state_tensor, *prob_matrix, score));
            });
        vector
    }

    fn direct_move(self: &mut Self, pos: (usize, usize)) -> TerminalState {
        let mut prob_matrix = SquareMatrix::default();
        prob_matrix[pos.0][pos.1] = 1f32;

        // prob_matrix : one-hot encoding probability matrix represents the best move from MCTS
        if self.board.is_black_turn() {
            self.black_pairs
                .push((self.board.get_state_tensor(), prob_matrix));
        } else {
            self.white_pairs
                .push((self.board.get_state_tensor(), prob_matrix));
        }

        self.board.do_move(pos)
    }

    async fn think_then_move(self: &mut Self, choices: &Vec<(usize, usize)>) -> TerminalState {
        assert!(!choices.is_empty());
        let (pos, prob_matrix) = self
            .self_player
            .think(self.board.clone(), choices, self.mcts_iterations)
            .await;

        // prob_matrix : one-hot encoding probability matrix represents the best move from MCTS
        if self.board.is_black_turn() {
            self.black_pairs
                .push((self.board.get_state_tensor(), prob_matrix));
        } else {
            self.white_pairs
                .push((self.board.get_state_tensor(), prob_matrix));
        }

        self.board.do_move(pos)
    }

    async fn play_to_end(self: &mut Self, mut state: TerminalState) -> TerminalState {
        loop {
            if let TerminalState::AvailableMoves(ref choices) = state {
                state = self.think_then_move(choices).await;
                if state.is_over() {
                    return state;
                }
            } else {
                unreachable!();
            }
        }
    }
}

// play with self to produce training data
pub struct SelfPlayer {
    tree: MonteCarloTree,
    sender: UnboundedSender<PredictionPromise>,
    // temperature parameter in (0, 1] controls the level of exploration
    temperature: f32,
    mcts_c_puct: f32,
}

impl SelfPlayer {
    pub fn new(mcts_c_puct: f32, sender: UnboundedSender<PredictionPromise>) -> Self {
        let tree = MonteCarloTree::new(mcts_c_puct);
        Self {
            tree: tree,
            mcts_c_puct: mcts_c_puct,
            temperature: 1e-3,
            sender: sender,
        }
    }

    fn choose_with_dirichlet_noice(probabilities: &Vec<f32>) -> usize {
        assert!(probabilities.len() > 1);
        let concentration = vec![0.3f32; probabilities.len()];
        let dirichlet = Dirichlet::new(&concentration).unwrap();
        let samples = dirichlet.sample(&mut rand::thread_rng());

        assert_eq!(samples.len(), probabilities.len());

        let probabilities: Vec<f32> = probabilities
            .iter()
            .enumerate()
            .map(|(index, prob)| *prob * 0.75 + 0.25 * samples[index])
            .collect();

        let dist = WeightedIndex::new(&probabilities).unwrap();

        let mut rng = thread_rng();
        dist.sample(&mut rng)
    }

    // For self-player, choose a position by probabilities with dirichlet noice
    fn pick_move(
        self: &Self,
        move_prob_pairs: &Vec<((usize, usize), f32)>,
        board: &mut RenjuBoard,
    ) -> (usize, usize) {
        assert!(!move_prob_pairs.is_empty());

        let mut pairs = Cow::from(move_prob_pairs);
        while pairs.len() > 1 {
            // determine the position to move

            let vector: Vec<_> = pairs.iter().map(|(_, probability)| *probability).collect();

            let index = Self::choose_with_dirichlet_noice(&vector);

            let pos = pairs[index].0;
            if !board.is_forbidden(pos) {
                return pos;
            }
            pairs.to_mut().swap_remove(index); // remove the forbidden position and try again
        }
        move_prob_pairs[0].0
    }

    fn reset_tree(self: &mut Self, _: &RenjuBoard, pos: (usize, usize)) {
        self.tree = MonteCarloTree::new_with_position(pos, self.mcts_c_puct);
    }

    async fn think(
        self: &mut Self,
        mut board: RenjuBoard,
        choices: &Vec<(usize, usize)>,
        iterations: usize,
    ) -> ((usize, usize), SquareMatrix) {
        for _ in 0..iterations {
            self.tree
                .rollout(board.clone(), choices, |promise| {
                    if let Err(_) = self.sender.send(promise) {
                        panic!("Unable to send")
                    }
                })
                .await
                .expect("rollout failed");
        }

        // get the probabilities of next moves
        // basing on visited times of direct children of root node
        let move_prob_pairs: Vec<((usize, usize), f32)> = self
            .tree
            .get_move_probability(self.temperature)
            .await
            .expect("get_move_probability failed()");

        // 15x15 tensor records the probability of each move
        let mut mcts_prob_matrix = SquareMatrix::default();
        move_prob_pairs
            .iter()
            .for_each(|((row, col), probability)| {
                mcts_prob_matrix[*row][*col] = *probability;
            });

        let pos = self.pick_move(&move_prob_pairs, &mut board);

        self.tree
            .update_with_position(pos)
            .await
            .expect("update_with_position() failed");
        (pos, mcts_prob_matrix)
    }
}

fn get_equivalents(
    state_tensor: &StateTensor,
    probability_tensor: &SquareMatrix,
) -> Vec<([[[f32; 15]; 15]; 4], SquareMatrix)> {
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

#[test]
fn test_get_equivalents() {
    let state_tensor: StateTensor<f32> = [
        [
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        ],
        [
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        ],
        [
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        ],
        [
            [0., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        ],
    ];

    let prob_matrix: SquareMatrix<f32> = [
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    ];

    let equivalents = get_equivalents(&state_tensor, &prob_matrix);

    let state_tensor_0: StateTensor<f32> = [
        [
            [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        ],
        [
            [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        ],
        [
            [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        ],
        [
            [0., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        ],
    ];

    let prob_matrix_0: SquareMatrix<f32> = [
        [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    ];
    assert_eq!(state_tensor_0, equivalents[0].0);
    assert_eq!(prob_matrix_0, equivalents[0].1);

    let matrix1: SquareMatrix<f32> = [
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    ];

    assert_eq!(matrix1, equivalents[1].0[0]);
    assert_eq!(matrix1, equivalents[1].0[1]);
    assert_eq!(matrix1, equivalents[1].0[2]);
    assert_eq!(matrix1, equivalents[1].1);

    let matrix2: SquareMatrix<f32> = [
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
    ];
    assert_eq!(matrix2, equivalents[2].0[0]);
    assert_eq!(matrix2, equivalents[2].0[1]);
    assert_eq!(matrix2, equivalents[2].0[2]);
    assert_eq!(matrix2, equivalents[2].1);

    let matrix3: SquareMatrix<f32> = [
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    ];
    assert_eq!(matrix3, equivalents[3].0[0]);
    assert_eq!(matrix3, equivalents[3].0[1]);
    assert_eq!(matrix3, equivalents[3].0[2]);
    assert_eq!(matrix3, equivalents[3].1);

    let matrix4: SquareMatrix<f32> = [
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    ];
    assert_eq!(matrix4, equivalents[4].0[0]);
    assert_eq!(matrix4, equivalents[4].0[1]);
    assert_eq!(matrix4, equivalents[4].0[2]);
    assert_eq!(matrix4, equivalents[4].1);

    let matrix5: SquareMatrix<f32> = [
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    ];
    assert_eq!(matrix5, equivalents[5].0[0]);
    assert_eq!(matrix5, equivalents[5].0[1]);
    assert_eq!(matrix5, equivalents[5].0[2]);
    assert_eq!(matrix5, equivalents[5].1);

    let matrix6: SquareMatrix<f32> = [
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    ];
    assert_eq!(matrix6, equivalents[6].0[0]);
    assert_eq!(matrix6, equivalents[6].0[1]);
    assert_eq!(matrix6, equivalents[6].0[2]);
    assert_eq!(matrix6, equivalents[6].1);
}
