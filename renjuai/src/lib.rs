extern crate wee_alloc;

// Use `wee_alloc` as the global allocator.
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

mod game;
mod mcts;
mod utils;

use game::{RenjuBoard, SquareMatrix, TerminalState, BOARD_SIZE};
use js_sys::Array;
use mcts::MonteCarloTree;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    fn predict(state_tensor: JsValue) -> Prediction;
    fn console_log(text: &str);
}

#[wasm_bindgen]
pub struct Prediction {
    prob_matrix: SquareMatrix<f32>,
    score: f32, // color in hex code
}

#[wasm_bindgen]
impl Prediction {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            prob_matrix: SquareMatrix::<f32>::default(),
            score: 0f32,
        }
    }

    #[wasm_bindgen(setter)]
    pub fn set_probabilities(&mut self, probabilities: Array) {
        assert_eq!(probabilities.length() as usize, BOARD_SIZE * BOARD_SIZE);
        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                let index = row * BOARD_SIZE + row;
                self.prob_matrix[row][col] =
                    probabilities.get(index as u32).as_f64().unwrap() as f32;
            }
        }
    }

    #[wasm_bindgen(setter)]
    pub fn set_score(&mut self, score: f32) {
        self.score = score;
    }
}

#[wasm_bindgen]
#[derive(PartialEq, Debug, Clone, Copy, serde::Serialize)]
pub enum MatchState {
    HumanThinking,
    MachineThinking,
    Draw,
    HumanWon,
    MachineWon,
}

#[wasm_bindgen]
#[derive(Clone, Copy, serde::Serialize)]
pub struct BoardInfo {
    matrix: SquareMatrix<u8>,
    stones: u8,
    state: MatchState,
    last: Option<(usize, usize)>,
    visited: SquareMatrix<u32>,
}

#[wasm_bindgen]
pub struct Brain {
    tree: MonteCarloTree,
    board: RenjuBoard,
    choices: Vec<(usize, usize)>,
    state: MatchState,
    human_play_black: bool,
    visit_time_matrix: Option<SquareMatrix<u32>>,
}

impl Brain {
    fn get_state(self: &Self) -> JsValue {
        let info = BoardInfo {
            matrix: self.board.get_matrix().clone(),
            stones: self.board.get_stones(),
            state: self.state,
            last: self.board.get_last_move(),
            visited: self.visit_time_matrix.unwrap_or_default(),
        };
        serde_wasm_bindgen::to_value(&info).unwrap()
    }
}

#[wasm_bindgen]
impl Brain {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        utils::set_panic_hook();
        Self {
            tree: MonteCarloTree::new(3f32),
            board: RenjuBoard::default(),
            choices: vec![(7usize, 7usize)],
            state: MatchState::Draw,
            human_play_black: false,
            visit_time_matrix: None,
        }
    }

    pub fn reset(&mut self, human_play_black: bool) -> JsValue {
        self.tree = MonteCarloTree::new(5f32);
        self.board = RenjuBoard::default();
        self.choices = vec![(7usize, 7usize)];
        self.state = if human_play_black {
            MatchState::HumanThinking
        } else {
            MatchState::MachineThinking
        };
        self.human_play_black = human_play_black;
        self.get_state()
    }

    pub async fn human_move(&mut self, row: u32, col: u32) -> JsValue {
        if self.state == MatchState::HumanThinking {
            let pos = (row as usize, col as usize);
            self.tree
                .update_with_position(pos)
                .await
                .expect("update_with_position failed");
            self.state = match self.board.do_move(pos) {
                TerminalState::AvailableMoves(choices) => {
                    self.choices = choices;
                    MatchState::MachineThinking
                }
                TerminalState::BlackWon if self.human_play_black => MatchState::HumanWon,
                TerminalState::BlackWon if !self.human_play_black => MatchState::MachineWon,
                TerminalState::WhiteWon if self.human_play_black => MatchState::MachineWon,
                TerminalState::WhiteWon if !self.human_play_black => MatchState::HumanWon,
                TerminalState::Draw => MatchState::Draw,
                _ => unreachable!(),
            };
        }

        self.get_state()
    }

    pub async fn think(&self, iterations: u32) {
        for _ in 0..iterations {
            self.tree
                .rollout(
                    self.board.clone(),
                    &self.choices,
                    |state_tensor| -> (SquareMatrix<f32>, f32) {
                        let input = serde_wasm_bindgen::to_value(&state_tensor).unwrap();
                        let prediction = predict(input);
                        return (prediction.prob_matrix, prediction.score);
                    },
                )
                .await
                .expect("rollout failed")
        }
    }

    pub async fn machine_move(&mut self) -> JsValue {
        if self.state == MatchState::MachineThinking {
            let pos = if self.choices.len() == 1 {
                self.choices[0]
            } else {
                let move_prob_pairs: Vec<((usize, usize), f32)> = self
                    .tree
                    .get_move_probability(1f32)
                    .await
                    .expect("get_move_probability() failed");

                match self.board.get_stones() {
                    /*
                    1..=2 if self.board.get_matrix()[BOARD_SIZE / 2][BOARD_SIZE / 2] == 1 => {
                        move_prob_pairs.choose(&mut rand::thread_rng()).unwrap().0
                    } */
                    _ => {
                        let pair = move_prob_pairs
                            .into_iter()
                            .max_by(|(_, left_score), (_, right_score)| {
                                left_score
                                    .partial_cmp(right_score)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .expect("At least one pair");
                        pair.0
                    }
                }
            };

            let visit_time_matrix = self
                .tree
                .get_visit_times()
                .await
                .expect("get_visit_times() failed");
            self.visit_time_matrix = Some(visit_time_matrix);
            self.tree
                .update_with_position(pos)
                .await
                .expect("update_with_position() failed");

            self.state = match self.board.do_move(pos) {
                TerminalState::AvailableMoves(choices) => {
                    self.choices = choices;
                    MatchState::HumanThinking
                }
                TerminalState::BlackWon if self.human_play_black => MatchState::HumanWon,
                TerminalState::BlackWon if !self.human_play_black => MatchState::MachineWon,
                TerminalState::WhiteWon if self.human_play_black => MatchState::MachineWon,
                TerminalState::WhiteWon if !self.human_play_black => MatchState::HumanWon,
                TerminalState::Draw => MatchState::Draw,
                _ => unreachable!(),
            };
        }

        self.get_state()
    }
}
