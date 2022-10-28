use crate::game::{RenjuBoard, SquareMatrix, StateTensor, TerminalState, BOARD_SIZE};

use crate::mcts::MonteCarloTree;
use crate::model::TfLiteModel;

use crossbeam::atomic::AtomicCell;

use std::sync::atomic::Ordering;
use tokio::sync::{Mutex, RwLock};

use rand::seq::SliceRandom;
use std::cell::RefCell;
use std::sync::{atomic::AtomicBool, Arc};
use tokio::sync::oneshot::{self, Receiver};
use tokio::task::JoinHandle;

lazy_static! {
    static ref SINGLETON: Mutex<Option<Arc<RwLock<HumanVsMachineMatch>>>> = Mutex::new(None);
    static ref SINGLETON_CHANNEL: AtomicCell<Receiver<HumanVsMachineMatch>> = {
        let instance = HumanVsMachineMatch::new(true);
        let (tx, rx) = oneshot::channel();
        if let Err(_) = tx.send(instance) {
            unreachable!();
        }
        AtomicCell::new(rx)
    };
}

pub async fn access<Fut, F>(mut f: F) -> MatchState
where
    F: FnMut(HumanVsMachineMatch) -> Fut,
    Fut: std::future::Future<Output = HumanVsMachineMatch>,
{
    let (tx, rx) = oneshot::channel();
    let rx = SINGLETON_CHANNEL.swap(rx);
    let mut instance = rx.await.unwrap();

    instance = f(instance).await;

    let state = instance.state;

    if let Err(_) = tx.send(instance) {
        unreachable!()
    }
    state
}

#[derive(PartialEq, Debug, Clone, Copy, serde::Serialize)]
pub enum MatchState {
    HumanThinking,
    MachineThinking,
    Draw,
    HumanWon,
    MachineWon,
}
pub struct HumanVsMachineMatch {
    ai_player: Arc<AiPlayer>,
    board: RenjuBoard,
    human_play_black: bool,
    state: MatchState,
    choices: Vec<(usize, usize)>,

    thinking: Arc<AtomicBool>,
    max_threads: u32,
    handles: Vec<JoinHandle<()>>,
}

#[derive(Clone, Copy, serde::Serialize)]
pub struct BoardInfo {
    matrix: SquareMatrix<u8>,
    stones: u8,
    state: MatchState,
    last: Option<(usize, usize)>,
    visited: SquareMatrix<u32>,
}

impl BoardInfo {
    pub fn get_stones(self: &Self) -> u8 {
        self.stones
    }
}

impl HumanVsMachineMatch {
    fn new(human_play_black: bool) -> Self {
        let ai_player = AiPlayer::new();
        let max_threads = 1; //(num_cpus::get() - 0).max(1);
        Self {
            ai_player: Arc::new(ai_player),
            board: RenjuBoard::default(),
            human_play_black: human_play_black,
            state: if human_play_black {
                MatchState::HumanThinking
            } else {
                MatchState::MachineThinking
            },
            choices: Vec::new(),
            max_threads: max_threads as u32,
            thinking: Arc::new(AtomicBool::new(!human_play_black)),
            handles: Vec::with_capacity(max_threads),
        }
    }

    pub async fn restart(self: &mut Self, human_play_black: bool) {
        self.stop_thinking().await;
        self.ai_player = Arc::new(AiPlayer::new());
        self.board = RenjuBoard::default();
        self.human_play_black = human_play_black;
        if human_play_black {
            self.state = MatchState::HumanThinking;
            self.choices = Vec::new();
        } else {
            self.state = MatchState::MachineThinking;
            self.choices = vec![(7, 7)];
            self.machine_move().await;
        };
    }

    pub fn get_board(self: &Self) -> BoardInfo {
        BoardInfo {
            matrix: self.board.get_matrix().clone(),
            stones: self.board.get_stones(),
            state: self.state,
            last: self.board.get_last_move(),
            visited: self.ai_player.get_visit_time_matrix(),
        }
    }

    async fn think(
        board: RenjuBoard,
        choices: Vec<(usize, usize)>,
        thinking: Arc<AtomicBool>,
        ai_player: Arc<AiPlayer>,
    ) {
        while thinking.load(Ordering::SeqCst) {
            ai_player.think(board.clone(), &choices).await;
        }
    }

    async fn start_thinking(self: &mut Self) {
        self.thinking.store(true, Ordering::SeqCst);

        for _ in 0..self.max_threads {
            let (board, choices, thinking, ai_player) = {
                (
                    self.board.clone(),
                    self.choices.clone(),
                    self.thinking.clone(),
                    self.ai_player.clone(),
                )
            };

            self.handles.push(tokio::spawn(async move {
                Self::think(board, choices, thinking, ai_player).await;
            }));
        }
    }

    async fn stop_thinking(self: &mut Self) {
        self.thinking.store(false, Ordering::SeqCst);

        while let Some(handle) = self.handles.pop() {
            _ = handle.await;
        }
    }

    pub async fn human_move(self: &mut Self, pos: (usize, usize)) -> MatchState {
        self.stop_thinking().await;

        let state = {
            assert_eq!(self.state, MatchState::HumanThinking);
            self.ai_player.notify_opponent_moved(&self.board, pos).await;
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
            self.state
        };

        if state == MatchState::MachineThinking {
            self.start_thinking().await;
        }
        state
    }

    pub async fn machine_move(self: &mut Self) -> MatchState {
        self.stop_thinking().await;

        let state = {
            assert_eq!(self.state, MatchState::MachineThinking);
            let pos = self
                .ai_player
                .do_next_move(&self.board, &self.choices)
                .await;
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
            self.state
        };

        if state == MatchState::HumanThinking {
            self.start_thinking().await;
        }

        state
    }
}

// each thread creates a dedicated model
thread_local!(static MODEL: RefCell<Option<TfLiteModel>> = RefCell::new(None));

pub struct AiPlayer {
    tree: MonteCarloTree,
    // temperature parameter in (0, 1] controls the level of exploration
    temperature: f32,
    visit_time_matrix: AtomicCell<SquareMatrix<u32>>,
}

impl AiPlayer {
    pub fn get_visit_time_matrix(self: &Self) -> SquareMatrix<u32> {
        self.visit_time_matrix.load()
    }
    pub fn new() -> Self {
        let tree = MonteCarloTree::new(3f32);
        Self {
            tree: tree,
            temperature: 1e-3,
            visit_time_matrix: AtomicCell::new(SquareMatrix::default()),
        }
    }

    pub async fn think(self: &Self, board: RenjuBoard, choices: &Vec<(usize, usize)>) {
        self.tree
            .rollout(board, choices, |state_tensor: StateTensor| {
                MODEL.with(|ref_cell| {
                    let mut model = ref_cell.borrow_mut();
                    if model.is_none() {
                        *model = Some(
                            TfLiteModel::load("best.tflite").expect("Unable to load saved model"),
                        )
                    }
                    model
                        .as_ref()
                        .unwrap()
                        .predict_one(state_tensor)
                        .expect("Unable to predict_one")
                })
            })
            .await
            .expect("rollout failed")
    }

    pub async fn do_next_move(
        self: &Self,
        board: &RenjuBoard,
        choices: &Vec<(usize, usize)>,
    ) -> (usize, usize) {
        let pos = if choices.len() == 1 {
            choices[0]
        } else {
            let move_prob_pairs: Vec<((usize, usize), f32)> = self
                .tree
                .get_move_probability(self.temperature)
                .await
                .expect("get_move_probability() failed");

            match board.get_stones() {
                1..=2 if board.get_matrix()[BOARD_SIZE / 2][BOARD_SIZE / 2] == 1 => {
                    move_prob_pairs.choose(&mut rand::thread_rng()).unwrap().0
                }
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
        self.visit_time_matrix.store(visit_time_matrix);
        self.tree
            .update_with_position(pos)
            .await
            .expect("update_with_position() failed");
        pos
    }

    pub async fn notify_opponent_moved(self: &Self, _: &RenjuBoard, pos: (usize, usize)) {
        self.tree
            .update_with_position(pos)
            .await
            .expect("update_with_position failed");
    }
}
