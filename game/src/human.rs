use crate::game::{RenjuBoard, SquareMatrix, StateTensor, TerminalState, BOARD_SIZE};

use crate::mcts::MonteCarloTree;
use crate::model::{OnnxModel, TfLiteModel};

use crossbeam::atomic::AtomicCell;

use std::sync::atomic::Ordering;
use tokio::sync::RwLock;

use std::cell::RefCell;
use std::sync::{atomic::AtomicBool, Arc, Mutex};
use tokio::task::JoinHandle;

use rand::seq::SliceRandom;

static SINGLETON: Mutex<Option<Arc<RwLock<HumanVsMachineMatch>>>> = Mutex::new(None);

pub async fn start_new_match(human_play_black: bool) -> BoardInfo {
    let instance = HumanVsMachineMatch::new(human_play_black).await;
    *SINGLETON.lock().unwrap() = Some(instance.clone());
    if !human_play_black {
        return machine_move().await;
    }
    {
        let lock = instance.read().await;
        lock.get_board()
    }
}
fn get_match() -> Arc<RwLock<HumanVsMachineMatch>> {
    if let Some(ref x) = *SINGLETON.lock().unwrap() {
        return x.clone();
    }
    unreachable!()
}

pub async fn get_state() -> MatchState {
    let instance = get_match();
    let lock = instance.read().await;
    lock.get_board().get_state()
}

pub async fn human_move(pos: (usize, usize)) -> BoardInfo {
    let instance = get_match();
    HumanVsMachineMatch::human_move(instance.clone(), pos).await;
    {
        let lock = instance.read().await;
        lock.get_board()
    }
}

pub async fn machine_move() -> BoardInfo {
    let instance = get_match();
    HumanVsMachineMatch::machine_move(instance.clone()).await;
    {
        let lock = instance.read().await;
        lock.get_board()
    }
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

    pub fn get_state(self: &Self) -> MatchState {
        self.state
    }
}

impl HumanVsMachineMatch {
    pub async fn new(human_play_black: bool) -> Arc<RwLock<Self>> {
        let ai_player = AiPlayer::new();
        let max_threads = 1; //(num_cpus::get() - 0).max(1);
        let instance = Arc::new(RwLock::new(Self {
            ai_player: Arc::new(ai_player),
            board: RenjuBoard::default(),
            human_play_black: human_play_black,
            state: if human_play_black {
                MatchState::HumanThinking
            } else {
                MatchState::MachineThinking
            },
            choices: if human_play_black {
                Vec::new()
            } else {
                vec![(7, 7)]
            },
            max_threads: max_threads as u32,
            thinking: Arc::new(AtomicBool::new(!human_play_black)),
            handles: Vec::with_capacity(max_threads),
        }));

        if !human_play_black {
            Self::start_thinking(instance.clone()).await;
        }

        instance
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

    async fn start_thinking(this: Arc<RwLock<Self>>) {
        let thread_num = {
            let instance = this.read().await;
            instance.thinking.store(true, Ordering::SeqCst);
            instance.max_threads
        };

        for _ in 0..thread_num {
            let (board, choices, thinking, ai_player) = {
                let instance = this.read().await;
                (
                    instance.board.clone(),
                    instance.choices.clone(),
                    instance.thinking.clone(),
                    instance.ai_player.clone(),
                )
            };
            let handle = tokio::spawn(async move {
                Self::think(board, choices, thinking, ai_player).await;
            });
            this.write().await.handles.push(handle);
        }
    }

    async fn stop_thinking(this: Arc<RwLock<Self>>) {
        {
            let instance = this.read().await;
            instance.thinking.store(false, Ordering::SeqCst);
        }

        let mut instance = this.write().await;

        while let Some(handle) = instance.handles.pop() {
            _ = handle.await;
        }
    }

    async fn human_move(this: Arc<RwLock<Self>>, pos: (usize, usize)) -> MatchState {
        Self::stop_thinking(this.clone()).await;
        let state = {
            let mut instance = this.write().await;

            assert_eq!(instance.state, MatchState::HumanThinking);
            instance
                .ai_player
                .notify_opponent_moved(&instance.board, pos)
                .await;
            instance.state = match instance.board.do_move(pos) {
                TerminalState::AvailableMoves(choices) => {
                    instance.choices = choices;
                    MatchState::MachineThinking
                }
                TerminalState::BlackWon if instance.human_play_black => MatchState::HumanWon,
                TerminalState::BlackWon if !instance.human_play_black => MatchState::MachineWon,
                TerminalState::WhiteWon if instance.human_play_black => MatchState::MachineWon,
                TerminalState::WhiteWon if !instance.human_play_black => MatchState::HumanWon,
                TerminalState::Draw => MatchState::Draw,
                _ => unreachable!(),
            };
            instance.state
        };

        if state == MatchState::MachineThinking {
            Self::start_thinking(this).await;
        }
        state
    }

    async fn machine_move(this: Arc<RwLock<Self>>) -> MatchState {
        Self::stop_thinking(this.clone()).await;
        let state = {
            let mut instance = this.write().await;

            assert_eq!(instance.state, MatchState::MachineThinking);
            let pos = instance
                .ai_player
                .do_next_move(&instance.board, &instance.choices)
                .await;
            instance.state = match instance.board.do_move(pos) {
                TerminalState::AvailableMoves(choices) => {
                    instance.choices = choices;
                    MatchState::HumanThinking
                }
                TerminalState::BlackWon if instance.human_play_black => MatchState::HumanWon,
                TerminalState::BlackWon if !instance.human_play_black => MatchState::MachineWon,
                TerminalState::WhiteWon if instance.human_play_black => MatchState::MachineWon,
                TerminalState::WhiteWon if !instance.human_play_black => MatchState::HumanWon,
                TerminalState::Draw => MatchState::Draw,
                _ => unreachable!(),
            };
            instance.state
        };

        if state == MatchState::HumanThinking {
            let cloned_insance = this.clone();
            Self::start_thinking(cloned_insance).await;
        }

        state
    }
}

// each thread creates a dedicated model
// thread_local!(static MODEL: RefCell<Option<OnnxModel>> = RefCell::new(None));

use unsafe_send_sync::UnsafeSendSync;

lazy_static! {
    pub static ref MODEL: UnsafeSendSync<OnnxModel> =
        UnsafeSendSync::new(OnnxModel::load("model.onnx"));
}

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
        let tree = MonteCarloTree::new(5f32);
        Self {
            tree: tree,
            temperature: 1e-3,
            visit_time_matrix: AtomicCell::new(SquareMatrix::default()),
        }
    }

    pub async fn think(self: &Self, board: RenjuBoard, choices: &Vec<(usize, usize)>) {
        self.tree
            .rollout(board, choices, |state_tensor: StateTensor| {
                MODEL.predict_one(state_tensor)
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
