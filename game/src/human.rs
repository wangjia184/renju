use crate::game::{RenjuBoard, SquareMatrix, TerminalState};

use crate::model::{OnDeviceModel, RenjuModel};
use crate::player::AiPlayer;

use std::sync::atomic::Ordering;
use tokio::sync::RwLock;

use std::sync::{atomic::AtomicBool, Arc, Mutex};
use tokio::task::JoinHandle;

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
    ai_player: Arc<AiPlayer<OnDeviceModel>>,
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
        let model = OnDeviceModel::load("renju_15x15_model").expect("Unable to load saved model");

        let ai_player = AiPlayer::new(model, 0 /*we will rollout manually */);
        let max_threads = num_cpus::get();
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
            visited: self.ai_player.get_visit_times(),
        }
    }

    async fn think<T>(
        board: RenjuBoard,
        choices: Vec<(usize, usize)>,
        thinking: Arc<AtomicBool>,
        ai_player: Arc<AiPlayer<T>>,
    ) where
        T: RenjuModel + Send,
    {
        while thinking.load(Ordering::SeqCst) {
            ai_player.rollout(board.clone(), &choices).await;
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
