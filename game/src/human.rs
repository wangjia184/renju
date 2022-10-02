use crate::game::{RenjuBoard, SquareMatrix, TerminalState};

use crate::model::OnDeviceModel;
use crate::player::AiPlayer;

use std::any::Any;

use futures::future::BoxFuture;
use std::sync::Mutex;
use tokio::sync::mpsc::{self, error::TryRecvError, UnboundedReceiver, UnboundedSender};
use tokio::sync::oneshot;

static SENDER: Mutex<
    Option<UnboundedSender<(CallbackFunction, oneshot::Sender<Box<dyn Any + Send>>)>>,
> = Mutex::new(None);

type CallbackFunction =
    Box<dyn FnMut(&mut HumanVsMachineMatch) -> BoxFuture<Box<dyn Any + Send>> + Send>;

pub fn start_new_match(human_play_black: bool) {
    let (tx, rx) = mpsc::unbounded_channel();
    SENDER.lock().unwrap().replace(tx);
    tokio::task::spawn_blocking(move || {
        run(rx, human_play_black);
    });
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
    ai_player: AiPlayer<OnDeviceModel>,
    board: RenjuBoard,
    human_play_black: bool,
    state: MatchState,
    choices: Vec<(usize, usize)>,
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
    pub fn new(human_play_black: bool) -> Self {
        let model = OnDeviceModel::load("renju_15x15_model").expect("Unable to load saved model");

        let ai_player = AiPlayer::new(model, 0 /*we will rollout manually */);
        let instance = Self {
            ai_player: ai_player,
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
        };

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

    pub async fn rollout(self: &mut Self) {
        self.ai_player
            .rollout(self.board.clone(), &self.choices)
            .await
    }

    pub async fn human_move(self: &mut Self, pos: (usize, usize)) -> MatchState {
        assert_eq!(self.state, MatchState::HumanThinking);
        self.ai_player.notify_opponent_moved(&self.board, pos).await;
        match self.board.do_move(pos) {
            TerminalState::AvailableMoves(choices) => {
                self.choices = choices;
                self.state = MatchState::MachineThinking;
            }
            TerminalState::BlackWon => {
                self.state = if self.human_play_black {
                    MatchState::HumanWon
                } else {
                    MatchState::MachineWon
                };
            }
            TerminalState::WhiteWon => {
                self.state = if !self.human_play_black {
                    MatchState::HumanWon
                } else {
                    MatchState::MachineWon
                };
            }
            TerminalState::Draw => {
                self.state = MatchState::Draw;
            }
        };
        self.state
    }

    pub async fn machine_move(self: &mut Self) -> MatchState {
        assert_eq!(self.state, MatchState::MachineThinking);
        assert!(!self.choices.is_empty());

        let pos = self
            .ai_player
            .do_next_move(&mut self.board, &self.choices)
            .await;
        match self.board.do_move(pos) {
            TerminalState::AvailableMoves(choices) => {
                self.choices = choices;
                self.state = MatchState::HumanThinking;
            }
            TerminalState::BlackWon => {
                self.state = if self.human_play_black {
                    MatchState::HumanWon
                } else {
                    MatchState::MachineWon
                };
            }
            TerminalState::WhiteWon => {
                self.state = if !self.human_play_black {
                    MatchState::HumanWon
                } else {
                    MatchState::MachineWon
                };
            }
            TerminalState::Draw => {
                self.state = MatchState::Draw;
            }
        };
        self.state
    }

    pub fn is_thinking(self: &mut Self) -> bool {
        !self.board.get_last_move().is_none()
            && self.state != MatchState::Draw
            && self.state != MatchState::MachineWon
            && self.state != MatchState::HumanWon
    }
}

async fn run(
    mut rx: UnboundedReceiver<(CallbackFunction, oneshot::Sender<Box<dyn Any + Send>>)>,
    human_play_black: bool,
) {
    let mut instance = HumanVsMachineMatch::new(human_play_black);
    loop {
        loop {
            match rx.try_recv() {
                Ok((mut func, reply_tx)) => {
                    let ret = func(&mut instance).await;
                    if let Err(_) = reply_tx.send(ret) {
                        println!("reply_tx.send() failed. run() exited");
                        return; // this thread exits
                    }
                }
                Err(TryRecvError::Disconnected) => {
                    println!("callback channel closed. exiting");
                    return; // this thread exits
                }
                Err(TryRecvError::Empty) => {
                    break;
                }
            }
        }

        if !instance.is_thinking() {
            match rx.blocking_recv() {
                Some((mut func, reply_tx)) => {
                    let ret = func(&mut instance).await;
                    if let Err(_) = reply_tx.send(ret) {
                        println!("reply_tx.send() failed. run() exited");
                        return; // this thread exits
                    }
                }
                _ => return,
            };
        } else {
            instance.rollout().await;
        }
    }
}

pub async fn execute<T>(func: CallbackFunction) -> Box<T>
where
    T: 'static,
{
    let (sender, receiver) = oneshot::channel();
    let tx = SENDER.lock().unwrap().clone();
    let tx = tx.expect("Sender must not be None");
    if let Err(e) = tx.send((func, sender)) {
        panic!("Unable to send callback function on channel. {}", e);
    }
    let ret = receiver.await.expect("Unable to receive");
    ret.downcast::<T>().expect("BUG! Unable to cast")
}
