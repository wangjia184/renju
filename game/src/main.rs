#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]
extern crate clap;
extern crate num_cpus;
extern crate tensorflow;
use bytes::Bytes;
use clap::{Parser, Subcommand};

use futures::SinkExt;
use futures_util::stream::StreamExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::process::{Child, Command};
use tokio::sync::mpsc::{self, UnboundedSender};
use tokio::sync::watch::{self, Receiver};
use tokio::time::{sleep, Duration};
use tokio_util::codec::{FramedRead, FramedWrite, LengthDelimitedCodec};

mod contest;
mod game;
mod mcts;
mod player;
mod train;

use train::{DataProducer, TrainDataItem, Trainer};
mod human;
mod model;

use human::MatchState;

static ABOUT_TEXT: &str = "Renju game ";

static PRODUCE_MATCH_HELP_TEXT: &str = "
Produce matches by self-play
";

static TRAIN_HELP_TEXT: &str = "
Train the model by reinforcement learning
";

/// Renju Game
#[derive(Parser, Debug)]
#[clap(author, version, about = ABOUT_TEXT, long_about = Some(ABOUT_TEXT), trailing_var_arg=true)]
struct Arguments {
    #[clap(subcommand)]
    verb: Option<Verb>,
}

#[derive(Subcommand, Debug)]
enum Verb {
    /// Producing matches
    #[clap(after_help=PRODUCE_MATCH_HELP_TEXT)]
    Produce {
        /// tcp endpoint to connect to report results. e.g. tcp://127.0.0.1:2222
        #[clap(required = true)]
        address: String,
    },

    /// Exporting matches to dataset for training
    #[clap(after_help=TRAIN_HELP_TEXT)]
    Train {
        /// tcp port to listen on
        #[clap(default_value_t = 55590)]
        port: u16,
    },

    /// contest between two model
    #[clap()]
    Contest {
        /// old model name
        #[clap(required = true)]
        old_model: String,

        /// new model name
        #[clap(required = true)]
        new_model: String,
    },
}

#[tokio::main(flavor = "multi_thread", worker_threads = 20)]
async fn main() {
    let args = Arguments::parse();

    match args.verb {
        Some(Verb::Produce { address }) => {
            println!("Connecting to {}", address);
            let mut stream = TcpStream::connect(address)
                .await
                .expect("Unable to connect to trainer");
            println!("Connection is established");
            let (r, w) = stream.split();

            let (mut producer, mut rx, latest_parameters) = DataProducer::new();

            let mut join_handle = tokio::task::spawn_blocking(move || {
                producer.run();
            });

            // send data set back to trainer
            let mut output_stream = FramedWrite::new(w, LengthDelimitedCodec::new());

            let mut input_stream = FramedRead::new(r, LengthDelimitedCodec::new());

            loop {
                tokio::select! {
                    result = input_stream.next() => {
                        match result {
                            Some(read) => {
                                match read {
                                    Ok(data) => {
                                        *latest_parameters.lock().unwrap() = Some(data.freeze());
                                        //println!("Model parameters have been updated");
                                    }
                                    Err(e) => {
                                        eprintln!("Connection is lost: {}", e);
                                        return;
                                    }
                                }
                            },
                            None => {
                                eprintln!("Connection is closed");
                                return
                            }
                        }
                    }
                    option = rx.recv() => {
                        match option {
                            Some(item) => {
                                let payload : Bytes = item.into();
                                if let Err(e) = output_stream.send(payload).await {
                                    eprintln!("Failed to send data back to trainer. {}", e);
                                    return;
                                }
                            },
                            None => {
                                eprintln!("Channel is closed");
                                return
                            }
                        }
                    }
                    _ = &mut join_handle => {
                        println!("Data producer exited");
                        return;
                    }
                }
            }
        }

        Some(Verb::Train { port }) => {
            let (train_data_tx, train_data_rx) = mpsc::unbounded_channel::<TrainDataItem>();
            let (params_tx, params_rx) = watch::channel(Bytes::new());
            // start tcp server, payload_tx is used to broadcast model parameters
            let mut server = TcpServer::start(port, train_data_tx, params_rx)
                .await
                .expect("Unable to start TCP server");

            let number = if num_cpus::get() > 2 {
                num_cpus::get() - 2
            } else {
                1
            };
            let address = format!("127.0.0.1:{}", port);
            let mut children = Vec::with_capacity(number);
            for _ in 0..number {
                let child = start_child_process(vec!["produce", &address])
                    .expect("Unable to start child process");
                children.push(child);
            }

            let _join_handle = tokio::spawn(async move {
                let mut trainer = Trainer::new(params_tx, train_data_rx);
                trainer.run().await;
            });

            // TODO : multiple wait
            server.run().await.expect("Unable to accept connection");
            drop(children);
        }

        Some(Verb::Contest {
            old_model,
            new_model,
        }) => contest::run(&old_model, &new_model),

        _ => {
            tauri::Builder::default()
                .invoke_handler(tauri::generate_handler![new_match, do_move])
                .run(tauri::generate_context!())
                .expect("error while running tauri application");
        }
    }

    println!("Exiting...");
}

fn start_child_process(parameters: Vec<&str>) -> std::io::Result<Child> {
    let path = std::env::current_exe()?;
    let app_path = path.display().to_string();
    println!("{} {}", app_path, parameters.join(" "));
    let child = Command::new(&app_path)
        .args(parameters)
        //.stdin(Stdio::piped())
        //.stdout(Stdio::piped())
        //.stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()?;

    let _pid = child.id().expect("child did not have a id");

    Ok(child)
}

struct TcpServer {
    listener: TcpListener,
    data_sender: UnboundedSender<TrainDataItem>,
    payload_receiver: Receiver<Bytes>,
}

impl TcpServer {
    pub async fn start(
        port: u16,
        train_data_tx: UnboundedSender<TrainDataItem>,
        payload_rx: Receiver<Bytes>,
    ) -> std::io::Result<Self> {
        let address = format!("0.0.0.0:{}", port);
        let listener = TcpListener::bind(&address).await?;

        Ok(Self {
            listener: listener,
            data_sender: train_data_tx,
            payload_receiver: payload_rx,
        })
    }

    pub async fn run(self: &mut Self) -> std::io::Result<()> {
        loop {
            let (socket, _) = self.listener.accept().await?;
            let rx = self.payload_receiver.clone();
            let tx = self.data_sender.clone();
            tokio::spawn(async move {
                TcpServer::process(socket, rx, tx).await;
            });
        }
    }

    async fn process(
        mut stream: TcpStream,
        mut rx: Receiver<Bytes>,
        tx: UnboundedSender<TrainDataItem>,
    ) {
        let (r, w) = stream.split();
        let mut output_stream = FramedWrite::new(w, LengthDelimitedCodec::new());

        // send latest model parameters
        let payload = rx.borrow().clone();
        output_stream.send(payload).await.expect("Failed to send");

        let mut input_stream = FramedRead::new(r, LengthDelimitedCodec::new());
        loop {
            tokio::select! {
                result = input_stream.next() => {
                    match result {
                        Some(read) => {
                            match read {
                                Ok(data) => {
                                    let item = TrainDataItem::from(data.freeze());
                                    if let Err(e) = tx.send(item) {
                                        eprintln!("Failed to forward data item. {}", e);
                                        return;
                                    }
                                }
                                Err(e) => {
                                    eprintln!("Connection is lost: {}", e);
                                    return;
                                }
                            }
                        }
                        None => {
                            println!("Connection is closed");
                            return;
                        }
                    }
                }

                // Waits for a change notification, then marks the newest value as seen.
                status = rx.changed() => {
                    match status {
                        Ok(_) => {
                            let payload = rx.borrow().clone();
                            if let Err(e) = output_stream.send(payload).await {
                                eprintln!("Unable to send payload. {}", e);
                                return;
                            }
                        },
                        Err(e) => {
                            eprintln!("Unexpected error. {}", e);
                            return;
                        }
                    }
                }
            }
        }
    }
}

#[tauri::command]
async fn new_match(window: tauri::Window, black: bool) {
    let board_info = human::start_new_match(black).await;

    window.emit("board_updated", board_info).unwrap();
}

#[tauri::command]
async fn do_move(window: tauri::Window, pos: (usize, usize)) -> MatchState {
    let board_info = human::human_move(pos).await;

    let state = board_info.get_state();

    let seconds = match board_info.get_stones() {
        0..=3 => 3,
        4..=6 => 7,
        7..=14 => 10,
        _ => 12,
    };

    window.emit("board_updated", board_info).unwrap();

    if state != MatchState::HumanWon && state != MatchState::Draw && state != MatchState::MachineWon
    {
        sleep(Duration::from_secs(seconds)).await;

        let board_info = human::machine_move().await;

        window.emit("board_updated", board_info).unwrap();
    }

    state
}
