#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]
extern crate clap;
extern crate num_cpus;

use clap::{Parser, Subcommand};

use tokio::time::{sleep, Duration};

mod contest;
mod game;
mod human;
mod mcts;
mod model;

mod selfplay;
use human::MatchState;
use selfplay::Trainer;

static ABOUT_TEXT: &str = "Renju game ";

static SELF_PLAY_MATCH_HELP_TEXT: &str = "
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
    #[clap(after_help=SELF_PLAY_MATCH_HELP_TEXT)]
    SelfPlay {
        #[clap(required = true)]
        model_file: String,

        #[clap(required = true)]
        export_dir: String,
    },

    /// Self play and train
    #[clap(after_help=TRAIN_HELP_TEXT)]
    Train {},

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

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let args = Arguments::parse();

    match args.verb {
        Some(Verb::SelfPlay {
            model_file,
            export_dir,
        }) => {
            let mut trainer = Trainer::new();

            trainer
                .produce_self_play_data(&model_file, &export_dir)
                .await;
        }

        Some(Verb::Train {}) => {
            let mut trainer = Trainer::new();
            trainer.run().await;
        }

        _ => {
            tauri::Builder::default()
                .invoke_handler(tauri::generate_handler![new_match, do_move])
                .run(tauri::generate_context!())
                .expect("error while running tauri application");
        }
    }

    println!("Exiting...");
}

/*
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
 */
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
