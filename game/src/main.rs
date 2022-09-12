extern crate clap;
extern crate num_cpus;
use bytes::Bytes;
use clap::{Args, Parser, Subcommand};

use futures::SinkExt;
use futures_util::stream::StreamExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::process::{Child, Command};
use tokio::sync::mpsc::{self, UnboundedSender};
use tokio::sync::watch::{self, Receiver};
use tokio_util::codec::{FramedRead, FramedWrite, LengthDelimitedCodec};
mod game;
mod mcts;
mod player;
mod storage;
mod train;
use game::{RenjuBoard, SquaredMatrix, SquaredMatrixExtension, StateTensor, TerminalState};
use mcts::MonteCarloTree;
use player::{Match, SelfPlayer};
use train::{DataProducer, TrainDataItem, Trainer};
mod model;
use model::{PolicyValueModel, RenjuModel};

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
    /*
    /// The root folder on local filesystem to scan
    #[clap(display_order=1, short, long, default_value_t = utils::get_current_dir())]
    folder: String,

    /// Regular expression to include files. Only files whose name matches the specified regexp will be listed. E.g. to only list *.dll and *.exe files, you can specify `((\\.dll)$)|((\\.exe)$)`
    #[clap(display_order=2, short, long, default_value(".+"))]
    include: String,

    /// Regular expression to exclude files and directories. Files or directories whose name matches this regexp will be skipped
    #[clap(display_order=3, short, long, default_value("(^~)|(^\\.)|((\\.tmp)$)|((\\.log)$)"))]
    exclude: String,

    /// Max number of recursive folders to scan
    #[clap(display_order=4, short, long, default_value_t = 5)]
    depth: u8,

    /// Optional parameter. When a file path and name is supplied, file list is stored into the specified path in CSV format
    #[clap(display_order=5, short, long)]
    output : Option<String>,

    /// For security consideration, application should not run as root-user. This optional paramerer allows to set a non-privileged user. This parameter only works for Linux/Unix/Mac
    #[clap(display_order=6, short, long)]
    user: Option<String>,
     */
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
}

#[tokio::main(flavor = "multi_thread")]
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
                                        println!("Model parameters have been updated");
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

        _ => {}
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
