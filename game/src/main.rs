extern crate clap;
extern crate tensorflow;

use clap::{Args, Parser, Subcommand};

use tensorflow::Tensor;

mod game;
mod mcts;
mod storage;
mod train;
use game::{RenjuBoard, SquaredMatrix, SquaredMatrixExtension, StateTensor, TerminalState};
use mcts::TreeSearcher;
use train::Trainer;

mod model;
use model::{PolicyValueModel, RenjuModel};

static ABOUT_TEXT: &str = "Renju game ";

static GENERATE_MATCH_HELP_TEXT: &str = "
Using YiXin.exe to generate match 
";

static EXPORT_DATASET_HELP_TEXT: &str = "
Export matches into training dataset
";

/// File Scanner
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
    /// Generating matches
    #[clap(after_help=GENERATE_MATCH_HELP_TEXT)]
    Generate {
        /// Path to YiXin.exe
        #[clap(required = true)]
        filepath: String,
    },

    /// Exporting matches to dataset for training
    #[clap(after_help=EXPORT_DATASET_HELP_TEXT)]
    Train {},
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let mut trainer = Trainer::new();
    trainer.execute();

    let args = Arguments::parse();

    match args.verb {
        /*
        Some(Verb::Generate { filepath }) => {
            let (tx, rx) = mpsc::unbounded_channel();

            // http://petr.lastovicka.sweb.cz/protocl2en.htm

            //tx.send(("INFO max_node 120000\n".to_string(), None)).unwrap();
            //tx.send(("INFO caution_factor 2\n".to_string(), None)).unwrap();
            tx.send(("info show_detail 1\n".to_string(), None)).unwrap();
            tx.send(("INFO thread_split_depth 5\n".to_string(), None))
                .unwrap();
            tx.send(("INFO max_thread_num 16\n".to_string(), None))
                .unwrap();
            tx.send(("INFO thread_num 16\n".to_string(), None)).unwrap();
            tx.send(("INFO rule 2\n".to_string(), None)).unwrap();
            tx.send(("INFO max_node -1\n".to_string(), None)).unwrap();
            tx.send(("INFO max_depth 100\n".to_string(), None)).unwrap();
            tx.send(("INFO time_increment 0\n".to_string(), None))
                .unwrap();

            tx.send(("START 15 15\n".to_string(), None)).unwrap();
            tx.send(("RESTART\n".to_string(), None)).unwrap();

            tokio::spawn(async move {
                match_routine(tx).await;
            });

            run(&filepath, Vec::new(), rx).await;
        } */
        Some(Verb::Train {}) => {}
        _ => {}
    }

    println!("Exiting...");
}

/*
// https://github.com/accreator/Yixin-Board/blob/609c06015a0a239a3a5365c28ff8c7be96ecef9a/main.c#L5739
async fn run(
    app_path: &str,
    parameters: Vec<String>,
    mut input_rx: UnboundedReceiver<(String, Option<oneshot::Sender<(i8, i8, i32)>>)>,
) {
    let mut child = match Command::new(&app_path)
        .args(parameters)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
    {
        Ok(c) => c,
        Err(e) => panic!("Unable to start process `{}`. {}", &app_path, e),
    };

    let _pid = child.id().expect("child did not have a id");

    let stdout = child
        .stdout
        .take()
        .expect("child did not have a handle to stdout");
    let stderr = child
        .stderr
        .take()
        .expect("child did not have a handle to stderr");

    let mut stdin = child
        .stdin
        .take()
        .expect("child did not have a handle to stdin");
    let mut stdin_writer = BufWriter::new(&mut stdin);

    let mut stdout_reader = BufReader::new(stdout).lines();
    let mut stderr_reader = BufReader::new(stderr).lines();

    let mut handler = ReplyHanlder::new();

    loop {
        tokio::select! {
            result = stdout_reader.next_line() => {
                match result {
                    Ok(Some(line)) => handler.parse_result(line),
                    Err(e) => {
                        eprintln!("Unable to read child process stdout. {}", e);
                        break
                    }
                    _ => (),
                }
            }
            result = stderr_reader.next_line() => {
                match result {
                    Ok(Some(line)) => eprintln!("{}", line),
                    Err(e) => {
                        eprintln!("Unable to read child process stderr. {}", e);
                        break
                    }
                    _ => (),
                }
            }
            result = child.wait() => {
                match result {
                    Ok(exit_code) => println!("{} exited. {}", &app_path, exit_code),
                    Err(e) => eprintln!("Failed to wait child process. {}", e),
                }
                break // child process exited
            },
            option = input_rx.recv() => {
                match option {
                    Some(mut input) => {
                        if let Some(reply_tx) = input.1.take() {
                            handler.add_replier(reply_tx);
                        }

                        if let Err(e) = stdin_writer.write(input.0.as_bytes()).await {
                            eprintln!("Failed to send command. {}", e);
                            break;
                        }
                        if let Err(e) = stdin_writer.flush().await{
                            eprintln!("Failed to flush command. {}", e);
                            break;
                        }
                    },
                    None => {
                        eprintln!("Input stream is closed.");
                        break
                    }
                }
            },
        };
    }
}

async fn match_routine(tx: UnboundedSender<(String, Option<oneshot::Sender<(i8, i8, i32)>>)>) {
    let mut pieces = 10;

    let db = Database::default();
    db.store();

    //tx.send("yxblockreset\n".to_string()).unwrap();
    //tx.send("BOARD\n".to_string()).unwrap();
    //tx.send("0,2,1\n".to_string()).unwrap();
    //tx.send("DONE\n".to_string()).unwrap();
    //tx.send("yxprintfeature\n".to_string()).unwrap();

    let mut last_store_time = SystemTime::now();
    loop {
        let turns = db.query_by_pieces(pieces);
        for turn in turns {
            let moves = db.query_by_previd(turn.rowid);
            if moves.iter().any(|x| x.evaluation <= -10000) {
                continue;
            }
            match pieces {
                1..=5 if moves.len() > 5 => continue,
                6..=7 if moves.len() > 3 => continue,
                8..=9 if moves.len() > 2 => continue,
                _ => {
                    if moves.len() > 0 {
                        continue;
                    }
                }
            };

            let mut m = BoardMatrix::from_base81_string(&turn.key);
            let mut command = String::with_capacity(1024);

            command.push_str("RESTART\nyxhashclear\nyxblockreset\n");

            if !moves.is_empty() {
                command.push_str("yxblock\n");
                moves.iter().for_each(|historical_move| {
                    command.push_str(&historical_move.answer);
                    command.push_str("\n");
                });
                command.push_str("DONE\n");
            }

            command.push_str("BOARD\n");

            /*
            m.for_each_piece(|row, col, value| {
                command.push_str(&format!("{},{},{}\n", row, col, value));
            }); */

            command.push_str("DONE\n");

            let (reply_tx, mut reply_rx) = oneshot::channel();

            tx.send((command, Some(reply_tx))).unwrap();

            println!("Evaluating {}", turn.key);
            let sleep = time::sleep(Duration::from_secs(30));
            tokio::pin!(sleep);

            loop {
                tokio::select! {
                    reply = (&mut reply_rx) => {

                        match reply
                        {
                            Ok((row , col , evaluation )) => {
                                if row >= 0 && col >= 0 {
                                    m[(row as usize, col as usize)] = match turn.pieces % 2 {
                                        0 => 1, // black
                                        1 => 2, // white
                                        _ => unreachable!()
                                    }; // set the correct value

                                    db.insert_only(&Turn{
                                        rowid : 0,
                                        previd : turn.rowid,
                                        key : m.to_base81_string(),
                                        answer : format!("{},{}", row, col),
                                        evaluation : evaluation,
                                        pieces : turn.pieces+1,
                                        over : false, //m.is_over(),
                                    });
                                }
                                else {
                                    db.set_over(turn.rowid);
                                }


                                //m.print();
                                println!("{},{} = {}; pieces={}", row, col, evaluation, turn.pieces + 1);
                            },
                            Err(e) => {
                                panic!("Unable to read response. {}", e);
                            }
                        };
                        break;
                    },
                    _ = &mut sleep => {
                        sleep.as_mut().reset(Instant::now() + Duration::from_secs(100));
                        tx.send(("yxstop\n".to_owned(), None) ).unwrap();
                        continue;
                    }
                }
            }

            if let Ok(elapsed) = last_store_time.elapsed() {
                if elapsed.as_secs() > 60 {
                    // store to file every X seconds
                    db.store();
                    last_store_time = SystemTime::now();
                }
            }
        }
        db.store();
        pieces += 1;
    }
}

struct ReplyHanlder {
    sender_deque: VecDeque<oneshot::Sender<(i8, i8, i32)>>,
    evaluation: i32,
    evaluation_regex: Regex,
    position_regex: Regex,
}

impl ReplyHanlder {
    fn new() -> Self {
        Self {
            sender_deque: VecDeque::new(),
            evaluation: 0,
            evaluation_regex: Regex::new(r"Evaluation\s*:\s*(?P<marks>\-?\d+)").unwrap(),
            position_regex: Regex::new(r"^(?P<row>\-?\d{1,2}),(?P<col>\-?\d{1,2})$").unwrap(),
        }
    }

    fn add_replier(self: &mut Self, tx: oneshot::Sender<(i8, i8, i32)>) {
        self.sender_deque.push_back(tx);
    }

    // https://github.com/accreator/Yixin-Board/blob/609c06015a0a239a3a5365c28ff8c7be96ecef9a/main.c#L4910
    fn parse_result(self: &mut Self, line: String) {
        if line.starts_with("MESSAGE REALTIME ") {
            //println!("{}", line);
        } else if line.starts_with("MESSAGE ") {
            if line.starts_with("MESSAGE DETAIL ") {
            } else {
                //println!("{}", line);
                // MESSAGE Speed: 2400 | Evaluation: -94
                for cap in self.evaluation_regex.captures_iter(&line) {
                    self.evaluation = cap["marks"].parse::<i32>().unwrap();
                }
            }
        } else {
            //println!("{}", line);
            for cap in self.position_regex.captures_iter(&line) {
                let row = cap["row"].parse::<i8>().unwrap();
                let col = cap["col"].parse::<i8>().unwrap();
                if let Some(tx) = self.sender_deque.pop_front() {
                    tx.send((row, col, self.evaluation))
                        .expect("tx.send() failed");
                    self.evaluation = 0;
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
 */
