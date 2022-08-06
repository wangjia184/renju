extern crate clap;

use clap::{Parser, Args, Subcommand};


use regex::{self, Regex};
use tokio::sync::mpsc::{ self, UnboundedReceiver, UnboundedSender };
use tokio::sync::oneshot;
use tokio::io::{AsyncWriteExt, BufReader, AsyncBufReadExt, BufWriter};
use tokio::process::{ Command };
use tokio::time::{self, Duration, Instant};

use std::time::{ SystemTime};
use std::process::Stdio;
use std::collections::VecDeque;

mod game;
mod storage;
use game::{BoardMatrix, Board};
use storage::{ Turn, Database };



#[tokio::main(flavor = "current_thread")] // use single-threaded runtime
async fn main() {

    let (tx, rx) = mpsc::unbounded_channel();

    // http://petr.lastovicka.sweb.cz/protocl2en.htm    

    //tx.send(("INFO max_node 120000\n".to_string(), None)).unwrap();
    //tx.send(("INFO caution_factor 2\n".to_string(), None)).unwrap();
    tx.send(("info show_detail 1\n".to_string(), None)).unwrap();
    tx.send(("INFO thread_split_depth 5\n".to_string(), None)).unwrap();
    tx.send(("INFO max_thread_num 16\n".to_string(), None) ).unwrap();
    tx.send(("INFO thread_num 16\n".to_string(), None) ).unwrap();
    tx.send(("INFO rule 2\n".to_string(), None)).unwrap();
    tx.send(("INFO max_node -1\n".to_string(), None)).unwrap();
    tx.send(("INFO max_depth 100\n".to_string(), None)).unwrap();
    tx.send(("INFO time_increment 0\n".to_string(), None)).unwrap();
    
    
    tx.send(("START 15 15\n".to_string(), None)).unwrap();
    tx.send(("RESTART\n".to_string(), None)).unwrap();
   
    
    tokio::spawn(async move {
        match_routine(tx).await;
    });

    run("C:\\projects\\renju\\match_generator\\Yixin\\engine.exe", Vec::new(), rx).await;
}


// https://github.com/accreator/Yixin-Board/blob/609c06015a0a239a3a5365c28ff8c7be96ecef9a/main.c#L5739
async fn run(app_path : &str, parameters : Vec<String>, mut input_rx : UnboundedReceiver<(String, Option<oneshot::Sender<(i8, i8, i32)>>)>){

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

    let stdout = child.stdout.take().expect("child did not have a handle to stdout");
    let stderr = child.stderr.take().expect("child did not have a handle to stderr");

    let mut stdin = child.stdin.take().expect("child did not have a handle to stdin");
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

async fn match_routine(tx : UnboundedSender<(String, Option<oneshot::Sender<(i8, i8, i32)>>)>){

    let mut pieces = 10;

    let db = Database::default();
    BoardMatrix::generate_opening_patterns().iter().for_each( |(key, _)| {
        db.insert_only(&Turn{
            rowid : 0,
            previd : 0,
            key : key.to_owned(),
            answer : String::default(),
            evaluation : 0,
            pieces : 3,
            over : false,
        });
    });

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
            if moves.iter().any(|x| x.evaluation <= -10000 ) {
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
                },
            };
    
            let mut m = BoardMatrix::from_base81_string(&turn.key);
            let mut command = String::with_capacity(1024);
    
            command.push_str("RESTART\nyxhashclear\nyxblockreset\n");
    
            if !moves.is_empty() {
                command.push_str("yxblock\n");
                moves.iter().for_each( |historical_move| {
                    command.push_str(&historical_move.answer);
                    command.push_str("\n");
                });
                command.push_str("DONE\n");
            }

            command.push_str("BOARD\n");
    
            m.for_each_piece( |row, col, value| {
                command.push_str(&format!("{},{},{}\n", row, col, value));
            });
    
            command.push_str("DONE\n");
    
            let (reply_tx, mut reply_rx) = oneshot::channel();
    
            tx.send((command, Some(reply_tx)) ).unwrap();
    
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
                                        over : m.is_over(),
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
                if elapsed.as_secs() > 60 { // store to file every X seconds
                    db.store();
                    last_store_time = SystemTime::now();
                }
            }
        }
        db.store();
        pieces += 1;
    }
    

}


struct ReplyHanlder
{
    sender_deque: VecDeque<oneshot::Sender<(i8, i8, i32)>>,
    evaluation : i32,
    evaluation_regex : Regex,
    position_regex : Regex,
}

impl ReplyHanlder {
    fn new() -> Self {
        Self { 
            sender_deque: VecDeque::new(), 
            evaluation: 0, 
            evaluation_regex : Regex::new(r"Evaluation\s*:\s*(?P<marks>\-?\d+)").unwrap(),
            position_regex : Regex::new(r"^(?P<row>\-?\d{1,2}),(?P<col>\-?\d{1,2})$").unwrap(),
        }
    }

    fn add_replier(self : &mut Self, tx : oneshot::Sender<(i8, i8, i32)>) {
        self.sender_deque.push_back(tx);
    }

    // https://github.com/accreator/Yixin-Board/blob/609c06015a0a239a3a5365c28ff8c7be96ecef9a/main.c#L4910
    fn parse_result(self : &mut Self, line : String){
        if line.starts_with("MESSAGE REALTIME ") {
            //println!("{}", line);
        } else if line.starts_with("MESSAGE ") {
            if line.starts_with("MESSAGE DETAIL ") {

            }
            else {
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
                    tx.send((row, col, self.evaluation)).expect("tx.send() failed");
                    self.evaluation = 0;
                }
                
            }
        }
        
    }
}

