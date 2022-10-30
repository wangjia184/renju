/*
 * (C) Copyright 2022 Jerry.Wang (https://github.com/wangjia184).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]
extern crate clap;
extern crate num_cpus;
#[macro_use]
extern crate lazy_static;

use clap::{Parser, Subcommand};

use tokio::time::{sleep, Duration};

mod contest;
mod game;
mod human;
mod mcts;
mod model;
#[cfg(feature="train")]
mod selfplay;
use human::MatchState;
#[cfg(feature="train")]
use selfplay::Trainer;

static ABOUT_TEXT: &str = "Renju Game";

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

    #[cfg(feature="train")]
    /// Self play and train
    #[clap(after_help=TRAIN_HELP_TEXT)]
    Train {},


}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let args = Arguments::parse();

    match args.verb {
        #[cfg(feature="train")]
        Some(Verb::SelfPlay {
            model_file,
            export_dir,
        }) => {
            let mut trainer = Trainer::new();

            trainer
                .produce_self_play_data(&model_file, &export_dir)
                .await;
        }

        #[cfg(feature="train")]
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

#[tauri::command]
async fn new_match(window: tauri::Window, black: bool) {
    let window_ref = &window;
    human::access(|mut m| async move {
        m.restart(black).await;
        window_ref.emit("board_updated", m.get_board()).unwrap();
        m
    })
    .await;
}

#[tauri::command]
async fn do_move(window: tauri::Window, pos: (usize, usize)) -> MatchState {
    let window_ref = &window;
    human::access(|mut m| async move {
        let state = m.human_move(pos).await;

        let bi = m.get_board();
        let seconds = match bi.get_stones() {
            0..=3 => 3,
            4..=6 => 7,
            7..=14 => 10,
            _ => 12,
        };
        window_ref.emit("board_updated", bi).unwrap();

        if state != MatchState::HumanWon
            && state != MatchState::Draw
            && state != MatchState::MachineWon
        {
            sleep(Duration::from_secs(seconds)).await;

            m.machine_move().await;

            window_ref.emit("board_updated", m.get_board()).unwrap();
        }

        m
    })
    .await
}
