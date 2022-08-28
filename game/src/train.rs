use crate::*;
use std::{cell::RefCell, fs, rc::Rc};

pub struct SelfPlayerOutcome(
    Vec<Tensor<f32>>, /*state*/
    Vec<Tensor<f32>>, /*mcts probs */
    Vec<f32>,         /*winner */
);

pub struct Trainer {
    // temperature parameter in (0, 1] controls the level of exploration
    temperature: f32,
    model: Rc<RefCell<PolicyValueModel>>,
    searcher: TreeSearcher<PolicyValueModel>,
}

impl Trainer {
    pub fn new() -> Self {
        let m = Rc::new(RefCell::new(get_best_model()));

        let searcher = TreeSearcher::new(5f32, 4000u32, m.clone());
        Self {
            temperature: 1e-3,
            model: m,
            searcher: searcher,
        }
    }

    pub fn self_play(self: &mut Self) -> SelfPlayerOutcome {
        let mut board = RenjuBoard::default();

        let capacity = board.width() * board.height();
        let mut board_state_list = Vec::<Tensor<f32>>::with_capacity(capacity);
        let mut mcts_probs_list = Vec::<Tensor<f32>>::with_capacity(capacity);
        let mut player_list = Vec::<bool>::with_capacity(capacity); // true if black turn

        loop {
            let move_prob_pairs: Vec<((usize, usize), f32)> =
                self.searcher.get_move_probability(&board, self.temperature);

            // determine the position to move
            // For self-player, choose a position by probabilities and also dirichlet noice
            let mut probability_tensor = Tensor::<f32>::new(&[move_prob_pairs.len() as u64]);
            move_prob_pairs
                .iter()
                .enumerate()
                .for_each(|(index, (_, probability))| {
                    probability_tensor[index] = *probability;
                });
            let index = self
                .model
                .borrow()
                .random_choose_with_dirichlet_noice(&probability_tensor)
                .expect("random_choose_with_dirichlet_noice failed");

            let pos = move_prob_pairs[index].0;
            self.searcher.update_with_position(pos);

            // save board state
            board_state_list.push(board.get_state_tensor());

            // convert to (1, 225) tensor
            let mut mcts_probs =
                Tensor::<f32>::new(&[1, board.width() as u64 * board.height() as u64]);
            move_prob_pairs.iter().for_each(|((row, col), prob)| {
                let index =
                    mcts_probs.get_index(&[0, *row as u64 * board.width() as u64 + *col as u64]);
                mcts_probs[index] = *prob;
            });
            mcts_probs_list.push(mcts_probs);

            player_list.push(board.is_black_turn());

            match board.do_move(pos) {
                TerminalState::AvailableMoves(_) => continue,
                s => {
                    board.print();
                    let winner_list: Vec<f32> = player_list
                        .iter()
                        .map(|is_black_turn| match s {
                            TerminalState::BlackWon if *is_black_turn => 1f32,
                            TerminalState::WhiteWon if !*is_black_turn => 1f32,
                            TerminalState::WhiteWon if *is_black_turn => -1f32,
                            TerminalState::BlackWon if !*is_black_turn => -1f32,
                            _ => 0f32,
                        })
                        .collect();

                    return SelfPlayerOutcome(board_state_list, mcts_probs_list, winner_list);
                }
            }
        }
    }
}

fn get_best_model() -> PolicyValueModel {
    let export_dir = "/Users/jerry/projects/renju/renju.git/game/renju_15x15_model/";

    let ai_model = PolicyValueModel::load(export_dir).expect("Unable to load model");

    let checkpoint_filename = "/Users/jerry/projects/renju/renju.git/game/saved.ckpt";
    if fs::metadata(checkpoint_filename).is_ok() {
        match ai_model.restore(checkpoint_filename) {
            Err(e) => {
                println!(
                    "WARNING : Unable to restore checkpoint {}. {}",
                    checkpoint_filename, e
                );
            }
            _ => {
                println!("Successfully loaded checkpoint {}", checkpoint_filename);
            }
        }
    }

    ai_model
}
