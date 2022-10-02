use crate::game::{RenjuBoard, SquareMatrix, SquaredMatrixExtension, StateTensor, TerminalState};
use crate::mcts::MonteCarloTree;
use crate::model::RenjuModel;
use std::cmp::Ordering;
use std::sync::{Arc, RwLock};
use std::{borrow::Cow, cell::RefCell, rc::Rc};

#[async_trait::async_trait]
pub trait Player {
    async fn do_next_move(
        self: &mut Self,
        board: &mut RenjuBoard,
        choices: &Vec<(usize, usize)>,
    ) -> (usize, usize);

    fn notify_opponent_moved(self: &mut Self, board: &RenjuBoard, pos: (usize, usize));
}

pub struct Match<'a, B, W>
where
    B: Player,
    W: Player,
{
    board: RenjuBoard,
    black_player: &'a mut B,
    white_player: &'a mut W,
}

// a match between black and white
impl<'a, B, W> Match<'a, B, W>
where
    B: Player,
    W: Player,
{
    pub fn new(black_player: &'a mut B, white_player: &'a mut W) -> Self {
        Self {
            board: RenjuBoard::default(),
            black_player: black_player,
            white_player: white_player,
        }
    }

    pub async fn play_to_end(self: &mut Self) -> TerminalState {
        let mut state = TerminalState::default();

        loop {
            let is_black_turn = self.board.is_black_turn();
            if let TerminalState::AvailableMoves(ref moves) = state {
                let pos = if is_black_turn {
                    self.black_player.do_next_move(&mut self.board, moves).await
                } else {
                    self.white_player.do_next_move(&mut self.board, moves).await
                };

                state = self.board.do_move(pos);

                if is_black_turn {
                    self.white_player.notify_opponent_moved(&self.board, pos);
                } else {
                    self.black_player.notify_opponent_moved(&self.board, pos);
                };

                match state {
                    TerminalState::BlackWon => {
                        return state;
                    }
                    TerminalState::WhiteWon => {
                        return state;
                    }
                    TerminalState::Draw => {
                        return state;
                    }
                    _ => continue,
                }
            } else {
                unreachable!("Should always be able to move in the loop before end");
            }
        }
    }
}

// play with self to produce training data
pub struct SelfPlayer<M>
where
    M: RenjuModel,
{
    tree: Arc<MonteCarloTree<M>>,
    // temperature parameter in (0, 1] controls the level of exploration
    temperature: f32,
    state_prob_pairs: Vec<(StateTensor, SquareMatrix)>,
    iterations: u32,
}

impl<M> SelfPlayer<M>
where
    M: RenjuModel,
{
    pub fn new_pair(model: M) -> (Self, Self) {
        let tree = Arc::new(MonteCarloTree::new(5f32, model));
        (
            Self {
                tree: tree.clone(),
                temperature: 1e-3,
                state_prob_pairs: Vec::with_capacity(100),
                iterations: 1000u32,
            },
            Self {
                tree: tree,
                temperature: 1e-3,
                state_prob_pairs: Vec::with_capacity(100),
                iterations: 1000u32,
            },
        )
    }

    // For self-player, choose a position by probabilities with dirichlet noice
    fn pick_move(
        self: &Self,
        move_prob_pairs: &Vec<((usize, usize), f32)>,
        board: &mut RenjuBoard,
    ) -> (usize, usize) {
        assert!(!move_prob_pairs.is_empty());

        let mut pairs = Cow::from(move_prob_pairs);
        while pairs.len() > 1 {
            // determine the position to move

            let vector: Vec<_> = pairs.iter().map(|(_, probability)| *probability).collect();

            let index = self
                .model
                .borrow()
                .random_choose_with_dirichlet_noice(&vector)
                .expect("random_choose_with_dirichlet_noice failed");

            let pos = pairs[index].0;
            if !board.is_forbidden(pos) {
                return pos;
            }
            pairs.to_mut().swap_remove(index); // remove the forbidden position and try again
        }
        move_prob_pairs[0].0
    }

    pub fn consume<F>(mut self: Self, mut cb: F)
    where
        F: FnMut(StateTensor, SquareMatrix),
    {
        while !self.state_prob_pairs.is_empty() {
            let (state_tensor, prob_matrix) = self.state_prob_pairs.swap_remove(0);

            get_equivalents(&state_tensor, &prob_matrix)
                .into_iter()
                .for_each(|(equi_state_tensor, equi_prob_matrix)| {
                    cb(equi_state_tensor, equi_prob_matrix);
                });
            cb(state_tensor, prob_matrix);
        }
    }
}

#[async_trait::async_trait]
impl<M> Player for SelfPlayer<M>
where
    M: RenjuModel,
{
    async fn do_next_move(
        self: &mut Self,
        board: &mut RenjuBoard,
        choices: &Vec<(usize, usize)>,
    ) -> (usize, usize) {
        for _ in 0..self.iterations {
            self.tree
                .rollout(board.clone(), choices)
                .await
                .expect("rollout failed");
        }

        let move_prob_pairs: Vec<((usize, usize), f32)> = self
            .tree
            .get_move_probability(self.temperature)
            .await
            .expect("get_move_probability() failed");

        // 15x15 tensor records the probability of each move
        let mut mcts_prob_matrix = SquareMatrix::default();
        move_prob_pairs
            .iter()
            .for_each(|((row, col), probability)| {
                mcts_prob_matrix[*row][*col] = *probability;
            });

        self.state_prob_pairs
            .push((board.get_state_tensor(), mcts_prob_matrix));

        let pos = self.pick_move(&move_prob_pairs, board);

        self.tree.update_with_position(pos);
        pos
    }

    fn notify_opponent_moved(self: &mut Self, _: &RenjuBoard, _: (usize, usize)) {
        // black and white are sharing the same tree, no need to update
        //self.tree.borrow_mut().update_with_position(pos);
    }
}

#[async_trait::async_trait]
pub struct AiPlayer<M>
where
    M: RenjuModel,
{
    tree: Rc<RefCell<MonteCarloTree<M>>>,
    // temperature parameter in (0, 1] controls the level of exploration
    temperature: f32,
    last_visit_times: SquareMatrix<u32>,
    iterations: u32,
}

impl<M> AiPlayer<M>
where
    M: RenjuModel,
{
    pub fn get_visit_times(self: &Self) -> SquareMatrix<u32> {
        self.last_visit_times
    }
    pub fn new(model: Rc<RefCell<M>>, iterations: u32) -> Self {
        let tree = Rc::new(RefCell::new(MonteCarloTree::new(5f32, model)));
        Self {
            tree: tree,
            temperature: 1e-3,
            last_visit_times: SquareMatrix::default(),
            iterations: iterations,
        }
    }

    pub fn rollout(self: &mut Self, board: RenjuBoard, choices: &Vec<(usize, usize)>) {
        self.tree.borrow_mut().rollout(board, choices)
    }
}

impl<M> Player for AiPlayer<M>
where
    M: RenjuModel,
{
    fn do_next_move(
        self: &mut Self,
        board: &mut RenjuBoard,
        choices: &Vec<(usize, usize)>,
    ) -> (usize, usize) {
        let pos = if choices.len() == 1 {
            choices[0]
        } else {
            for _ in 0..self.iterations {
                self.tree.borrow_mut().rollout(board.clone(), choices);
            }
            let move_prob_pairs: Vec<((usize, usize), f32)> = self
                .tree
                .borrow_mut()
                .get_move_probability(self.temperature);

            let pair = move_prob_pairs
                .into_iter()
                .max_by(|(_, left_score), (_, right_score)| {
                    left_score
                        .partial_cmp(right_score)
                        .unwrap_or(Ordering::Equal)
                })
                .expect("At least one pair");
            pair.0
        };

        self.last_visit_times = self.tree.borrow().get_visit_times();
        self.tree.borrow_mut().update_with_position(pos);
        pos
    }

    fn notify_opponent_moved(self: &mut Self, _: &RenjuBoard, pos: (usize, usize)) {
        self.tree.borrow_mut().update_with_position(pos);
    }
}

fn get_equivalents(
    state_tensor: &StateTensor,
    probability_tensor: &SquareMatrix,
) -> Vec<([[[f32; 15]; 15]; 4], SquareMatrix)> {
    let mut state_tensor1 = state_tensor.clone();
    let mut probability_tensor1 = probability_tensor.clone();

    state_tensor1[0].flip_over_main_diagonal();
    state_tensor1[1].flip_over_main_diagonal();
    state_tensor1[2].flip_over_main_diagonal();
    probability_tensor1.flip_over_main_diagonal();

    let mut state_tensor2 = state_tensor1.clone();
    let mut probability_tensor2 = probability_tensor1.clone();
    state_tensor2[0].flip_top_bottom();
    state_tensor2[1].flip_top_bottom();
    state_tensor2[2].flip_top_bottom();
    probability_tensor2.flip_top_bottom();

    let mut state_tensor3 = state_tensor.clone();
    let mut probability_tensor3 = probability_tensor.clone();
    state_tensor3[0].flip_over_anti_diagonal();
    state_tensor3[1].flip_over_anti_diagonal();
    state_tensor3[2].flip_over_anti_diagonal();
    probability_tensor3.flip_over_anti_diagonal();

    let mut state_tensor4 = state_tensor3.clone();
    let mut probability_tensor4 = probability_tensor3.clone();
    state_tensor4[0].flip_top_bottom();
    state_tensor4[1].flip_top_bottom();
    state_tensor4[2].flip_top_bottom();
    probability_tensor4.flip_top_bottom();

    let mut state_tensor5 = state_tensor.clone();
    let mut probability_tensor5 = probability_tensor.clone();
    state_tensor5[0].flip_left_right();
    state_tensor5[1].flip_left_right();
    state_tensor5[2].flip_left_right();
    probability_tensor5.flip_left_right();

    let mut state_tensor6 = state_tensor.clone();
    let mut probability_tensor6 = probability_tensor.clone();
    state_tensor6[0].flip_top_bottom();
    state_tensor6[1].flip_top_bottom();
    state_tensor6[2].flip_top_bottom();
    probability_tensor6.flip_top_bottom();

    let mut state_tensor7 = state_tensor6.clone();
    let mut probability_tensor7 = probability_tensor6.clone();
    state_tensor7[0].flip_left_right();
    state_tensor7[1].flip_left_right();
    state_tensor7[2].flip_left_right();
    probability_tensor7.flip_left_right();

    vec![
        (state_tensor1, probability_tensor1),
        (state_tensor2, probability_tensor2),
        (state_tensor3, probability_tensor3),
        (state_tensor4, probability_tensor4),
        (state_tensor5, probability_tensor5),
        (state_tensor6, probability_tensor6),
        (state_tensor7, probability_tensor7),
    ]
}

#[test]
fn test_get_equivalents() {
    let state_tensor: StateTensor<f32> = [
        [
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        ],
        [
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        ],
        [
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        ],
        [
            [0., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        ],
    ];

    let prob_matrix: SquareMatrix<f32> = [
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    ];

    let equivalents = get_equivalents(&state_tensor, &prob_matrix);

    let state_tensor_0: StateTensor<f32> = [
        [
            [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        ],
        [
            [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        ],
        [
            [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        ],
        [
            [0., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        ],
    ];

    let prob_matrix_0: SquareMatrix<f32> = [
        [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    ];
    assert_eq!(state_tensor_0, equivalents[0].0);
    assert_eq!(prob_matrix_0, equivalents[0].1);

    let matrix1: SquareMatrix<f32> = [
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    ];

    assert_eq!(matrix1, equivalents[1].0[0]);
    assert_eq!(matrix1, equivalents[1].0[1]);
    assert_eq!(matrix1, equivalents[1].0[2]);
    assert_eq!(matrix1, equivalents[1].1);

    let matrix2: SquareMatrix<f32> = [
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
    ];
    assert_eq!(matrix2, equivalents[2].0[0]);
    assert_eq!(matrix2, equivalents[2].0[1]);
    assert_eq!(matrix2, equivalents[2].0[2]);
    assert_eq!(matrix2, equivalents[2].1);

    let matrix3: SquareMatrix<f32> = [
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    ];
    assert_eq!(matrix3, equivalents[3].0[0]);
    assert_eq!(matrix3, equivalents[3].0[1]);
    assert_eq!(matrix3, equivalents[3].0[2]);
    assert_eq!(matrix3, equivalents[3].1);

    let matrix4: SquareMatrix<f32> = [
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    ];
    assert_eq!(matrix4, equivalents[4].0[0]);
    assert_eq!(matrix4, equivalents[4].0[1]);
    assert_eq!(matrix4, equivalents[4].0[2]);
    assert_eq!(matrix4, equivalents[4].1);

    let matrix5: SquareMatrix<f32> = [
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    ];
    assert_eq!(matrix5, equivalents[5].0[0]);
    assert_eq!(matrix5, equivalents[5].0[1]);
    assert_eq!(matrix5, equivalents[5].0[2]);
    assert_eq!(matrix5, equivalents[5].1);

    let matrix6: SquareMatrix<f32> = [
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    ];
    assert_eq!(matrix6, equivalents[6].0[0]);
    assert_eq!(matrix6, equivalents[6].0[1]);
    assert_eq!(matrix6, equivalents[6].0[2]);
    assert_eq!(matrix6, equivalents[6].1);
}
