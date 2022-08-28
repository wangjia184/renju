use nalgebra::SMatrix;
use num_traits::identities::Zero;
use std::cmp;
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use tensorflow::Tensor;

const SIZE: usize = 15;

const DISTANCE: usize = 1;
const AVAILABLE: u8 = 1;
pub type BoardMatrix = SMatrix<u8, SIZE, SIZE>;

#[derive(Debug, Clone)]
pub struct RenjuBoard {
    matrix: BoardMatrix,                      // 0=BLANK; 1=BLACK; 2=WHITE
    available_matrix: BoardMatrix,            // available positions, 1=available
    last_move: Option<(usize, usize)>,        // recent move
    pieces: usize,                            // number of moves
    black_win_moves: HashSet<(usize, usize)>, // just one black move in specified position to win
    white_win_moves: HashSet<(usize, usize)>, // just one white move in specified position to win
    forbidden_moves: HashSet<(usize, usize)>, // forbidden black moves
}

#[repr(u8)]
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Color {
    None = 0,
    Black = 1,
    White = 2,
}

impl Into<u8> for Color {
    fn into(self) -> u8 {
        self as u8
    }
}

impl TryFrom<u8> for Color {
    type Error = ();

    fn try_from(v: u8) -> Result<Self, Self::Error> {
        match v {
            x if x == Color::None as u8 => Ok(Color::None),
            x if x == Color::Black as u8 => Ok(Color::Black),
            x if x == Color::White as u8 => Ok(Color::White),
            _ => Err(()),
        }
    }
}

impl Default for RenjuBoard {
    fn default() -> Self {
        let mut available = BoardMatrix::zero();
        available[(SIZE / 2, SIZE / 2)] = Color::Black.into();
        Self {
            matrix: BoardMatrix::zeros(),
            available_matrix: available,
            last_move: None,
            pieces: 0,
            black_win_moves: HashSet::new(),
            white_win_moves: HashSet::new(),
            forbidden_moves: HashSet::new(),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum TerminalState {
    AvailableMoves(Vec<(usize, usize)>),
    BlackWon,
    WhiteWon,
    Draw, // tie
}

impl Default for TerminalState {
    fn default() -> Self {
        TerminalState::AvailableMoves(vec![(SIZE / 2, SIZE / 2)])
    }
}

impl TerminalState {
    pub fn is_over(self: &Self) -> bool {
        match self {
            TerminalState::AvailableMoves(_) => false,
            _ => true,
        }
    }
}

impl RenjuBoard {
    pub fn print(self: &Self) {
        self.matrix.print();
    }
    pub fn width(self: &Self) -> usize {
        SIZE
    }
    pub fn height(self: &Self) -> usize {
        SIZE
    }
    // clear board
    pub fn clear(self: &mut Self) {
        self.last_move = None;
        self.matrix.fill(Color::None.into());
        self.available_matrix.fill(Color::None.into());
        self.available_matrix[(SIZE / 2, SIZE / 2)] = Color::Black.into();
        self.pieces = 0;
    }

    // next move is black?
    pub fn is_black_turn(self: &Self) -> bool {
        self.pieces % 2 == 0
    }

    // Add a new move into board
    pub fn do_move(self: &mut Self, pos: (usize, usize)) -> TerminalState {
        assert_eq!(self.matrix[pos], Color::None.into());
        assert!(self.available_matrix[pos] == AVAILABLE || self.last_move.is_none());

        let color = if (self.pieces % 2) == 0 {
            Color::Black
        } else {
            Color::White
        };

        let is_black_turn = self.is_black_turn();

        // update board pieces
        let mut state = self.matrix.scan_continuous_pieces(pos, color);

        // update status
        self.matrix[pos] = if is_black_turn {
            Color::Black.into()
        } else {
            Color::White.into()
        };
        self.last_move = Some(pos);
        self.pieces += 1;

        if state.has_won() {
            return if is_black_turn {
                TerminalState::BlackWon
            } else {
                TerminalState::WhiteWon
            };
        }

        if is_black_turn && state.is_forbidden() {
            return TerminalState::WhiteWon;
        }

        // remove win positions if exists
        self.black_win_moves.remove(&pos);
        self.white_win_moves.remove(&pos);

        // merge new win positions
        while !state.black_win_moves.is_empty() {
            self.black_win_moves.insert(state.black_win_moves.remove(0));
        }
        while !state.white_win_moves.is_empty() {
            self.black_win_moves.insert(state.white_win_moves.remove(0));
        }
        while !state.forbidden_moves.is_empty() {
            self.forbidden_moves.insert(state.forbidden_moves.remove(0));
        }

        // update available positions
        // only allow srounding positions
        let start_row = if pos.0 > DISTANCE {
            pos.0 - DISTANCE
        } else {
            0
        };
        let start_col = if pos.1 > DISTANCE {
            pos.1 - DISTANCE
        } else {
            0
        };
        let end_row = cmp::min(SIZE - 1, pos.0 + DISTANCE);
        let end_col = cmp::min(SIZE - 1, pos.1 + DISTANCE);
        for row in start_row..=end_row {
            for col in start_col..=end_col {
                self.available_matrix[(row, col)] = if self.matrix[(row, col)] == Color::None.into()
                {
                    AVAILABLE
                } else {
                    Color::None.into()
                };
            }
        }

        if self.pieces >= SIZE * SIZE {
            TerminalState::Draw
        } else {
            // remove forbidden move. perhaps it was a win move but now it is forbidden
            self.black_win_moves
                .retain(|pos| !self.matrix.is_forbidden(*pos));
            if self.is_black_turn() {
                // next move is black, and there is at least one move to win
                if !self.black_win_moves.is_empty() {
                    return TerminalState::AvailableMoves(
                        self.black_win_moves.clone().into_iter().collect(),
                    );
                }
                // white is going to win, try to prevent lose
                if !self.white_win_moves.is_empty() {
                    return TerminalState::AvailableMoves(
                        self.white_win_moves.clone().into_iter().collect(),
                    );
                }

                // double check forbidden moves
                self.forbidden_moves
                    .retain(|pos| self.matrix.is_forbidden(*pos));

                // exclude forbidden moves if there is any other available one
                return TerminalState::AvailableMoves(self.get_available_moves(true));
            } else {
                // next move is white, and there is at least one move to win
                if !self.white_win_moves.is_empty() {
                    return TerminalState::AvailableMoves(
                        self.white_win_moves.clone().into_iter().collect(),
                    );
                }
                // black is going to win, try to prevent lose
                if !self.black_win_moves.is_empty() {
                    return TerminalState::AvailableMoves(
                        self.black_win_moves.clone().into_iter().collect(),
                    );
                }
                return TerminalState::AvailableMoves(self.get_available_moves(false));
            }
        }
    }

    // return available moves
    fn get_available_moves(
        self: &Self,
        exclude_forbidden: bool,
    ) -> Vec<(usize /*row*/, usize /*col*/)> {
        // first black
        if self.last_move.is_none() {
            return vec![(SIZE / 2, SIZE / 2)];
        }

        let mut positions = Vec::new();
        let mut forbidden_moves = Vec::new();
        for row in 0..SIZE {
            for col in 0..SIZE {
                if self.available_matrix[(row, col)] == AVAILABLE {
                    if exclude_forbidden && self.forbidden_moves.contains(&(row, col)) {
                        forbidden_moves.push((row, col));
                    } else {
                        positions.push((row, col));
                    }
                }
            }
        }

        if positions.is_empty() {
            forbidden_moves // if no other moves except forbidden one, then return
        } else {
            positions
        }
    }

    pub fn get_state_tensor(self: &Self) -> Tensor<f32> {
        let mut state_tensor = Tensor::<f32>::new(&[1, 4, SIZE as u64, SIZE as u64]);

        let (black_base_index, white_base_index) = if self.is_black_turn() {
            // next move is black
            (0, state_tensor.get_index(&[0, 1, 0, 0]))
        } else {
            // next move is white
            (state_tensor.get_index(&[0, 1, 0, 0]), 0)
        };

        let mut offset: usize = 0;
        for row in 0..SIZE {
            for col in 0..SIZE {
                match Color::try_from(self.matrix[(row, col)]).expect("Invalid board state") {
                    Color::Black => {
                        state_tensor[black_base_index + offset] = 1f32;
                    }
                    Color::White => {
                        state_tensor[white_base_index + offset] = 1f32;
                    }
                    _ => {}
                };
                offset += 1;
            }
        }

        if let Some((row, col)) = self.last_move {
            let index = state_tensor.get_index(&[0, 2, row as u64, col as u64]);
            state_tensor[index] = 1f32;
        }

        if self.is_black_turn() {
            // next move is black {
            let base_index = state_tensor.get_index(&[0, 3, 0, 0]);
            for index in base_index..state_tensor.len() {
                state_tensor[index] = 1f32;
            }
        }
        state_tensor
    }
}

pub trait BoardMatrixExtension {
    fn print(self: &Self);

    fn from_base81_string(text: &str) -> Self;
    fn to_base81_string(self: &Self) -> String;
    //fn for_each_piece<F: FnMut(usize, usize, u8)>(self: &Self, cb: F);
    //fn is_over(self: &Self) -> bool;
    fn scan_continuous_pieces(self: &Self, position: (usize, usize), color: Color) -> NewMoveState;
    fn is_forbidden(self: &Self, position: (usize, usize)) -> bool;
    fn get_all_appearances(
        self: &Self,
        last_move: (usize, usize),
        answer: (usize, usize),
    ) -> Vec<(BoardMatrix, (usize, usize), (usize, usize))>;
    fn get_blacks_whites(self: &Self) -> (BoardMatrix /*black*/, BoardMatrix /*white */);

    fn scan_direction<F>(
        self: &Self,
        pos: (usize, usize),
        color: Color,
        get_neighbor: F,
    ) -> [Vec<(usize, usize)>; 5]
    where
        F: Fn((usize, usize)) -> Option<(usize, usize)>;

    fn scan_row<F1, F2>(
        self: &Self,
        pos: (usize, usize),
        color: Color,
        get_neighbor_of_main_direction: F1,
        get_neighbor_of_opposite_direction: F2,
        state: &mut NewMoveState,
    ) where
        F1: Fn((usize, usize)) -> Option<(usize, usize)>,
        F2: Fn((usize, usize)) -> Option<(usize, usize)>;
}

#[derive(Debug)]
pub struct NewMoveState {
    color: Color,
    three_count: u8, // live three
    four_count: u8,  // open four
    has_five: bool,
    over_five: bool,
    black_win_moves: Vec<(usize, usize)>, // white has to choose one of them in next turn if non-empty
    white_win_moves: Vec<(usize, usize)>, // black has to choose one of them in next turn if non-empty
    forbidden_moves: Vec<(usize, usize)>, // forbidden moves
}

impl NewMoveState {
    fn new(color: Color) -> Self {
        Self {
            color: color,
            three_count: 0,
            four_count: 0,
            has_five: false,
            over_five: false,
            black_win_moves: Vec::new(),
            white_win_moves: Vec::new(),
            forbidden_moves: Vec::new(),
        }
    }

    pub fn get_black_win_moves(self: &Self) -> &Vec<(usize, usize)> {
        &self.black_win_moves
    }

    pub fn get_white_win_moves(self: &Self) -> &Vec<(usize, usize)> {
        &self.white_win_moves
    }

    #[allow(dead_code)]
    pub fn get_live_three_count(self: &Self) -> u8 {
        self.three_count
    }

    #[allow(dead_code)]
    pub fn get_open_four_count(self: &Self) -> u8 {
        self.four_count
    }

    #[allow(dead_code)]
    pub fn has_five(self: &Self) -> bool {
        self.has_five
    }

    #[allow(dead_code)]
    pub fn over_five(self: &Self) -> bool {
        self.over_five
    }

    fn is_forbidden(self: &Self) -> bool {
        self.color == Color::Black
            && !self.has_five
            && (self.over_five || self.three_count > 1 || self.four_count > 1)
    }

    fn has_won(self: &Self) -> bool {
        self.has_five || (self.color == Color::White && self.over_five)
    }
}

impl BoardMatrixExtension for BoardMatrix {
    fn scan_continuous_pieces(self: &Self, position: (usize, usize), color: Color) -> NewMoveState {
        let mut state = NewMoveState::new(color);

        // horizontal
        self.scan_row(
            position,
            color,
            |(row, col)| {
                if col > 0 {
                    Some((row, col - 1))
                } else {
                    None
                }
            },
            |(row, col)| {
                if col < SIZE - 1 {
                    Some((row, col + 1))
                } else {
                    None
                }
            },
            &mut state,
        );

        // vertical
        self.scan_row(
            position,
            color,
            |(row, col)| {
                if row > 0 {
                    Some((row - 1, col))
                } else {
                    None
                }
            },
            |(row, col)| {
                if row < SIZE - 1 {
                    Some((row + 1, col))
                } else {
                    None
                }
            },
            &mut state,
        );

        // main diagonal
        self.scan_row(
            position,
            color,
            |(row, col)| {
                if row > 0 && col > 0 {
                    Some((row - 1, col - 1))
                } else {
                    None
                }
            },
            |(row, col)| {
                if row < SIZE - 1 && col < SIZE - 1 {
                    Some((row + 1, col + 1))
                } else {
                    None
                }
            },
            &mut state,
        );

        // anti diagonal
        self.scan_row(
            position,
            color,
            |(row, col)| {
                if row > 0 && col < SIZE - 1 {
                    Some((row - 1, col + 1))
                } else {
                    None
                }
            },
            |(row, col)| {
                if row < SIZE - 1 && col > 0 {
                    Some((row + 1, col - 1))
                } else {
                    None
                }
            },
            &mut state,
        );

        state
    }

    // check if the black at (usize, usize) is forbidden
    // Double three – Black cannot place a stone that builds two separate lines with three black stones in unbroken rows (i.e. rows not blocked by white stones).
    // Double four – Black cannot place a stone that builds two separate lines with four black stones in a row.
    // Overline – six or more black stones in a row.
    fn is_forbidden(self: &Self, position: (usize, usize)) -> bool {
        let state = self.scan_continuous_pieces(position, Color::Black);
        !state.has_five && (state.over_five || state.three_count > 1 || state.four_count > 1)
    }

    /// Scan same color pieces and blanks in a direction of position
    /// Return an array of vectors, each of them contains the postions of pieces or blanks
    ///          0          1          2          3          4
    ///  X --> BLACKS --> BLANKS --> BLACKS --> BLANKS --> BLACKS
    ///  X --> WHITES --> BLANKS --> WHITES --> BLANKS --> WHITES
    fn scan_direction<F>(
        self: &Self,
        pos: (usize, usize),
        color: Color,
        get_neighbor: F,
    ) -> [Vec<(usize, usize)>; 5]
    where
        F: Fn((usize, usize)) -> Option<(usize, usize)>,
    {
        let mut vectors: [Vec<(usize, usize)>; 5] = Default::default();

        let mut current = pos;
        let mut expected_state = color.into();
        let mut index = 0;
        while index < vectors.len() {
            match get_neighbor(current) {
                None => break, // out of board
                Some(p) => {
                    let state = Color::try_from(self[p]).expect("Invalid piece color");
                    if state == expected_state {
                        current = p;
                        vectors[index].push(current);
                    } else {
                        index += 1;
                        match state {
                            Color::White if color == Color::White => expected_state = Color::White,
                            Color::Black if color == Color::Black => expected_state = Color::Black,
                            Color::None => expected_state = Color::None,
                            _ => break, // opposite player's piece
                        }
                    }
                }
            }
        } // end of while
        vectors
    }

    fn scan_row<F1, F2>(
        self: &Self,
        pos: (usize, usize),
        color: Color,
        get_neighbor_of_main_direction: F1,
        get_neighbor_of_opposite_direction: F2,
        state: &mut NewMoveState,
    ) where
        F1: Fn((usize, usize)) -> Option<(usize, usize)>,
        F2: Fn((usize, usize)) -> Option<(usize, usize)>,
    {
        let main_diretion: [Vec<(usize, usize)>; 5] =
            self.scan_direction(pos, color, get_neighbor_of_main_direction);
        let opposite_diretion: [Vec<(usize, usize)>; 5] =
            self.scan_direction(pos, color, get_neighbor_of_opposite_direction);

        let continuous_blacks = main_diretion[0].len() + opposite_diretion[0].len() + 1;

        match continuous_blacks {
            // ┼┼┼┼┼┼┼
            // ┼●●●●●┼
            // ┼┼┼┼┼┼┼
            5 => state.has_five = true, // five in a row, no forbidden

            //////////////////// four continuous blacks
            4 => {
                let mut four = false;
                if main_diretion[1].len() > 0 {
                    // ┼●●●●┼
                    // ^
                    if color == Color::Black {
                        let mut cloned = self.clone();
                        cloned[pos] = Color::Black.into(); // suppose the specified position is black
                        if !cloned.is_forbidden(main_diretion[1][0]) {
                            four = true;
                            state.black_win_moves.push(main_diretion[1][0]);
                        } else {
                            state.forbidden_moves.push(main_diretion[1][0]);
                        }
                    } else {
                        four = true;
                        state.white_win_moves.push(main_diretion[1][0]);
                    }
                }

                // ┼●●●●┼
                //      ^
                if opposite_diretion[1].len() > 0 {
                    if color == Color::Black {
                        let mut cloned = self.clone();
                        cloned[pos] = Color::Black.into(); // suppose the specified position is black
                        if !cloned.is_forbidden(opposite_diretion[1][0]) {
                            four = true;
                            state.black_win_moves.push(opposite_diretion[1][0]);
                        } else {
                            state.forbidden_moves.push(opposite_diretion[1][0]);
                        }
                    } else {
                        four = true;
                        state.white_win_moves.push(opposite_diretion[1][0]);
                    }
                }

                if four {
                    state.four_count += 1;
                }
            }

            //////////////////// three continuous blacks
            3 => {
                // ┼●┼●●●┼
                //   ^
                if main_diretion[1].len() == 1 && main_diretion[2].len() == 1 {
                    if color == Color::Black {
                        let mut cloned = self.clone();
                        cloned[pos] = Color::Black.into(); // suppose the specified position is black
                        if !cloned.is_forbidden(main_diretion[1][0]) {
                            state.four_count += 1;
                            state.black_win_moves.push(main_diretion[1][0]);
                        } else {
                            state.forbidden_moves.push(main_diretion[1][0]);
                        }
                    } else {
                        state.four_count += 1;
                        state.white_win_moves.push(main_diretion[1][0]);
                    }
                }

                // ┼●●●┼●┼
                //     ^
                if opposite_diretion[1].len() == 1 && opposite_diretion[2].len() == 1 {
                    if color == Color::Black {
                        let mut cloned = self.clone();
                        cloned[pos] = Color::Black.into(); // suppose the specified position is black
                        if !cloned.is_forbidden(opposite_diretion[1][0]) {
                            state.four_count += 1;
                            state.black_win_moves.push(opposite_diretion[1][0]);
                        } else {
                            state.forbidden_moves.push(opposite_diretion[1][0]);
                        }
                    } else {
                        state.four_count += 1;
                        state.white_win_moves.push(opposite_diretion[1][0]);
                    }
                }

                let mut live_three = false;
                // ┼┼●●●┼
                if (main_diretion[1].len() > 2
                    || (main_diretion[1].len() == 2 && main_diretion[2].is_empty()))
                    && (opposite_diretion[1].len() > 1
                        || (opposite_diretion[1].len() == 1 && opposite_diretion[2].is_empty()))
                {
                    if color == Color::Black {
                        let mut cloned = self.clone();
                        cloned[pos] = Color::Black.into(); // suppose the specified position is black

                        // ┼┼●●●┼
                        // ^^   ^
                        let mut is_forbidden = false;
                        if cloned.is_forbidden(main_diretion[1][0]) {
                            state.forbidden_moves.push(main_diretion[1][0]);
                            is_forbidden = true;
                        } else if cloned.is_forbidden(main_diretion[1][1]) {
                            state.forbidden_moves.push(main_diretion[1][1]);
                            is_forbidden = true;
                        } else if cloned.is_forbidden(opposite_diretion[1][0]) {
                            state.forbidden_moves.push(opposite_diretion[1][0]);
                            is_forbidden = true;
                        }
                        if !is_forbidden {
                            live_three = true;
                        }
                    } else {
                        live_three = true;
                    }
                }

                // ┼●●●┼┼
                if (opposite_diretion[1].len() > 2
                    || (opposite_diretion[1].len() == 2 && opposite_diretion[2].is_empty()))
                    && (main_diretion[1].len() > 1
                        || (main_diretion[1].len() == 1 && main_diretion[2].is_empty()))
                {
                    if color == Color::Black {
                        let mut cloned = self.clone();
                        cloned[pos] = Color::Black.into(); // suppose the specified position is black

                        // ┼┼●●●┼
                        //  ^   ^
                        let mut is_forbidden = false;
                        if cloned.is_forbidden(opposite_diretion[1][0]) {
                            state.forbidden_moves.push(opposite_diretion[1][0]);
                            is_forbidden = true;
                        } else if cloned.is_forbidden(opposite_diretion[1][1]) {
                            state.forbidden_moves.push(opposite_diretion[1][1]);
                            is_forbidden = true;
                        } else if cloned.is_forbidden(main_diretion[1][0]) {
                            state.forbidden_moves.push(main_diretion[1][0]);
                            is_forbidden = true;
                        }
                        if !is_forbidden {
                            live_three = true;
                        }
                    } else {
                        live_three = true;
                    }
                }

                if live_three {
                    state.three_count += 1;
                }
            }

            //////////////////// two continuous blacks
            2 => {
                // ●●┼●●
                if main_diretion[1].len() == 1 && main_diretion[2].len() == 2 {
                    if color == Color::Black {
                        let mut cloned = self.clone();
                        cloned[pos] = Color::Black.into(); // suppose the specified position is black
                        if !cloned.is_forbidden(main_diretion[1][0]) {
                            state.four_count += 1;
                            state.black_win_moves.push(main_diretion[1][0]);
                        } else {
                            state.forbidden_moves.push(main_diretion[1][0]);
                        }
                    } else {
                        state.four_count += 1;
                        state.white_win_moves.push(main_diretion[1][0]);
                    }
                }

                // ●●┼●●
                if opposite_diretion[1].len() == 1 && opposite_diretion[2].len() == 2 {
                    if color == Color::Black {
                        let mut cloned = self.clone();
                        cloned[pos] = Color::Black.into(); // suppose the specified position is black
                        if !cloned.is_forbidden(opposite_diretion[1][0]) {
                            state.four_count += 1;
                            state.black_win_moves.push(opposite_diretion[1][0]);
                        } else {
                            state.forbidden_moves.push(opposite_diretion[1][0]);
                        }
                    } else {
                        state.four_count += 1;
                        state.white_win_moves.push(opposite_diretion[1][0]);
                    }
                }

                // ┼●┼●●┼
                if main_diretion[1].len() == 1
                    && main_diretion[2].len() == 1
                    && (
                        main_diretion[3].len() > 1 // at least two blanks
              ||
              ( main_diretion[3].len() == 1 && main_diretion[4].is_empty() )
                        // or one blank without further blacks
                    )
                    && (
                        opposite_diretion[1].len() > 1 // at least two blanks
              ||
              ( opposite_diretion[1].len() == 1 && opposite_diretion[2].is_empty() )
                        // or one blank without further blacks
                    )
                {
                    if color == Color::Black {
                        let mut cloned = self.clone();
                        cloned[pos] = Color::Black.into(); // suppose the specified position is black

                        // ┼●┼●●┼
                        //   ^
                        if !cloned.is_forbidden(main_diretion[1][0]) {
                            state.three_count += 1;
                        } else {
                            state.forbidden_moves.push(main_diretion[1][0]);
                        }
                    } else {
                        state.three_count += 1;
                    }
                }

                // ┼●●┼●┼
                if opposite_diretion[1].len() == 1
                    && opposite_diretion[2].len() == 1
                    && (
                        opposite_diretion[3].len() > 1 // at least two blanks
              ||
              ( opposite_diretion[3].len() == 1 && opposite_diretion[4].is_empty() )
                        // or one blank without further blacks
                    )
                    && (
                        main_diretion[1].len() > 1 // at least two blanks
              ||
              ( main_diretion[1].len() == 1 && main_diretion[2].is_empty() )
                        // or one blank without further blacks
                    )
                {
                    if color == Color::Black {
                        let mut cloned = self.clone();
                        cloned[pos] = Color::Black.into(); // suppose the specified position is black

                        // ┼●●┼●┼
                        //    ^
                        if !cloned.is_forbidden(opposite_diretion[1][0]) {
                            state.three_count += 1;
                        } else {
                            state.forbidden_moves.push(opposite_diretion[1][0]);
                        }
                    } else {
                        state.three_count += 1;
                    }
                }
            }

            //////////////////// single black
            1 => {
                // ●●●┼●
                if main_diretion[1].len() == 1 && main_diretion[2].len() == 3 {
                    if color == Color::Black {
                        let mut cloned = self.clone();
                        cloned[pos] = Color::Black.into(); // suppose the specified position is black

                        // ●●●┼●
                        //    ^
                        if !cloned.is_forbidden(main_diretion[1][0]) {
                            state.four_count += 1;
                            state.black_win_moves.push(main_diretion[1][0]);
                        } else {
                            state.forbidden_moves.push(main_diretion[1][0]);
                        }
                    } else {
                        state.four_count += 1;
                        state.white_win_moves.push(main_diretion[1][0]);
                    }
                }

                // ●┼●●●
                if opposite_diretion[1].len() == 1 && opposite_diretion[2].len() == 3 {
                    if color == Color::Black {
                        let mut cloned = self.clone();
                        cloned[pos] = Color::Black.into(); // suppose the specified position is black

                        // ●●●┼●
                        //    ^
                        if !cloned.is_forbidden(opposite_diretion[1][0]) {
                            state.four_count += 1;
                            state.black_win_moves.push(opposite_diretion[1][0]);
                        } else {
                            state.forbidden_moves.push(opposite_diretion[1][0]);
                        }
                    } else {
                        state.four_count += 1;
                        state.white_win_moves.push(opposite_diretion[1][0]);
                    }
                }

                // ┼●●┼●┼
                if main_diretion[1].len() == 1
                    && main_diretion[2].len() == 2
                    && (main_diretion[3].len() > 1
                        || (main_diretion[3].len() == 1 && main_diretion[4].is_empty()))
                    && (opposite_diretion[1].len() > 1
                        || (opposite_diretion[1].len() == 1 && opposite_diretion[2].is_empty()))
                {
                    if color == Color::Black {
                        let mut cloned = self.clone();
                        cloned[pos] = Color::Black.into(); // suppose the specified position is black

                        // ┼●●┼●┼
                        //    ^
                        if !cloned.is_forbidden(main_diretion[1][0]) {
                            state.three_count += 1;
                        } else {
                            state.forbidden_moves.push(main_diretion[1][0]);
                        }
                    } else {
                        state.three_count += 1;
                    }
                }

                // ┼●┼●●┼
                if opposite_diretion[1].len() == 1
                    && opposite_diretion[2].len() == 2
                    && (opposite_diretion[3].len() > 1
                        || (opposite_diretion[3].len() == 1 && opposite_diretion[4].is_empty()))
                    && (main_diretion[1].len() > 1
                        || (main_diretion[1].len() == 1 && main_diretion[2].is_empty()))
                {
                    if color == Color::Black {
                        let mut cloned = self.clone();
                        cloned[pos] = Color::Black.into(); // suppose the specified position is black

                        // ┼●●┼●┼
                        //    ^
                        if !cloned.is_forbidden(opposite_diretion[1][0]) {
                            state.three_count += 1;
                        } else {
                            state.forbidden_moves.push(opposite_diretion[1][0]);
                        }
                    } else {
                        state.three_count += 1;
                    }
                }
            }
            0 => unreachable!("Number of black stone will never be zero!"),

            //////////////////// move than five
            _ => {
                state.over_five = true;
            }
        }
    }
    /*
    fn is_over(self: &Self) -> bool {
        let mut states = [[[[0u8; 4]; 2]; SIZE]; SIZE];

        for row in 0..SIZE {
            for col in 0..SIZE {
                let index = match self[(row, col)] {
                    BLACK => 0,
                    WHITE => 1,
                    _ => continue,
                };

                let mut horizontal_len = 1;
                let mut vertical_len = 1;
                let mut main_diagonal_len = 1;
                let mut anti_diagonal_len = 1;

                if col > 0 {
                    // horizontal
                    horizontal_len += states[row][col - 1][index][0];
                    if horizontal_len > 4 {
                        return true;
                    }
                }
                if row > 0 {
                    // vertical
                    vertical_len += states[row - 1][col][index][1];
                    if vertical_len > 4 {
                        return true;
                    }

                    if col > 0 {
                        // main diagonal
                        main_diagonal_len += states[row - 1][col - 1][index][2];
                        if main_diagonal_len > 4 {
                            return true;
                        }
                    }
                    if col < SIZE - 1 {
                        // anti diagonal
                        anti_diagonal_len += states[row - 1][col + 1][index][3];
                        if anti_diagonal_len > 4 {
                            return true;
                        }
                    }
                }

                states[row][col][index][0] = horizontal_len;
                states[row][col][index][1] = vertical_len;
                states[row][col][index][2] = main_diagonal_len;
                states[row][col][index][3] = anti_diagonal_len;
            }
        }

        false
    } */

    /*
       fn for_each_piece<F: FnMut(usize, usize, u8)>(self: &Self, mut cb: F) {
           let mut vector = Vec::<(usize, usize)>::new();
           let mut expected_value = 1u8;
           for row in 0..SIZE {
               for col in 0..SIZE {
                   let value = self[(row, col)];
                   if value != 0 {
                       if value == expected_value {
                           // expected
                           cb(row, col, value);
                           expected_value = match expected_value {
                               BLACK => WHITE,
                               WHITE => BLACK,
                               _ => panic!("incorrect value"),
                           };
                           // accumulated
                           if let Some(more) = vector.pop() {
                               cb(more.0, more.1, expected_value);
                               expected_value = match expected_value {
                                   BLACK => WHITE,
                                   WHITE => BLACK,
                                   _ => panic!("incorrect value"),
                               };
                           }
                       } else {
                           vector.push((row, col));
                       }
                   }
               }
           }
           if !vector.is_empty() {
               self.print();
               panic!("piece is unbalanced!");
           }
       }
    */
    // https://www.wuziqi123.com/jiangzuo/dingshiyanjiu/156.html
    /*
    fn generate_opening_patterns() -> HashMap<String, Self> where Self : Sized
    {
      let mut map = HashMap::new();

      let mut m = BoardMatrix::zero();
      m[(7,7)] = 1;
      m[(7,8)] = 2;
      for row in 5..10 {
        for col in 7..10 {
          let mut pattern = m.clone();
          if pattern[(row, col)] == 0 {
            pattern[(row, col)] = 1;
            let id = pattern.mediocritize();
            map.insert( id, pattern);
          }
        }
      }

      let mut m = BoardMatrix::zero();
      m[(7,7)] = 1;
      m[(8,8)] = 2;
      for row in 5..10 {
        for col in 5..10 {
          let mut pattern = m.clone();
          if pattern[(row, col)] == 0 {
            pattern[(row, col)] = 1;
            let id = pattern.mediocritize();
            map.insert( id, pattern);
          }
        }
      }
      map
    }
     */

    fn from_base81_string(text: &str) -> Self {
        let mut m = BoardMatrix::zero();

        let mut count = 0;
        text.chars().rev().enumerate().for_each(|(_, ch)| {
            for offset in 0..4 {
                if count < SIZE * SIZE {
                    let row = SIZE - 1 - count / SIZE;
                    let col = SIZE - 1 - (count % SIZE);
                    count = count + 1;

                    let table_index = (ch as usize) * 4 + (4 - 1 - offset);
                    let value = BASE81_REVERSE_TABLE[table_index];
                    if value > 2 {
                        panic!("Invalid value in matrix. {}", value);
                    }
                    m[(row, col)] = value;
                }
            }
        });
        m
    }

    fn print(self: &BoardMatrix) {
        let mut text = String::with_capacity((SIZE + 1) * (SIZE + 2));
        for i in 0..SIZE {
            if i == 0 {
                text.push_str("┌");
            } else {
                text.push_str("┬");
            }
            text.push_str("───");
        }
        text.push_str("┐\n");
        for row in 0..SIZE {
            for col in 0..SIZE {
                text.push_str("│");

                match self[(row, col)] {
                    1 => text.push_str(" O "), // black
                    2 => text.push_str(" X "), // white
                    0 => text.push_str("   "),
                    _ => panic!("Unexpected value {} in matrix", self[(row, col)]),
                }
            }
            text.push_str("│\n");

            for i in 0..SIZE {
                if i == 0 {
                    if row < SIZE - 1 {
                        text.push_str("├");
                    } else {
                        text.push_str("└");
                    }
                } else {
                    if row < SIZE - 1 {
                        text.push_str("┼");
                    } else {
                        text.push_str("┴");
                    }
                }
                text.push_str("───");
            }
            if row < SIZE - 1 {
                text.push_str("┤\n");
            } else {
                text.push_str("┘\n");
            }
        }

        println!("{}", text);
    }

    // The borad only contains three types of values: 0(blank) 1(black) 2(white)
    // hence the matrix can be seen as a 225(15*15)-trits ternary array
    // here convert into base81, which means every 4 trits are mapped into a single letter
    // then it can be presented in a 57-characters string, much shorter
    fn to_base81_string(self: &BoardMatrix) -> String {
        let mut code: usize = 0;
        let mut base = 1;

        let mut codes = Vec::with_capacity(SIZE * SIZE);

        for row in 0..SIZE {
            for col in 0..SIZE {
                // from bottom to top, from right to left
                let value: usize = self[(SIZE - row - 1, SIZE - col - 1)].into();
                if value > 2 {
                    panic!("Unexpected value {} in matrix", value);
                } else {
                    code = code + value * base;
                }
                if base < 27 {
                    // 3^3, every four-trits convert into a letter
                    base = base * 3;
                } else {
                    codes.push(BASE81_TABLE[code]);
                    base = 1;
                    code = 0; // reset
                }
            }
        }
        codes.push(BASE81_TABLE[code]);
        codes.into_iter().rev().collect::<String>()
    }

    fn get_all_appearances(
        self: &BoardMatrix,
        last_move: (usize, usize),
        answer: (usize, usize),
    ) -> Vec<(BoardMatrix, (usize, usize), (usize, usize))> {
        let mut hash_map = HashMap::new();

        let mut m1 = self.clone();
        flip_main_diagonal(&mut m1);
        let id1 = m1.to_base81_string();
        let answer1 = main_diagonal_transform(answer);
        let last_move1 = main_diagonal_transform(last_move);

        let mut m2 = m1.clone();
        flip_vertical(&mut m2);
        let id2 = m2.to_base81_string();
        let answer2 = vertical_transform(answer1);
        let last_move2 = vertical_transform(last_move1);

        let mut m3 = self.clone();
        flip_anti_diagonal(&mut m3);
        let id3 = m3.to_base81_string();
        let answer3 = anti_diagonal_transform(answer);
        let last_move3 = anti_diagonal_transform(last_move);

        let mut m4 = m3.clone();
        flip_vertical(&mut m4);
        let id4 = m4.to_base81_string();
        let answer4 = vertical_transform(answer3);
        let last_move4 = vertical_transform(last_move3);

        let mut m5 = self.clone();
        flip_horizontal(&mut m5);
        let id5 = m5.to_base81_string();
        let answer5 = horizontal_transform(answer);
        let last_move5 = horizontal_transform(last_move);

        let mut m6 = self.clone();
        flip_vertical(&mut m6);
        let id6 = m6.to_base81_string();
        let answer6 = vertical_transform(answer);
        let last_move6 = vertical_transform(last_move);

        let mut m7 = m6.clone();
        flip_horizontal(&mut m7);
        let id7 = m7.to_base81_string();
        let answer7 = horizontal_transform(answer6);
        let last_move7 = horizontal_transform(last_move6);

        hash_map.insert(id1, (m1, last_move1, answer1));
        hash_map.insert(id2, (m2, last_move2, answer2));
        hash_map.insert(id3, (m3, last_move3, answer3));
        hash_map.insert(id4, (m4, last_move4, answer4));
        hash_map.insert(id5, (m5, last_move5, answer5));
        hash_map.insert(id6, (m6, last_move6, answer6));
        hash_map.insert(id7, (m7, last_move7, answer7));
        hash_map.insert(self.to_base81_string(), (self.clone(), last_move, answer));

        hash_map.into_values().collect()
    }

    fn get_blacks_whites(self: &Self) -> (BoardMatrix /*black*/, BoardMatrix /*white */) {
        let mut black = BoardMatrix::zeros();
        let mut white = BoardMatrix::zeros();

        for row in 0..SIZE {
            for col in 0..SIZE {
                match self[(row, col)] {
                    BLACK => black[(row, col)] = 1,
                    WHITE => white[(row, col)] = 1,
                    _ => (),
                };
            }
        }

        (black, white)
    }
}

static BASE81_TABLE: &'static [char] = &[
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
    'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B',
    'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
    'V', 'W', 'X', 'Y', 'Z', '.', '-', ':', '+', '=', '^', '!', '*', '?', '<', '>', '(', ')', '[',
    ']', '{', '}', '@', '#',
];

static BASE81_REVERSE_TABLE: &'static [u8] = &[
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 2, 1, 1, 2, 9, 9, 9, 9, 2, 2, 2, 2, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    2, 2, 0, 1, 2, 2, 0, 2, 2, 1, 2, 0, 2, 1, 0, 2, 9, 9, 9, 9, 2, 1, 0, 0, 2, 0, 2, 2, 9, 9, 9, 9,
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 2, 0, 0, 0, 2, 1,
    0, 0, 2, 2, 0, 1, 0, 0, 2, 1, 0, 1, 9, 9, 9, 9, 2, 1, 2, 2, 2, 1, 1, 0, 2, 2, 0, 0, 2, 1, 2, 1,
    2, 2, 2, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 2, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 0,
    1, 1, 2, 1, 1, 1, 2, 2, 1, 2, 0, 0, 1, 2, 0, 1, 1, 2, 0, 2, 1, 2, 1, 0, 1, 2, 1, 1, 1, 2, 1, 2,
    1, 2, 2, 0, 1, 2, 2, 1, 1, 2, 2, 2, 2, 0, 0, 0, 2, 0, 0, 1, 2, 0, 0, 2, 2, 0, 1, 0, 2, 0, 1, 1,
    2, 0, 1, 2, 2, 0, 2, 0, 2, 0, 2, 1, 2, 2, 1, 0, 9, 9, 9, 9, 2, 2, 1, 1, 2, 1, 1, 1, 9, 9, 9, 9,
    9, 9, 9, 9, 0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 2, 0, 1, 2, 0, 0, 1, 2, 1,
    0, 1, 2, 2, 0, 2, 0, 0, 0, 2, 0, 1, 0, 2, 0, 2, 0, 2, 1, 0, 0, 2, 1, 1, 0, 2, 1, 2, 0, 2, 2, 0,
    0, 2, 2, 1, 0, 2, 2, 2, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 2, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 2,
    1, 0, 2, 0, 1, 0, 2, 1, 1, 0, 2, 2, 2, 2, 1, 2, 9, 9, 9, 9, 2, 2, 2, 0,
];

#[test]
fn generate_reverse_table() {
    let mut coll: Vec<_> = BASE81_TABLE
        .iter()
        .enumerate()
        .map(|(index, ch)| {
            let mut trits: [u8; 4] = [0; 4];
            trits[0] = u8::try_from(index / 27).unwrap();
            let num = index % 27;
            trits[1] = u8::try_from(num / 9).unwrap();
            let num = num % 9;
            trits[2] = u8::try_from(num / 3).unwrap();
            let num = num % 3;
            trits[3] = u8::try_from(num).unwrap();

            (*ch as u8, trits)
        })
        .collect();

    coll.sort_by(|x, y| y.0.cmp(&x.0));

    let first = coll.remove(0);
    let mut source_code = format!(
        "{}, {}, {}, {},  ",
        first.1[0], first.1[1], first.1[2], first.1[3]
    );
    let mut count = 0;

    for code in (0..first.0).rev() {
        if !coll.is_empty() && coll[0].0 == code {
            let current = coll.remove(0);
            source_code.insert_str(
                0,
                &format!(
                    "{}, {}, {}, {},  ",
                    current.1[0], current.1[1], current.1[2], current.1[3]
                ),
            );
        } else {
            source_code.insert_str(0, "9, 9, 9, 9,  ");
        }
        count = count + 1;
        if count % 5 == 0 {
            source_code.insert_str(0, "\n");
        }
    }

    print!("{}", &source_code);
}

#[test]
fn test_base81() {
    use rand::Rng; // 0.8.0

    let m = BoardMatrix::from_fn(|_, _| rand::thread_rng().gen_range(0..3));
    let text = m.to_base81_string();
    println!("{}", &text);

    assert_eq!(m, BoardMatrixExtension::from_base81_string(&text));
}

fn swap(m: &mut BoardMatrix, pos1: (usize, usize), pos2: (usize, usize)) {
    let t = m[pos1];
    m[pos1] = m[pos2];
    m[pos2] = t;
}

// The diagonal from the top left corner to the bottom right corner of a square matrix
// is called the main diagonal or leading diagonal.
// The other diagonal from the top right to the bottom left corner
// is called antidiagonal or counterdiagonal.

/*
1,2,3     9,6,3
4,5,6  -> 8,5,2
7,8,9     7,4,1
 */
pub fn flip_anti_diagonal(m: &mut BoardMatrix) {
    for row in 0..SIZE {
        for col in 0..(SIZE - row) {
            swap(m, (row, col), anti_diagonal_transform((row, col)));
        }
    }
}

fn anti_diagonal_transform((row, col): (usize, usize)) -> (usize, usize) {
    return (SIZE - 1 - col, SIZE - 1 - row);
}

/*
1,2,3     1,4,7
4,5,6  -> 2,5,8
7,8,9     3,6,9
 */
pub fn flip_main_diagonal(m: &mut BoardMatrix) {
    for row in 0..SIZE {
        for col in (row + 1)..SIZE {
            swap(m, (row, col), main_diagonal_transform((row, col)));
        }
    }
}
fn main_diagonal_transform((row, col): (usize, usize)) -> (usize, usize) {
    return (col, row);
}

/*
1,2,3     7,8,9
4,5,6  -> 4,5,6
7,8,9     1,2,3
 */
pub fn flip_vertical(m: &mut BoardMatrix) {
    for row in 0..(SIZE / 2) {
        for col in 0..SIZE {
            swap(m, (row, col), vertical_transform((row, col)));
        }
    }
}
fn vertical_transform((row, col): (usize, usize)) -> (usize, usize) {
    return (SIZE - 1 - row, col);
}

/*
1,2,3     3,2,1
4,5,6  -> 6,5,2
7,8,9     9,8,2
 */
#[allow(dead_code)]
pub fn flip_horizontal(m: &mut BoardMatrix) {
    for row in 0..SIZE {
        for col in 0..(SIZE / 2) {
            swap(m, (row, col), horizontal_transform((row, col)));
        }
    }
}
fn horizontal_transform((row, col): (usize, usize)) -> (usize, usize) {
    return (row, SIZE - 1 - col);
}

/*
1,2,3     9,6,3       7,4,1
4,5,6  -> 8,5,2  ->   8,5,2
7,8,9     7,4,1       9,6,3
 */
#[allow(dead_code)]
pub fn rotate_right(m: &mut BoardMatrix) {
    flip_anti_diagonal(m);
    flip_vertical(m);
}

/*
1,2,3     1,4,7     3,6,9
4,5,6  -> 2,5,8  -> 2,5,8
7,8,9     3,6,9     1,2,3
 */
#[allow(dead_code)]
pub fn rotate_left(m: &mut BoardMatrix) {
    flip_main_diagonal(m);
    flip_vertical(m);
}
