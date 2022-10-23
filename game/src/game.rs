use std::cmp;
use std::collections::HashSet;

pub const BOARD_SIZE: usize = 15;
pub const CHANNELS: usize = 4;
const DISTANCE: usize = 2;
const NO_STONE: u8 = 0;
const AVAILABLE: u8 = 1;

pub type SquareMatrix<T = f32> = [[T; BOARD_SIZE]; BOARD_SIZE];
pub type StateTensor<T = f32> = [SquareMatrix<T>; CHANNELS];

pub trait StateTensorExtension {
    fn shape(self: &Self) -> [i64; 4];
}
impl<T> StateTensorExtension for [StateTensor<T>] {
    fn shape(self: &Self) -> [i64; 4] {
        return [
            self.len() as i64,
            CHANNELS as i64,
            BOARD_SIZE as i64,
            BOARD_SIZE as i64,
        ];
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Color {
    None = 0,
    Black = 1,
    White = 2,
}

impl From<u8> for Color {
    fn from(v: u8) -> Self {
        if v > 0 {
            if v % 2 == 0 {
                Color::White
            } else {
                Color::Black
            }
        } else {
            Color::None
        }
    }
}

impl From<f32> for Color {
    fn from(v: f32) -> Self {
        unimplemented!()
    }
}

pub trait SquaredMatrixExtension<T>
where
    T: From<u8>,
    Color: From<T>,
{
    fn swap(self: &mut Self, x: (usize, usize), y: (usize, usize));

    /*
    1,2,3     7,8,9
    4,5,6  -> 4,5,6
    7,8,9     1,2,3
     */
    fn flip_top_bottom(self: &mut Self);

    /*
    1,2,3     3,2,1
    4,5,6  -> 6,5,4
    7,8,9     9,8,7
    */
    fn flip_left_right(self: &mut Self);

    /*
    1,2,3     1,4,7
    4,5,6  -> 2,5,8
    7,8,9     3,6,9
     */
    fn flip_over_main_diagonal(self: &mut Self);

    /*
    1,2,3     9,6,3
    4,5,6  -> 8,5,2
    7,8,9     7,4,1
     */
    fn flip_over_anti_diagonal(self: &mut Self);

    fn scan_direction<F>(
        self: &Self,
        pos: (usize, usize),
        color: Color,
        get_neighbor: F,
    ) -> [Vec<(usize, usize)>; 5]
    where
        F: Fn((usize, usize)) -> Option<(usize, usize)>;

    fn scan_row<F1, F2>(
        self: &mut Self,
        pos: (usize, usize),
        color: Color,
        get_neighbor_of_main_direction: F1,
        get_neighbor_of_opposite_direction: F2,
        state: &mut StoneScanResult,
    ) where
        F1: Fn((usize, usize)) -> Option<(usize, usize)>,
        F2: Fn((usize, usize)) -> Option<(usize, usize)>;

    fn scan_continuous_stone(
        self: &mut Self,
        position: (usize, usize),
        color: Color,
    ) -> StoneScanResult;

    fn is_forbidden(self: &mut Self, position: (usize, usize)) -> bool;
}

#[derive(Debug, Clone)]
pub struct RenjuBoard {
    matrix: SquareMatrix<u8>,                 // 0=BLANK; 1=BLACK; 2=WHITE
    available_matrix: SquareMatrix<u8>,       // available positions, 1=available
    last_move: Option<(usize, usize)>,        // recent move
    stones: u8,                               // number of moves
    black_win_moves: HashSet<(usize, usize)>, // just one black move in specified position to win
    white_win_moves: HashSet<(usize, usize)>, // just one white move in specified position to win
}

impl Default for RenjuBoard {
    fn default() -> Self {
        let mut available = SquareMatrix::default();
        available[BOARD_SIZE / 2][BOARD_SIZE / 2] = AVAILABLE;
        Self {
            matrix: SquareMatrix::default(),
            available_matrix: available,
            last_move: None,
            stones: 0,
            black_win_moves: HashSet::new(),
            white_win_moves: HashSet::new(),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum TerminalState {
    AvailableMoves(Vec<(usize, usize)>),
    BlackWon,
    WhiteWon,
    Draw, // tie
}

impl Default for TerminalState {
    fn default() -> Self {
        TerminalState::AvailableMoves(vec![(BOARD_SIZE / 2, BOARD_SIZE / 2)])
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

#[derive(Debug)]
pub struct StoneScanResult {
    color: Color,
    three_count: u8, // live three
    four_count: u8,  // open four
    has_five: bool,
    over_five: bool,
    black_win_moves: Vec<(usize, usize)>, // white has to choose one of them in next turn if non-empty
    white_win_moves: Vec<(usize, usize)>, // black has to choose one of them in next turn if non-empty
}

impl StoneScanResult {
    fn new(color: Color) -> Self {
        Self {
            color: color,
            three_count: 0,
            four_count: 0,
            has_five: false,
            over_five: false,
            black_win_moves: Vec::new(),
            white_win_moves: Vec::new(),
        }
    }

    #[allow(dead_code)]
    pub fn get_black_win_moves(self: &Self) -> &Vec<(usize, usize)> {
        &self.black_win_moves
    }
    #[allow(dead_code)]
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

impl RenjuBoard {
    pub fn get_matrix(self: &Self) -> &SquareMatrix<u8> {
        &self.matrix
    }

    // next move is black?
    pub fn is_black_turn(self: &Self) -> bool {
        self.stones % 2 == 0
    }

    pub fn get_last_move(self: &Self) -> Option<(usize, usize)> {
        self.last_move
    }

    pub fn get_stones(self: &Self) -> u8 {
        self.stones
    }

    // Add a new stone into board
    pub fn do_move(self: &mut Self, pos: (usize, usize)) -> TerminalState {
        // this assert only needs for training
        //assert!(
        //    self.available_matrix[pos.0][pos.1] == (Color::Black as u8) || self.last_move.is_none()
        //);
        assert_eq!(self.matrix[pos.0][pos.1], NO_STONE);

        let color = if (self.stones % 2) == 0 {
            Color::Black
        } else {
            Color::White
        };

        let is_black_turn = self.is_black_turn();

        // update board pieces
        let mut state = self.matrix.scan_continuous_stone(pos, color);

        // update status
        self.stones += 1;
        self.matrix[pos.0][pos.1] = self.stones;
        self.last_move = Some(pos);

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

        // update available positions
        // only allow srounding positions
        let (start_row, start_col, end_row, end_col) = if self.stones == 2
            && self.matrix[BOARD_SIZE / 2][BOARD_SIZE / 2] == 1
            && pos.0.abs_diff(BOARD_SIZE / 2) <= 1
            && pos.1.abs_diff(BOARD_SIZE / 2) <= 1
        {
            let start_row = BOARD_SIZE / 2 - 2;
            let start_col = BOARD_SIZE / 2 - 2;
            let end_row = BOARD_SIZE / 2 + 2;
            let end_col = BOARD_SIZE / 2 + 2;

            (start_row, start_col, end_row, end_col)
        } else {
            let distance = if self.stones <= 1 { 1 } else { DISTANCE };
            let start_row = if pos.0 > distance {
                pos.0 - distance
            } else {
                0
            };
            let start_col = if pos.1 > distance {
                pos.1 - distance
            } else {
                0
            };
            let end_row = cmp::min(BOARD_SIZE - 1, pos.0 + distance);
            let end_col = cmp::min(BOARD_SIZE - 1, pos.1 + distance);
            (start_row, start_col, end_row, end_col)
        };

        for row in start_row..=end_row {
            for col in start_col..=end_col {
                self.available_matrix[row][col] = if self.matrix[row][col] == NO_STONE {
                    AVAILABLE
                } else {
                    NO_STONE
                };
            }
        }

        if self.stones as usize >= BOARD_SIZE * BOARD_SIZE {
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

                // exclude forbidden moves if there is any other available one
                return TerminalState::AvailableMoves(self.get_available_moves());
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
                return TerminalState::AvailableMoves(self.get_available_moves());
            }
        }
    }

    pub fn is_forbidden(self: &mut Self, position: (usize, usize)) -> bool {
        self.is_black_turn() && self.matrix.is_forbidden(position)
    }

    // return available moves
    fn get_available_moves(self: &Self) -> Vec<(usize /*row*/, usize /*col*/)> {
        // first black
        if self.last_move.is_none() {
            return vec![(BOARD_SIZE / 2, BOARD_SIZE / 2)];
        }

        let mut positions = Vec::new();

        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                if self.available_matrix[row][col] == Color::Black as u8 {
                    positions.push((row, col));
                }
            }
        }

        positions
    }

    pub fn get_state_tensor(self: &Self) -> StateTensor {
        let mut state_matrix = StateTensor::default();

        let (black_index, white_index) = if self.is_black_turn() {
            // next move is black
            (0, 1)
        } else {
            // next move is white
            (1, 0)
        };

        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                let val = self.matrix[row][col];
                if val > NO_STONE {
                    if val % 2 == 0 {
                        state_matrix[white_index][row][col] = 1f32; // even is white
                    } else {
                        state_matrix[black_index][row][col] = 1f32; // odds is black
                    }
                }
            }
        }

        if let Some((row, col)) = self.last_move {
            state_matrix[2][row][col] = 1f32;
        }

        if self.is_black_turn() {
            // next move is black
            state_matrix[3] = [[1f32; BOARD_SIZE]; BOARD_SIZE];
        }
        state_matrix
    }
}

impl<T> SquaredMatrixExtension<T> for SquareMatrix<T>
where
    T: Copy + bytemuck::Pod + std::fmt::Debug + PartialEq + Default + From<u8>,
    Color: From<T>,
{
    fn swap(self: &mut Self, x: (usize, usize), y: (usize, usize)) {
        let temp = self[x.0][x.1];
        self[x.0][x.1] = self[y.0][y.1];
        self[y.0][y.1] = temp;
    }

    /*
    1,2,3     7,8,9
    4,5,6  -> 4,5,6
    7,8,9     1,2,3
     */
    fn flip_top_bottom(self: &mut Self) {
        for row in 0..(BOARD_SIZE / 2) {
            for col in 0..BOARD_SIZE {
                self.swap((row, col), (BOARD_SIZE - 1 - row, col));
            }
        }
    }

    /*
    1,2,3     3,2,1
    4,5,6  -> 6,5,4
    7,8,9     9,8,7
     */
    fn flip_left_right(self: &mut Self) {
        for row in 0..BOARD_SIZE {
            for col in 0..(BOARD_SIZE / 2) {
                self.swap((row, col), (row, BOARD_SIZE - 1 - col));
            }
        }
    }

    /*
    1,2,3     1,4,7
    4,5,6  -> 2,5,8
    7,8,9     3,6,9
     */
    fn flip_over_main_diagonal(self: &mut Self) {
        for row in 0..BOARD_SIZE {
            for col in (row + 1)..BOARD_SIZE {
                self.swap((row, col), (col, row));
            }
        }
    }

    /*
    1,2,3     9,6,3
    4,5,6  -> 8,5,2
    7,8,9     7,4,1
     */
    fn flip_over_anti_diagonal(self: &mut Self) {
        for row in 0..BOARD_SIZE {
            for col in 0..(BOARD_SIZE - row) {
                self.swap((row, col), (BOARD_SIZE - 1 - col, BOARD_SIZE - 1 - row));
            }
        }
    }

    /// Scan same color stones and blanks in a direction of position
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
        let mut expected_color = color;
        let mut index = 0;
        while index < vectors.len() {
            match get_neighbor(current) {
                None => break, // out of board
                Some((row, col)) => {
                    let current_color = Color::from(self[row][col]);
                    if current_color == expected_color {
                        current = (row, col);
                        vectors[index].push(current);
                    } else {
                        index += 1;
                        match current_color {
                            Color::White if color == Color::White => expected_color = Color::White,
                            Color::Black if color == Color::Black => expected_color = Color::Black,
                            Color::None => expected_color = Color::None,
                            _ => break, // opposite player's piece
                        }
                    }
                }
            }
        } // end of while
        vectors
    }

    // https://blog.csdn.net/JkSparkle/article/details/822873
    fn scan_row<F1, F2>(
        self: &mut Self,
        pos: (usize, usize),
        color: Color,
        get_neighbor_of_main_direction: F1,
        get_neighbor_of_opposite_direction: F2,
        state: &mut StoneScanResult,
    ) where
        F1: Fn((usize, usize)) -> Option<(usize, usize)>,
        F2: Fn((usize, usize)) -> Option<(usize, usize)>,
    {
        let main_direction: [Vec<(usize, usize)>; 5] =
            self.scan_direction(pos, color, get_neighbor_of_main_direction);
        let opposite_direction: [Vec<(usize, usize)>; 5] =
            self.scan_direction(pos, color, get_neighbor_of_opposite_direction);

        let continuous_blacks = main_direction[0].len() + opposite_direction[0].len() + 1;

        match continuous_blacks {
            // ┼┼┼┼┼┼┼
            // ┼●●●●●┼
            // ┼┼┼┼┼┼┼
            5 => state.has_five = true, // five in a row, no forbidden

            //////////////////// four continuous blacks
            4 => {
                let mut four = false;
                if main_direction[1].len() > 0 {
                    // ┼●●●●┼
                    // ^
                    if color == Color::Black {
                        if !self.is_forbidden(main_direction[1][0]) {
                            four = true;
                            state.black_win_moves.push(main_direction[1][0]);
                        }
                    } else {
                        four = true;
                        state.white_win_moves.push(main_direction[1][0]);
                    }
                }

                // ┼●●●●┼
                //      ^
                if opposite_direction[1].len() > 0 {
                    if color == Color::Black {
                        if !self.is_forbidden(opposite_direction[1][0]) {
                            four = true;
                            state.black_win_moves.push(opposite_direction[1][0]);
                        }
                    } else {
                        four = true;
                        state.white_win_moves.push(opposite_direction[1][0]);
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
                if main_direction[1].len() == 1 && main_direction[2].len() == 1 {
                    if color == Color::Black {
                        if !self.is_forbidden(main_direction[1][0]) {
                            state.four_count += 1;
                            state.black_win_moves.push(main_direction[1][0]);
                        }
                    } else {
                        state.four_count += 1;
                        state.white_win_moves.push(main_direction[1][0]);
                    }
                }

                // ┼●●●┼●┼
                //     ^
                if opposite_direction[1].len() == 1 && opposite_direction[2].len() == 1 {
                    if color == Color::Black {
                        if !self.is_forbidden(opposite_direction[1][0]) {
                            state.four_count += 1;
                            state.black_win_moves.push(opposite_direction[1][0]);
                        }
                    } else {
                        state.four_count += 1;
                        state.white_win_moves.push(opposite_direction[1][0]);
                    }
                }

                let mut live_three = false;
                // ┼┼●●●┼
                if (
                    main_direction[1].len() > 2 // more than 2 blanks
                    || (main_direction[1].len() == 2 && main_direction[2].is_empty())
                    // or two blanks without black next to it
                ) && (
                    opposite_direction[1].len() > 1 // more than 1 blanks
                        || (opposite_direction[1].len() == 1 && opposite_direction[2].is_empty())
                    // or one blank without black next to it
                ) {
                    if color == Color::Black {
                        // ┼┼●●●┼
                        //  ^
                        live_three = !self.is_forbidden(main_direction[1][0]);
                    } else {
                        live_three = true;
                    }
                }

                // ┼●●●┼┼
                if (opposite_direction[1].len() > 2
                    || (opposite_direction[1].len() == 2 && opposite_direction[2].is_empty()))
                    && (main_direction[1].len() > 1
                        || (main_direction[1].len() == 1 && main_direction[2].is_empty()))
                {
                    if color == Color::Black {
                        // ┼●●●┼┼
                        //     ^
                        live_three = !self.is_forbidden(opposite_direction[1][0]);
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
                if main_direction[1].len() == 1 && main_direction[2].len() == 2 {
                    if color == Color::Black {
                        if !self.is_forbidden(main_direction[1][0]) {
                            state.four_count += 1;
                            state.black_win_moves.push(main_direction[1][0]);
                        }
                    } else {
                        state.four_count += 1;
                        state.white_win_moves.push(main_direction[1][0]);
                    }
                }

                // ●●┼●●
                if opposite_direction[1].len() == 1 && opposite_direction[2].len() == 2 {
                    if color == Color::Black {
                        if !self.is_forbidden(opposite_direction[1][0]) {
                            state.four_count += 1;
                            state.black_win_moves.push(opposite_direction[1][0]);
                        }
                    } else {
                        state.four_count += 1;
                        state.white_win_moves.push(opposite_direction[1][0]);
                    }
                }

                // ┼●┼●●┼
                if main_direction[1].len() == 1
                    && main_direction[2].len() == 1
                    && (main_direction[3].len() > 1
                        || (main_direction[3].len() == 1 && main_direction[4].is_empty()))
                    && (opposite_direction[1].len() > 1
                        || (opposite_direction[1].len() == 1 && opposite_direction[2].is_empty()))
                {
                    if color == Color::Black {
                        // ┼●┼●●┼
                        //   ^
                        if !self.is_forbidden(main_direction[1][0]) {
                            state.three_count += 1;
                        }
                    } else {
                        state.three_count += 1;
                    }
                }

                // ┼●●┼●┼
                if opposite_direction[1].len() == 1
                    && opposite_direction[2].len() == 1
                    && (opposite_direction[3].len() > 1
                        || (opposite_direction[3].len() == 1 && opposite_direction[4].is_empty()))
                    && (main_direction[1].len() > 1
                        || (main_direction[1].len() == 1 && main_direction[2].is_empty()))
                {
                    if color == Color::Black {
                        // ┼●●┼●┼
                        //    ^
                        if !self.is_forbidden(opposite_direction[1][0]) {
                            state.three_count += 1;
                        }
                    } else {
                        state.three_count += 1;
                    }
                }
            }

            //////////////////// single black
            1 => {
                // ●●●┼●
                if main_direction[1].len() == 1 && main_direction[2].len() == 3 {
                    if color == Color::Black {
                        // ●●●┼●
                        //    ^
                        if !self.is_forbidden(main_direction[1][0]) {
                            state.four_count += 1;
                            state.black_win_moves.push(main_direction[1][0]);
                        }
                    } else {
                        state.four_count += 1;
                        state.white_win_moves.push(main_direction[1][0]);
                    }
                }

                // ●┼●●●
                if opposite_direction[1].len() == 1 && opposite_direction[2].len() == 3 {
                    if color == Color::Black {
                        // ●●●┼●
                        //    ^
                        if !self.is_forbidden(opposite_direction[1][0]) {
                            state.four_count += 1;
                            state.black_win_moves.push(opposite_direction[1][0]);
                        }
                    } else {
                        state.four_count += 1;
                        state.white_win_moves.push(opposite_direction[1][0]);
                    }
                }

                // ┼●●┼●┼
                if main_direction[1].len() == 1
                    && main_direction[2].len() == 2
                    && (main_direction[3].len() > 1
                        || (main_direction[3].len() == 1 && main_direction[4].is_empty()))
                    && (opposite_direction[1].len() > 1
                        || (opposite_direction[1].len() == 1 && opposite_direction[2].is_empty()))
                {
                    if color == Color::Black {
                        // ┼●●┼●┼
                        //    ^
                        if !self.is_forbidden(main_direction[1][0]) {
                            state.three_count += 1;
                        }
                    } else {
                        state.three_count += 1;
                    }
                }

                // ┼●┼●●┼
                if opposite_direction[1].len() == 1
                    && opposite_direction[2].len() == 2
                    && (opposite_direction[3].len() > 1
                        || (opposite_direction[3].len() == 1 && opposite_direction[4].is_empty()))
                    && (main_direction[1].len() > 1
                        || (main_direction[1].len() == 1 && main_direction[2].is_empty()))
                {
                    if color == Color::Black {
                        // ┼●●┼●┼
                        //    ^
                        if !self.is_forbidden(opposite_direction[1][0]) {
                            state.three_count += 1;
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

    fn scan_continuous_stone(
        self: &mut Self,
        position: (usize, usize),
        color: Color,
    ) -> StoneScanResult {
        //assert_eq!(self[position.0][position.1], NO_STONE);
        self[position.0][position.1] = match color {
            Color::Black => 255u8.into(),
            Color::White => 254u8.into(),
            _ => 0u8.into(),
        };

        let mut state = StoneScanResult::new(color);

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
                if col < BOARD_SIZE - 1 {
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
                if row < BOARD_SIZE - 1 {
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
                if row < BOARD_SIZE - 1 && col < BOARD_SIZE - 1 {
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
                if row > 0 && col < BOARD_SIZE - 1 {
                    Some((row - 1, col + 1))
                } else {
                    None
                }
            },
            |(row, col)| {
                if row < BOARD_SIZE - 1 && col > 0 {
                    Some((row + 1, col - 1))
                } else {
                    None
                }
            },
            &mut state,
        );

        // restore
        self[position.0][position.1] = NO_STONE.into();

        state
    }

    // check if the black at (usize, usize) is forbidden
    // Double three – Black cannot place a stone that builds two separate lines with three black stones in unbroken rows (i.e. rows not blocked by white stones).
    // Double four – Black cannot place a stone that builds two separate lines with four black stones in a row.
    // Overline – six or more black stones in a row.
    fn is_forbidden(self: &mut Self, position: (usize, usize)) -> bool {
        //assert_eq!(self[position.0][position.1], NO_STONE);
        let state = self.scan_continuous_stone(position, Color::Black);
        !state.has_five && (state.over_five || state.three_count > 1 || state.four_count > 1)
    }
}

#[test]
fn test_get_state_tensor() {
    let mut board = RenjuBoard::default();
    board.do_move((7, 7));
    board.do_move((7, 8));
    board.do_move((8, 8));
    /*
    ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
    │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
    │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
    │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
    │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
    │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
    │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
    │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
    │   │   │   │   │   │   │   │ O │ X │   │   │   │   │   │   │
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
    │   │   │   │   │   │   │   │   │ O │   │   │   │   │   │   │
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
    │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
    │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
    │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
    │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
    │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
    │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │
    └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
         */

    let state_tensor = board.get_state_tensor();
    let expected_state_tensor: StateTensor<f32> = [
        [
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        ],
        [
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        ],
        [
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        ],
        [
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
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        ],
    ];
    assert_eq!(state_tensor, expected_state_tensor);
}
