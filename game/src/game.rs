


use std::{collections::HashMap};

use nalgebra::{ SMatrix};
use num_traits::identities::{ Zero };

// Statically sized and statically allocated 2x3 matrix using 32-bit floats.
const SIZE: usize = 15;
const BLANK: u8 = 0;
const BLACK: u8 = 1;
const WHITE: u8 = 2;
pub type BoardMatrix = SMatrix<u8, SIZE, SIZE>;

pub trait Board {
  fn print(self: &Self);
  
  fn from_base81_string(text : &str) -> Self;
  fn to_base81_string(self: &Self) -> String;
  fn for_each_piece<F : FnMut(usize, usize, u8)>(self : &Self, cb : F);
  fn is_over(self : &Self) -> bool;
  fn check_black_in_position( self : &Self, position:(usize, usize)) -> RowState;
  fn is_forbidden(self : &Self, position:(usize, usize)) -> bool;
  fn get_all_appearances(self : &Self, last_move : (usize, usize),  answer : (usize, usize)) -> Vec<(BoardMatrix, (usize, usize), (usize, usize))>;
  fn get_blacks_whites(self : &Self) -> (BoardMatrix/*black*/, BoardMatrix/*white */);
}


#[derive(Debug)]
pub struct RowState {
  three_count : u8, // live three
  four_count : u8,  // open four
  has_five : bool,
  over_five : bool,
}

impl Default for RowState {
  fn default() -> Self {
      Self { three_count: 0, four_count: 0, has_five: false, over_five: false }
  }
}

#[allow(dead_code)]
impl RowState {
  pub fn get_live_three_count(self : &Self) -> u8 { self.three_count }
  pub fn get_open_four_count(self : &Self) -> u8 { self.four_count }
  pub fn has_five(self : &Self) -> bool { self.has_five }
  pub fn over_five(self : &Self) -> bool { self.over_five }
}



impl Board for BoardMatrix {

  fn check_black_in_position( self : &Self, position:(usize, usize)) -> RowState {
    let mut state = RowState::default();

    // horizontal
    scan_row( self,
      position,
      |(row, col)| {
        if col > 0 {
          Some((row, col-1))
        } else {
          None
        }
      }, 
      |(row, col)| {
        if col < SIZE - 1 {
          Some((row, col+1))
        } else {
          None
        }
      },
      &mut state
    );


    // vertical
    scan_row( self,
      position,
      |(row, col)| {
        if row > 0 {
          Some((row-1, col))
        } else {
          None
        }
      }, 
      |(row, col)| {
        if row < SIZE - 1 {
          Some((row+1, col))
        } else {
          None
        }
      },
      &mut state
    );


    // main diagonal
    scan_row( self,
      position,
      |(row, col)| {
        if row > 0 && col > 0 {
          Some((row-1, col-1))
        } else {
          None
        }
      }, 
      |(row, col)| {
        if row < SIZE - 1 && col < SIZE - 1 {
          Some((row+1, col+1))
        } else {
          None
        }
      },
      &mut state
    );

    // anti diagonal
    scan_row( self,
      position,
      |(row, col)| {
        if row > 0 && col < SIZE - 1 {
          Some((row-1, col+1))
        } else {
          None
        }
      }, 
      |(row, col)| {
        if row < SIZE - 1 && col > 0 {
          Some((row+1, col-1))
        } else {
          None
        }
      },
      &mut state
    );

    state
  }


  // check if the black at (usize, usize) is forbidden
  // Double three – Black cannot place a stone that builds two separate lines with three black stones in unbroken rows (i.e. rows not blocked by white stones).
  // Double four – Black cannot place a stone that builds two separate lines with four black stones in a row.
  // Overline – six or more black stones in a row.
  fn is_forbidden(self : &Self, position:(usize, usize)) -> bool
  {
    let state = self.check_black_in_position(position);
    !state.has_five &&
    (
      state.over_five ||
      state.three_count > 1 ||
      state.four_count > 1
    )
  }

  fn is_over(self : &Self) -> bool {

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

        

        if col > 0 { // horizontal
          horizontal_len += states[row][col-1][index][0];
          if horizontal_len > 4 {
            return true;
          }
        }
        if row > 0 { // vertical
          vertical_len += states[row-1][col][index][1];
          if vertical_len > 4 {
            return true;
          }

          if col > 0 { // main diagonal
            main_diagonal_len += states[row-1][col-1][index][2];
            if main_diagonal_len > 4 {
              return true;
            }
          }
          if col < SIZE - 1 { // anti diagonal
            anti_diagonal_len += states[row-1][col+1][index][3];
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
  }

  fn for_each_piece<F : FnMut(usize, usize, u8)>(self : &Self, mut cb : F) {
    let mut vector = Vec::<(usize, usize)>::new();
    let mut expected_value = 1u8;
    for row in 0..SIZE {
      for col in 0..SIZE {
        let value = self[(row, col)];
        if value != 0 {
          if value == expected_value { // expected
            cb( row, col, value);
            expected_value = match expected_value {
                BLACK => WHITE,
                WHITE => BLACK,
                _ => panic!("incorrect value"),
            };
            // accumulated
            if let Some(more) = vector.pop() {
              cb( more.0, more.1, expected_value);
              expected_value = match expected_value {
                  BLACK => WHITE,
                  WHITE => BLACK,
                  _ => panic!("incorrect value"),
              };
            }
          }
          else {
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

  fn from_base81_string(text : &str) -> Self {
    let mut m = BoardMatrix::zero();

    let mut count = 0;
    text.chars().rev().enumerate().for_each( |(_, ch)| {

      
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


  fn print(self: &BoardMatrix)
  {
    let mut text = String::with_capacity((SIZE+1)*(SIZE+2));
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

        match self[(row,col)] {
          1 => text.push_str(" O "), // black
          2 => text.push_str(" X "), // white
          0 => text.push_str("   "),
          _ => panic!("Unexpected value {} in matrix", self[(row,col)]),
        }
      }
      text.push_str("│\n");

      for i in 0..SIZE {
        if i == 0 {
          if row < SIZE - 1 {
            text.push_str("├");
          }
          else {
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
      }
      else {
        text.push_str("┘\n");
      }
      
    }
    

    println!("{}", text);
  }

    

  // The borad only contains three types of values: 0(blank) 1(black) 2(white)
  // hence the matrix can be seen as a 225(15*15)-trits ternary array
  // here convert into base81, which means every 4 trits are mapped into a single letter
  // then it can be presented in a 57-characters string, much shorter
  fn to_base81_string(self: &BoardMatrix) -> String
  {
    let mut code : usize = 0;
    let mut base = 1;

    let mut codes = Vec::with_capacity(SIZE*SIZE);

    for row in 0..SIZE {
      for col in 0..SIZE {
        // from bottom to top, from right to left
        let value : usize =  self[(SIZE-row-1, SIZE-col-1)].into();
        if value > 2 {
          panic!("Unexpected value {} in matrix", value);
        } else {
          code = code + value * base;
        }
        if base < 27 { // 3^3, every four-trits convert into a letter
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




  fn get_all_appearances(self : &BoardMatrix, last_move : (usize, usize), answer : (usize, usize)) -> Vec<(BoardMatrix, (usize, usize), (usize, usize))> {
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
  

  fn get_blacks_whites(self : &Self) -> (BoardMatrix/*black*/, BoardMatrix/*white */) {
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
  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
  'a', 'b', 'c', 'd', 'e', 'f', 'g',
  'h', 'i', 'j', 'k', 'l', 'm', 'n',
  'o', 'p', 'q', 'r', 's', 't',
  'u', 'v', 'w', 'x', 'y', 'z',
  'A', 'B', 'C', 'D', 'E', 'F', 'G',
  'H', 'I', 'J', 'K', 'L', 'M', 'N',
  'O', 'P', 'Q', 'R', 'S', 'T',
  'U', 'V', 'W', 'X', 'Y', 'Z',
  '.', '-', ':', '+', '=', '^', '!', 
  '*', '?', '<', '>', '(', ')', '[', ']', '{',
  '}', '@', '#'
];

static BASE81_REVERSE_TABLE : &'static [u8] = &[
  9, 9, 9, 9,  9, 9, 9, 9,  9, 9, 9, 9,  9, 9, 9, 9,  9, 9, 9, 9,
  9, 9, 9, 9,  9, 9, 9, 9,  9, 9, 9, 9,  9, 9, 9, 9,  9, 9, 9, 9,
  9, 9, 9, 9,  9, 9, 9, 9,  9, 9, 9, 9,  9, 9, 9, 9,  9, 9, 9, 9,
  9, 9, 9, 9,  9, 9, 9, 9,  9, 9, 9, 9,  9, 9, 9, 9,  9, 9, 9, 9,
  9, 9, 9, 9,  9, 9, 9, 9,  9, 9, 9, 9,  9, 9, 9, 9,  9, 9, 9, 9,
  9, 9, 9, 9,  9, 9, 9, 9,  9, 9, 9, 9,  9, 9, 9, 9,  9, 9, 9, 9,
  9, 9, 9, 9,  9, 9, 9, 9,  9, 9, 9, 9,  2, 1, 1, 2,  9, 9, 9, 9,
  2, 2, 2, 2,  9, 9, 9, 9,  9, 9, 9, 9,  9, 9, 9, 9,  9, 9, 9, 9,
  2, 2, 0, 1,  2, 2, 0, 2,  2, 1, 2, 0,  2, 1, 0, 2,  9, 9, 9, 9,
  2, 1, 0, 0,  2, 0, 2, 2,  9, 9, 9, 9,  0, 0, 0, 0,  0, 0, 0, 1,
  0, 0, 0, 2,  0, 0, 1, 0,  0, 0, 1, 1,  0, 0, 1, 2,  0, 0, 2, 0,
  0, 0, 2, 1,  0, 0, 2, 2,  0, 1, 0, 0,  2, 1, 0, 1,  9, 9, 9, 9,
  2, 1, 2, 2,  2, 1, 1, 0,  2, 2, 0, 0,  2, 1, 2, 1,  2, 2, 2, 1,
  1, 1, 0, 0,  1, 1, 0, 1,  1, 1, 0, 2,  1, 1, 1, 0,  1, 1, 1, 1,
  1, 1, 1, 2,  1, 1, 2, 0,  1, 1, 2, 1,  1, 1, 2, 2,  1, 2, 0, 0,
  1, 2, 0, 1,  1, 2, 0, 2,  1, 2, 1, 0,  1, 2, 1, 1,  1, 2, 1, 2,
  1, 2, 2, 0,  1, 2, 2, 1,  1, 2, 2, 2,  2, 0, 0, 0,  2, 0, 0, 1,
  2, 0, 0, 2,  2, 0, 1, 0,  2, 0, 1, 1,  2, 0, 1, 2,  2, 0, 2, 0,
  2, 0, 2, 1,  2, 2, 1, 0,  9, 9, 9, 9,  2, 2, 1, 1,  2, 1, 1, 1,
  9, 9, 9, 9,  9, 9, 9, 9,  0, 1, 0, 1,  0, 1, 0, 2,  0, 1, 1, 0,
  0, 1, 1, 1,  0, 1, 1, 2,  0, 1, 2, 0,  0, 1, 2, 1,  0, 1, 2, 2,
  0, 2, 0, 0,  0, 2, 0, 1,  0, 2, 0, 2,  0, 2, 1, 0,  0, 2, 1, 1,
  0, 2, 1, 2,  0, 2, 2, 0,  0, 2, 2, 1,  0, 2, 2, 2,  1, 0, 0, 0,
  1, 0, 0, 1,  1, 0, 0, 2,  1, 0, 1, 0,  1, 0, 1, 1,  1, 0, 1, 2,
  1, 0, 2, 0,  1, 0, 2, 1,  1, 0, 2, 2,  2, 2, 1, 2,  9, 9, 9, 9,  2, 2, 2, 0,
];

#[test]
fn generate_reverse_table() {
  let mut coll : Vec<_> = BASE81_TABLE.iter()
    .enumerate()
    .map(|(index, ch)| {
      let mut trits :[u8; 4] = [0; 4];
      trits[0] = u8::try_from(index / 27).unwrap();
      let num = index % 27;
      trits[1] = u8::try_from(num / 9).unwrap();
      let num = num % 9;
      trits[2] = u8::try_from(num / 3).unwrap();
      let num = num % 3;
      trits[3] = u8::try_from(num).unwrap();

      ( *ch as u8, trits)
    }).collect();

  coll.sort_by( |x, y| y.0.cmp(&x.0) );

  let first = coll.remove(0);
  let mut source_code = format!("{}, {}, {}, {},  ", first.1[0], first.1[1], first.1[2], first.1[3]);  
  let mut count = 0;

  for code in (0..first.0).rev() {
    if !coll.is_empty() && coll[0].0 == code {
      let current = coll.remove(0);
      source_code.insert_str(0, &format!("{}, {}, {}, {},  ", current.1[0], current.1[1], current.1[2], current.1[3]));
    } else {
      source_code.insert_str(0, "9, 9, 9, 9,  ");
    }
    count = count + 1;
    if count % 5 == 0 {
      source_code.insert_str( 0, "\n");
    }
  }

  print!("{}", &source_code);
}




#[test]
fn test_base81(){
  use rand::Rng; // 0.8.0

  let m = BoardMatrix::from_fn( |_, _| rand::thread_rng().gen_range(0..3) );
  let text = m.to_base81_string();
  println!("{}", &text);

  assert_eq!(m, Board::from_base81_string(&text));
}




fn swap(m : &mut BoardMatrix, pos1 : (usize, usize), pos2 : (usize, usize)){
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
pub fn flip_anti_diagonal(m : &mut BoardMatrix) {
  for row in 0..SIZE {
    for col in 0..(SIZE-row) {
      swap(m, (row, col), anti_diagonal_transform((row, col)));
    }
  }
}

fn anti_diagonal_transform( (row, col) : (usize, usize)) -> (usize, usize) {
  return (SIZE-1-col, SIZE-1-row);
}

/*
1,2,3     1,4,7
4,5,6  -> 2,5,8
7,8,9     3,6,9
 */
pub fn flip_main_diagonal(m : &mut BoardMatrix) {
  for row in 0..SIZE {
    for col in (row+1)..SIZE {
      swap(m, (row, col), main_diagonal_transform((row, col)));
    }
  }
}
fn main_diagonal_transform( (row, col) : (usize, usize)) -> (usize, usize) {
  return (col, row);
}

/*
1,2,3     7,8,9
4,5,6  -> 4,5,6
7,8,9     1,2,3
 */
pub fn flip_vertical(m : &mut BoardMatrix) {
  for row in 0..(SIZE/2) {
    for col in 0..SIZE {
      swap(m, (row, col), vertical_transform((row, col)));
    }
  }
}
fn vertical_transform( (row, col) : (usize, usize)) -> (usize, usize) {
  return (SIZE-1-row, col);
}

/*
1,2,3     3,2,1
4,5,6  -> 6,5,2
7,8,9     9,8,2
 */
#[allow(dead_code)]
pub fn flip_horizontal(m : &mut BoardMatrix) {
  for row in 0..SIZE {
    for col in 0..(SIZE/2) {
      swap(m, (row, col), horizontal_transform((row, col)));
    }
  }
}
fn horizontal_transform( (row, col) : (usize, usize)) -> (usize, usize) {
  return (row, SIZE-1-col);
}

/*
1,2,3     9,6,3       7,4,1
4,5,6  -> 8,5,2  ->   8,5,2
7,8,9     7,4,1       9,6,3
 */
#[allow(dead_code)]
pub fn rotate_right(m : &mut BoardMatrix) {
  _ = flip_anti_diagonal(m);
  _ = flip_vertical(m);
}


/*
1,2,3     1,4,7     3,6,9
4,5,6  -> 2,5,8  -> 2,5,8
7,8,9     3,6,9     1,2,3
 */
#[allow(dead_code)]
pub fn rotate_left(m : &mut BoardMatrix) {
  _ = flip_main_diagonal(m);
  _ = flip_vertical(m);
}





/// Scan blacks and blanks in a direction of position
/// Return an array of vectors, each of them contains the postions of blacks or blanks
///          0          1          2          3          4
///  X --> BLACKS --> BLANKS --> BLACKS --> BLANKS --> BLACKS
fn scan_direction<F>( m: &BoardMatrix, pos:(usize, usize), get_neighbor:F ) -> [Vec<(usize, usize)>; 5]
  where F : Fn( (usize, usize) ) -> Option<(usize, usize)>
{
  let mut vectors: [Vec<(usize, usize)>; 5] = Default::default();

  let mut current = pos;
    let mut expected_state = BLACK;
    let mut index = 0;
    while index < vectors.len() {
      match get_neighbor(current) {
        None => break, // out of board
        Some(p) => {
          let state = m[p];
          if state == expected_state {
            current = p;
            vectors[index].push(current);
          } else {
            index += 1;
            match state {
              WHITE => break, // a white stone
              BLACK => expected_state = BLACK,
              BLANK => expected_state = BLANK,
              _ => unreachable!("Board state must not exceed 2"),
            }
          }
        }
      }
    }// end of while
  vectors
}





fn scan_row<F1, F2>( m: &BoardMatrix, pos:(usize, usize), get_neighbor_of_main_direction:F1, get_neighbor_of_opposite_direction:F2, state : &mut RowState)
    where F1 : Fn( (usize, usize) ) -> Option<(usize, usize)>,
          F2 : Fn( (usize, usize) ) -> Option<(usize, usize)>
  {

    let main_diretion: [Vec<(usize, usize)>; 5] = scan_direction( m, pos, get_neighbor_of_main_direction);
    let opposite_diretion: [Vec<(usize, usize)>; 5] = scan_direction( m, pos, get_neighbor_of_opposite_direction);


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
            let mut cloned = m.clone();
            cloned[pos] = BLACK; // suppose the specified position is black
            if !cloned.is_forbidden(main_diretion[1][0]) {
              four = true;
            }
          }

          // ┼●●●●┼
          //      ^
          if opposite_diretion[1].len() > 0 {
            let mut cloned = m.clone();
            cloned[pos] = BLACK; // suppose the specified position is black
            if !cloned.is_forbidden(opposite_diretion[1][0]) {
              four = true;
            }
          }

          if four {
            state.four_count += 1;
          }
        },

        //////////////////// three continuous blacks
        3 => {

          // ┼●┼●●●┼
          //   ^
          if main_diretion[1].len() == 1 && main_diretion[2].len() == 1 {
            let mut cloned = m.clone();
            cloned[pos] = BLACK; // suppose the specified position is black
            if !cloned.is_forbidden(main_diretion[1][0]) {
              state.four_count += 1;
            }
          }

          // ┼●●●┼●┼
          //     ^
          if opposite_diretion[1].len() == 1 && opposite_diretion[2].len() == 1 {
            let mut cloned = m.clone();
            cloned[pos] = BLACK; // suppose the specified position is black
            if !cloned.is_forbidden(opposite_diretion[1][0]) {
              state.four_count += 1;
            }
          }

          let mut live_three = false;
          // ┼┼●●●┼
          if ( 
                main_diretion[1].len() > 2 
                || 
                (
                  main_diretion[1].len() == 2 
                  && 
                  main_diretion[2].is_empty() 
                ) 
             ) 
             &&
             ( 
                opposite_diretion[1].len() > 1 
                || 
                ( 
                  opposite_diretion[1].len() == 1 
                  && 
                  opposite_diretion[2].is_empty() 
                ) 
             ) 
          {
            let mut cloned = m.clone();
            cloned[pos] = BLACK; // suppose the specified position is black

            // ┼┼●●●┼
            // ^^   ^
            if !cloned.is_forbidden(main_diretion[1][0]) &&
               !cloned.is_forbidden(main_diretion[1][1]) &&
               !cloned.is_forbidden(opposite_diretion[1][0])  {
                live_three = true;
            }
          }

          // ┼●●●┼┼
          if ( 
            opposite_diretion[1].len() > 2 
            || 
            (
              opposite_diretion[1].len() == 2 
              && 
              opposite_diretion[2].is_empty() 
            ) 
          ) 
          &&
          ( 
            main_diretion[1].len() > 1 
            || 
            ( 
              main_diretion[1].len() == 1 
              && 
              main_diretion[2].is_empty() 
            ) 
          ) 
          {
            let mut cloned = m.clone();
            cloned[pos] = BLACK; // suppose the specified position is black

            // ┼┼●●●┼
            //  ^   ^
            if !cloned.is_forbidden(opposite_diretion[1][0]) &&
               !cloned.is_forbidden(opposite_diretion[1][1]) &&
               !cloned.is_forbidden(main_diretion[1][0])  {
                live_three = true;
              
            }
          }

          if live_three {
            state.three_count += 1;
          }
        },

        //////////////////// two continuous blacks
        2 => {

          // ●●┼●●
          if main_diretion[1].len() == 1 && main_diretion[2].len() == 2 {
            let mut cloned = m.clone();
            cloned[pos] = BLACK; // suppose the specified position is black
            if !cloned.is_forbidden(main_diretion[1][0]) {
              state.four_count += 1;
            }
          }

          // ●●┼●●
          if opposite_diretion[1].len() == 1 && opposite_diretion[2].len() == 2 {
            let mut cloned = m.clone();
            cloned[pos] = BLACK; // suppose the specified position is black
            if !cloned.is_forbidden(opposite_diretion[1][0]) {
              state.four_count += 1;
            }
          }

          // ┼●┼●●┼
          if main_diretion[1].len() == 1 &&
             main_diretion[2].len() == 1 &&
             (
                main_diretion[3].len() > 1 // at least two blanks
                ||
                ( main_diretion[3].len() == 1 && main_diretion[4].is_empty() ) // or one blank without further blacks
             )
             &&
             (
                opposite_diretion[1].len() > 1 // at least two blanks
                ||
                ( opposite_diretion[1].len() == 1 && opposite_diretion[2].is_empty() ) // or one blank without further blacks
             )
          {
            let mut cloned = m.clone();
            cloned[pos] = BLACK; // suppose the specified position is black

            // ┼●┼●●┼
            //   ^
            if !cloned.is_forbidden(main_diretion[1][0]) {
              state.three_count += 1;
            }
          }

          // ┼●●┼●┼
          if opposite_diretion[1].len() == 1 &&
             opposite_diretion[2].len() == 1 &&
             (
                opposite_diretion[3].len() > 1 // at least two blanks
                ||
                ( opposite_diretion[3].len() == 1 && opposite_diretion[4].is_empty() ) // or one blank without further blacks
             )
             &&
             (
                main_diretion[1].len() > 1 // at least two blanks
                ||
                ( main_diretion[1].len() == 1 && main_diretion[2].is_empty() ) // or one blank without further blacks
             )
          {
            let mut cloned = m.clone();
            cloned[pos] = BLACK; // suppose the specified position is black

            // ┼●●┼●┼
            //    ^
            if !cloned.is_forbidden(opposite_diretion[1][0]) {
              state.three_count += 1;
            }
          }
        },

        //////////////////// single black
        1 => {

          // ●●●┼●
          if main_diretion[1].len() == 1 && main_diretion[2].len() == 3 {
            let mut cloned = m.clone();
            cloned[pos] = BLACK; // suppose the specified position is black

            // ●●●┼●
            //    ^
            if !cloned.is_forbidden(main_diretion[1][0]) {
              state.four_count += 1;
            }
          }

          // ●┼●●●
          if opposite_diretion[1].len() == 1 && opposite_diretion[2].len() == 3 {
            let mut cloned = m.clone();
            cloned[pos] = BLACK; // suppose the specified position is black

            // ●●●┼●
            //    ^
            if !cloned.is_forbidden(opposite_diretion[1][0]) {
              state.four_count += 1;
            }
          }


          // ┼●●┼●┼
          if main_diretion[1].len() == 1 &&
             main_diretion[2].len() == 2 &&
             (
              main_diretion[3].len() > 1 ||
              ( main_diretion[3].len() == 1 && main_diretion[4].is_empty() )
             ) &&
             (
              opposite_diretion[1].len() > 1 ||
              ( opposite_diretion[1].len() == 1 && opposite_diretion[2].is_empty() )
             )
          {
            let mut cloned = m.clone();
            cloned[pos] = BLACK; // suppose the specified position is black

            // ┼●●┼●┼
            //    ^
            if !cloned.is_forbidden(main_diretion[1][0]) {
              state.three_count += 1;
            }
          }


          // ┼●┼●●┼
          if opposite_diretion[1].len() == 1 &&
             opposite_diretion[2].len() == 2 &&
             (
              opposite_diretion[3].len() > 1 ||
              ( opposite_diretion[3].len() == 1 && opposite_diretion[4].is_empty() )
             ) &&
             (
              main_diretion[1].len() > 1 ||
              ( main_diretion[1].len() == 1 && main_diretion[2].is_empty() )
             )
          {
            let mut cloned = m.clone();
            cloned[pos] = BLACK; // suppose the specified position is black

            // ┼●●┼●┼
            //    ^
            if !cloned.is_forbidden(opposite_diretion[1][0]) {
              state.three_count += 1;
            }
          }

        },
        0 => unreachable!("Number of black stone will never be zero!"),

        //////////////////// move than five
        _ => {
          state.over_five = true;
        },
    }

  }