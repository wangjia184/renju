
use std::{cmp::Ordering, collections::HashMap};

use nalgebra::{ SMatrix};
use num_traits::identities::Zero;

// Statically sized and statically allocated 2x3 matrix using 32-bit floats.
const SIZE: usize = 15;
pub type BoardMatrix = SMatrix<u8, SIZE, SIZE>;

pub trait Board {
  fn print(self: &Self);
  
  fn mediocritize(self : &mut Self) ->  String;
  fn from_base81_string(text : &str) -> Self;
  fn to_base81_string(self: &Self) -> String;
  fn generate_opening_patterns() -> HashMap<String, Self> where Self : Sized;
  fn for_each_piece<F : FnMut(usize, usize, u8)>(self : &Self, cb : F);
  fn is_over(self : &Self) -> bool;
}



impl Board for BoardMatrix {

  fn is_over(self : &Self) -> bool {

    let mut states = [[[[0u8; 4]; 2]; SIZE]; SIZE];

    for row in 0..SIZE {
      for col in 0..SIZE {
        let index = match self[(row, col)] {
          1 => 0,
          2 => 1,
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
                1 => 2,
                2 => 1,
                _ => panic!("incorrect value"),
            };
            // accumulated
            if let Some(more) = vector.pop() {
              cb( more.0, more.1, expected_value);
              expected_value = match expected_value {
                  1 => 2,
                  2 => 1,
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



  fn mediocritize(self : &mut BoardMatrix) -> String {
    let mut min_matrix : Option<&BoardMatrix> = None;
    let mut min_id = self.to_base81_string();

    let mut m1 = self.clone();
    flip_main_diagonal(&mut m1);
    let id = m1.to_base81_string();
    if id.cmp(&min_id) == Ordering::Less {
      min_id = id;
      min_matrix = Some(&m1);
    }

    let mut m2 = m1.clone();
    flip_vertical(&mut m2);
    let id = m2.to_base81_string();
    if id.cmp(&min_id) == Ordering::Less {
      min_id = id;
      min_matrix = Some(&m2);
    }

    let mut m3 = self.clone();
    flip_anti_diagonal(&mut m3);
    let id = m3.to_base81_string();
    if id.cmp(&min_id) == Ordering::Less {
      min_id = id;
      min_matrix = Some(&m3);
    }

    let mut m4 = m3.clone();
    flip_vertical(&mut m4);
    let id = m4.to_base81_string();
    if id.cmp(&min_id) == Ordering::Less {
      min_id = id;
      min_matrix = Some(&m4);
    }

    let mut m5 = self.clone();
    flip_horizontal(&mut m5);
    let id = m5.to_base81_string();
    if id.cmp(&min_id) == Ordering::Less {
      min_id = id;
      min_matrix = Some(&m5);
    }

    let mut m6 = self.clone();
    flip_vertical(&mut m6);
    let id = m6.to_base81_string();
    if id.cmp(&min_id) == Ordering::Less {
      min_id = id;
      min_matrix = Some(&m6);
    }

    let mut m7 = m6.clone();
    flip_horizontal(&mut m7);
    let id = m7.to_base81_string();
    if id.cmp(&min_id) == Ordering::Less {
      min_id = id;
      min_matrix = Some(&m7);
    }

    if let Some(m) = min_matrix {
        self.copy_from(m);
    }

    min_id
    
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



#[test]
pub fn test()
{
  let mut m1 = BoardMatrix::zero();
  m1[(0,0)] = 1;
  m1[(0,1)] = 2;
  let id1 = m1.mediocritize();
  //print_board(&m1);

 
  let mut m3 = BoardMatrix::zero();
  m3[(0,14)] = 1;
  m3[(0,13)] = 2;
  let id3 = m1.mediocritize();
  //print_board(&m3);
  assert_eq!(m1 , m3);
  assert_eq!(id1, id3);
  
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








#[test]
fn test_pattern()
{
  BoardMatrix::generate_opening_patterns().values().enumerate().for_each( |(idx, x)| {
    println!("{}", idx + 1);
    x.print();
  });
  
}



#[test]
fn test_for_each_piece()
{
  let mut m = BoardMatrix::zeros();
  m[(0,0)] = 1;
  m[(0,1)] = 1;
  m[(0,2)] = 1;
  m[(0,3)] = 2;
  m[(0,4)] = 2;
  m[(0,5)] = 1;
  m[(0,6)] = 2;
  m[(0,7)] = 2;
  m[(0,8)] = 1;
  
  m.for_each_piece( | row, col, value | {
    println!("{}:{} = {}", row, col, value);
  })
}


#[test]
fn test_is_over()
{
  let mut m = BoardMatrix::zeros();
  m[(6,6)] = 1;
  m[(7,7)] = 1;
  m[(8,8)] = 1;
  m[(9,9)] = 1;
  m[(10,10)] = 2;
  m[(11,11)] = 1;
  m[(12,12)] = 1;
  m[(13,13)] = 1;
  m[(14,14)] = 1;
  
  assert_eq!( m.is_over(), false);
  m[(9,9)] = 2;
  m[(10,10)] = 1;
  assert_eq!( m.is_over(), true);

  let mut m = BoardMatrix::zeros();
  m[(0,0)] = 2;
  m[(0,1)] = 2;
  m[(0,2)] = 2;
  m[(0,4)] = 2;
  assert_eq!( m.is_over(), false);
  m[(0,3)] = 2;
  assert_eq!( m.is_over(), true);


  let mut m = BoardMatrix::zeros();
  m[(0,14)] = 1;
  m[(0,13)] = 1;
  m[(0,12)] = 1;
  m[(0,11)] = 1;
  assert_eq!( m.is_over(), false);
  m[(0,10)] = 1;
  assert_eq!( m.is_over(), true);


  let mut m = BoardMatrix::zeros();
  m[(0,14)] = 1;
  m[(1,14)] = 1;
  m[(2,14)] = 1;
  m[(3,14)] = 1;
  m[(4,14)] = 2;
  assert_eq!( m.is_over(), false);
  m[(4,14)] = 1;
  assert_eq!( m.is_over(), true);

  let m = BoardMatrix::from_base81_string("0000000000000i0003r00R200p904QS08?017N01J2020001000000000");
  m.print();
  assert_eq!( m.is_over(), false);
}