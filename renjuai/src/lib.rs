extern crate wee_alloc;

// Use `wee_alloc` as the global allocator.
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

mod game;
mod mcts;
mod utils;

use game::{RenjuBoard, SquareMatrix, StateTensor, BOARD_SIZE};
use mcts::MonteCarloTree;
use serde::Deserialize;
use js_sys::Array;

use std::{collections::HashMap, fs::read_to_string, default};

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    fn predict(state_tensor: JsValue) -> Prediction;
    fn console_log(text: &str);
}

#[wasm_bindgen]
pub fn greet(s: &str) -> Car {
    let board = game::RenjuBoard::default();
    //alert("Hello, renjuai!");
    return Car {
        number: board.get_last_move().unwrap().0,
        color: 2,
    };
}

#[wasm_bindgen]
pub struct Car {
    pub number: usize,
    pub color: usize, // color in hex code
}

#[wasm_bindgen]
pub struct Prediction {
    prob_matrix: SquareMatrix<f32>,
    score: f32, // color in hex code
}

#[wasm_bindgen]
impl Prediction {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            prob_matrix : SquareMatrix::<f32>::default(),
            score : 0f32,
        }
    }

    #[wasm_bindgen(setter)]
    pub fn set_probabilities(&mut self, probabilities : Array) {
        assert_eq!( probabilities.length() as usize, BOARD_SIZE * BOARD_SIZE);
        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                let index = row * BOARD_SIZE + row;
                self.prob_matrix[row][col] = probabilities.get(index as u32).as_f64().unwrap() as f32;
            }
            
        }
    }

    #[wasm_bindgen(setter)]
    pub fn set_score(&mut self, score : f32) {
        self.score = score;
    }
}



#[wasm_bindgen]
pub async fn test(input: &str) -> Result<JsValue, JsValue> {
    let mut ret: HashMap<String, Vec<i8>> = HashMap::new();
    ret.insert(input.to_string(), vec![2, 3, 5]);

    let board = RenjuBoard::default();
    let tree = MonteCarloTree::new(3f32);
    let choices = vec![(7usize, 7usize)];
    tree.rollout(
        board,
        &choices,
        |state_tensor| -> (SquareMatrix<f32>, f32) {
            let input = serde_wasm_bindgen::to_value(&state_tensor).unwrap();
            let prediction = predict(input);

            return (prediction.prob_matrix, prediction.score);
        },
    )
    .await
    .expect("Error");

    Ok(serde_wasm_bindgen::to_value(&ret).unwrap())
}

#[wasm_bindgen]
pub fn start(human_play_black: bool) {
    //let tree = TREE.lock().unwrap();
}
