extern crate wee_alloc;

// Use `wee_alloc` as the global allocator.
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

mod game;
mod mcts;
mod utils;

use game::{RenjuBoard, SquareMatrix, StateTensor};
use mcts::MonteCarloTree;
use serde::Deserialize;
use utils::set_panic_hook;

use std::collections::HashMap;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    fn predict(state_tensor: JsValue) -> JsValue;
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

#[derive(Debug, Deserialize)]
pub struct Prediction {
    pub probability_matrix: SquareMatrix<f32>,
    pub score: f32, // color in hex code
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
            let return_value = predict(serde_wasm_bindgen::to_value(&state_tensor).unwrap());
            if return_value.is_object() && !return_value.is_null() && !return_value.is_undefined() {
                let prediction: Prediction = match serde_wasm_bindgen::from_value(return_value) {
                    Ok(p) => p,
                    Err(e) => {
                        console_log(&format!("{:?}", e));
                        panic!();
                    }
                };
                console_log(&format!("{:?}", &prediction));
                return (prediction.probability_matrix, prediction.score);
            }
            console_log("predict() does not return valid value");
            unreachable!()
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
