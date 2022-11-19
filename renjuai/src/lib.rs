extern crate wee_alloc;

// Use `wee_alloc` as the global allocator.
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

mod utils;

use std::collections::HashMap;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet(s: &str) -> Car {
    //alert("Hello, renjuai!");
    return Car {
        number: 1,
        color: 2,
    };
}

#[wasm_bindgen]
pub struct Car {
    pub number: usize,
    pub color: usize, // color in hex code
}

#[wasm_bindgen]
pub fn parse(input: &str) -> JsValue {
    let mut ret: HashMap<String, Vec<i8>> = HashMap::new();
    ret.insert(input.to_string(), vec![2, 3, 5]);

    JsValue::from_serde(&ret).unwrap()
}
