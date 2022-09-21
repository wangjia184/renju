use crate::model::OnDeviceModel;
use crate::player::{AiPlayer, Match};
use std::{cell::RefCell, rc::Rc};

pub fn run(old_model_path: &str, new_model_path: &str) {
    let old_model = Rc::new(RefCell::new(
        OnDeviceModel::load(old_model_path)
            .expect(&format!("Unable to load model {}", old_model_path)),
    ));

    let new_model = Rc::new(RefCell::new(
        OnDeviceModel::load(old_model_path)
            .expect(&format!("Unable to load model {}", new_model_path)),
    ));

    let mut old_player = AiPlayer::new(old_model, 1000);
    let mut new_player = AiPlayer::new(new_model, 1000);

    println!("Old model play black; new model play white");
    let mut one_match = Match::new(&mut new_player, &mut old_player);
    let state = one_match.play_to_end();
    println!("Result = {:?}", state);
}
