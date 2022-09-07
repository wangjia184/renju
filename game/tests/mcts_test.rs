use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use bytemuck::cast_slice;
use ndarray::prelude::*;
use ndarray::{Array, OwnedRepr};
use renju_game::game::*;
use renju_game::mcts::*;
use renju_game::model::*;
use tensorflow::Status;

pub struct MockedRenjuModel {
    map: RefCell<Option<HashMap<(usize, usize), f32>>>,
    score: f32,
}

impl Default for MockedRenjuModel {
    fn default() -> Self {
        MockedRenjuModel {
            map: RefCell::new(None),
            score: 0f32,
        }
    }
}

impl MockedRenjuModel {
    fn generate_prob_matrix(self: &Self) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>> {
        let mut prob_matrix = [[1e-10f32.ln(); BOARD_SIZE]; BOARD_SIZE];

        let option = self.map.borrow_mut().take();
        assert!(option.is_some());
        option.unwrap().iter().for_each(|(pos, prob)| {
            prob_matrix[pos.0][pos.1] = (*prob + 1e-10f32).ln();
        });

        let dim = Dim([1, 1, 15 * 15]);
        let data: &[f32] = cast_slice(&prob_matrix);
        let v = Vec::from(data);
        let x = Array::from_shape_vec(dim, v).unwrap();
        x
    }
}
impl RenjuModel for MockedRenjuModel {
    fn predict(
        self: &Self,
        state_tensors: &[StateTensor],
    ) -> Result<(ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>>, f32), Status> {
        assert!(!state_tensors.is_empty());

        Ok((self.generate_prob_matrix(), self.score))
    }

    fn train(
        self: &Self,
        _: &[StateTensor],
        _: &[SquaredMatrix],
        _: &[f32],
        _: f32,
    ) -> Result<(f32 /*loss*/, f32 /*entropy*/), Status> {
        unimplemented!()
    }
}

#[test]
fn test_1() {
    let model = Rc::new(RefCell::new(MockedRenjuModel::default()));
    let mut board = RenjuBoard::default();

    let mut tree = MonteCarloTree::new(5f32, 1, model.clone());
    let choices = vec![(7, 7), (8, 8)];
    let mut map = HashMap::new();
    map.insert((7, 7), 1f32);
    map.insert((8, 8), 0f32);
    *model.borrow_mut().map.borrow_mut() = Some(map);
    model.borrow_mut().score = 0.2f32;

    tree.rollout(board.clone(), &choices);

    let root_q = tree.get_root().borrow().get_q();
    assert_eq!(root_q, -0.2);

    // white move
    let mut map = HashMap::new();
    map.insert((6, 6), 0f32);
    map.insert((6, 7), 0.5f32);
    map.insert((6, 8), 0f32);
    map.insert((7, 6), 0f32);
    map.insert((7, 8), 0f32);
    map.insert((8, 6), 0f32);
    map.insert((8, 7), 0.5f32);
    map.insert((8, 8), 0f32);
    *model.borrow_mut().map.borrow_mut() = Some(map);
    model.borrow_mut().score = -0.2f32;

    tree.rollout(board.clone(), &choices);

    let child = tree.get_root().borrow().get_child((7, 7));
    assert!(child.is_some());
    let child = child.unwrap();
    assert_eq!(Some((7, 7)), child.borrow().get_action());
    let q = child.borrow().get_q();
    assert_eq!(q, 0.2f32);

    //tree.rolloutboard.clone(), &choices);
}
