use bytemuck::cast_slice;
use ndarray::prelude::*;
use ndarray::{Array, IxDynImpl, OwnedRepr};
use renju_game::game::*;
use renju_game::mcts::*;
use renju_game::model::*;
use tensorflow::Status;

pub struct MockedRenjuModel {}

impl RenjuModel for MockedRenjuModel {
    fn predict(
        self: &Self,
        state_tensors: &[StateTensor],
    ) -> Result<(ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>>, f32), Status> {
        assert!(!state_tensors.is_empty());

        let mut temperature = Array3::<f64>::zeros((3, 4, 5));

        let mut prob_matrix = SquaredMatrix::<f32>::default();
        prob_matrix[7][7] = 1f32;

        let dim = Dim([1, 1, 15 * 15]);
        let data: &[f32] = cast_slice(&prob_matrix);
        let v = Vec::from(data);
        // We can safely unwrap this because we know that `data` will have the
        // correct number of elements to conform to `dim`.
        let x = Array::from_shape_vec(dim, v).unwrap();

        Ok((x, 0f32))
    }

    fn train(
        self: &Self,
        state_tensors: &[StateTensor],
        prob_matrixes: &[SquaredMatrix],
        scores: &[f32],
        lr: f32,
    ) -> Result<(f32 /*loss*/, f32 /*entropy*/), Status> {
        unimplemented!()
    }
}

#[test]
fn test_1() {}
