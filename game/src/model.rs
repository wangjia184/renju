use bytes::Bytes;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyTuple};
use std::fs::OpenOptions;
use std::io::prelude::*;

use crate::game::*;

pub struct PolicyValueModel {
    module: Py<PyModule>,
}

pub trait RenjuModel {
    fn train(
        self: &Self,
        state_tensors: &[StateTensor],
        prob_matrixes: &[SquaredMatrix],
        scores: &[f32],
        lr: f32,
    ) -> PyResult<(f32, f32)>;

    fn predict(
        self: &Self,
        state_tensors: &[StateTensor],
        use_log_prob: bool, // true to return log probability
    ) -> PyResult<(SquaredMatrix<f32>, f32)>;

    fn export(self: &Self) -> PyResult<Bytes>;

    fn import(self: &Self, buffer: Bytes) -> PyResult<()>;

    fn random_choose_with_dirichlet_noice(self: &Self, probs: &[f32]) -> PyResult<usize>;
}

//
impl PolicyValueModel {
    pub fn new(filename: &str) -> PyResult<Self> {
        let mut source_code = String::new();
        {
            let mut file = OpenOptions::new()
                .read(true)
                .write(false)
                .truncate(false)
                .create(false)
                .open(filename)?;

            file.read_to_string(&mut source_code)?;
        }

        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| -> PyResult<Self> {
            let module: Py<PyModule> =
                PyModule::from_code(py, source_code.as_str(), filename, "")?.into();

            Ok(Self { module: module })
        })
    }
}

impl RenjuModel for PolicyValueModel {
    fn train(
        self: &Self,
        state_tensors: &[StateTensor],
        prob_matrixes: &[SquaredMatrix],
        scores: &[f32],
        lr: f32,
    ) -> PyResult<(f32, f32)> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| -> PyResult<(f32, f32)> {
            py.check_signals()?;
            let train_fn: Py<PyAny> = self.module.as_ref(py).getattr("train")?.into();

            let state_batch: &PyList = PyList::new(py, state_tensors);
            let prob_batch: &PyList = PyList::new(py, prob_matrixes);
            let score_batch: &PyList = PyList::new(py, scores);

            let args = (state_batch, prob_batch, score_batch, lr);
            let result = train_fn.call1(py, args)?;
            let tuple = <PyTuple as PyTryFrom>::try_from(result.as_ref(py))?;
            let loss = tuple.get_item(0)?;
            let entropy = tuple.get_item(1)?;
            Ok((loss.extract()?, entropy.extract()?))
        })
    }

    fn predict(
        self: &Self,
        state_tensors: &[StateTensor],
        use_log_prob: bool,
    ) -> PyResult<(SquaredMatrix<f32>, f32)> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            py.check_signals()?;
            let predict_fn: Py<PyAny> = self.module.as_ref(py).getattr("predict")?.into();

            let state_batch: &PyList = PyList::new(py, state_tensors);

            let args = PyTuple::new(py, &[state_batch]);
            let result = predict_fn.call1(py, args)?;
            let tuple = <PyTuple as PyTryFrom>::try_from(result.as_ref(py))?;

            let log_prob_list = <PyList as PyTryFrom>::try_from(tuple.get_item(0)?)?;
            assert_eq!(log_prob_list.len(), BOARD_SIZE * BOARD_SIZE);
            let mut log_prob_matrix: SquaredMatrix<f32> = SquaredMatrix::default();
            for (index, log_prob) in log_prob_list.iter().enumerate() {
                let mut probability = log_prob.extract::<f32>()?;
                if use_log_prob {
                    probability = probability.ln();
                }
                log_prob_matrix[index / BOARD_SIZE][index % BOARD_SIZE] = probability;
            }

            let score: f32 = tuple.get_item(1)?.extract()?;

            Ok((log_prob_matrix, score))
        })
    }

    fn export(self: &Self) -> PyResult<Bytes> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let export_fn: Py<PyAny> = self.module.as_ref(py).getattr("export_parameters")?.into();

            let result: Py<PyAny> = export_fn.call0(py)?.into();
            let encoded_str = <PyBytes as PyTryFrom>::try_from(result.as_ref(py))?;

            let buffer = Bytes::copy_from_slice(encoded_str.as_bytes());
            Ok(buffer)
        })
    }

    fn import(self: &Self, buffer: Bytes) -> PyResult<()> {
        Python::with_gil(|py| {
            let import_fn: Py<PyAny> = self.module.as_ref(py).getattr("import_parameters")?.into();

            let bytes = PyBytes::new(py, &buffer);
            let args = PyTuple::new(py, &[bytes]);
            import_fn.call1(py, args)?;
            Ok(())
        })
    }

    fn random_choose_with_dirichlet_noice(self: &Self, probs: &[f32]) -> PyResult<usize> {
        Python::with_gil(|py| {
            let func: Py<PyAny> = self
                .module
                .as_ref(py)
                .getattr("random_choose_with_dirichlet_noice")?
                .into();

            let state_batch: &PyList = PyList::new(py, probs);

            let args = PyTuple::new(py, &[state_batch]);
            func.call1(py, args)?;
            Ok(0)
        })
    }
}
