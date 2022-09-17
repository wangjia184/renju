use bytemuck::cast_slice;
use bytes::Bytes;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyTuple};
use std::fs::OpenOptions;
use std::io::prelude::*;

use tensorflow::Graph;
use tensorflow::Operation;
use tensorflow::SavedModelBundle;
use tensorflow::SessionOptions;
use tensorflow_sys as tf;

use std::sync::Once;

use tensorflow::SessionRunArgs;
use tensorflow::Tensor;

use tensorflow::Status;

use crate::game::*;

pub struct PolicyValueModel {
    module: Py<PyModule>,
}

pub struct OnDeviceModel {
    graph: Graph,
    bundle: SavedModelBundle,
    predict_input: Operation,
    predict_output: Operation,
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

    pub fn get_best() -> Self {
        let renju_model =
            PolicyValueModel::new("/Users/jerry/projects/renju/renju.git/game/model.py")
                .expect("Unable to load model");

        if let Ok(mut file) = OpenOptions::new()
            .read(true)
            .write(false)
            .truncate(false)
            .create(false)
            .open("best.ckpt")
        {
            let mut buffer = Vec::<u8>::with_capacity(100000);
            if let Ok(size) = file.read_to_end(&mut buffer) {
                if size > 0 {
                    if let Err(e) = renju_model.import(Bytes::from(buffer)) {
                        println!("Unable to import parameters. {}", e);
                    } else {
                        println!("Imported parameters");
                    }
                }
            }
        }

        renju_model
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

static START: Once = Once::new();

impl OnDeviceModel {
    pub fn load(export_dir: &str) -> Result<Self, Status> {
        START.call_once(|| {
            if let Err(e) = tf::library::load() {
                println!("Unable to load libtensorflow. Please download from https://www.tensorflow.org/install/lang_c");
                panic!("Unable to initialize tensorflow. {}", e);
            }
        });

        let mut graph = Graph::new();
        let bundle =
            SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, export_dir)?;

        // predict
        let predict_signature = bundle
            .meta_graph_def()
            .get_signature("predict")
            .expect("Unable to find concreate function `predict`");
        let inputs_info = predict_signature
            .get_input("state_batch")
            .expect("Unable to find `state_batch` in concreate function `predict`");
        let act_output_info = predict_signature
            .get_output("output_0")
            .expect("Unable to find `output_0` in concreate function `predict`");
        let predict_input_op = graph
            .operation_by_name_required(&inputs_info.name().name)
            .expect("Unable to find `state_batch` op in concreate function `predict`");
        let predict_output_op = graph
            .operation_by_name_required(&act_output_info.name().name)
            .expect("Unable to find `output_0` in concreate function `predict`");

        let model = Self {
            graph: graph,
            bundle: bundle,
            predict_input: predict_input_op,
            predict_output: predict_output_op,
        };

        Ok(model)
    }
}

impl RenjuModel for OnDeviceModel {
    fn train(
        self: &Self,
        state_tensors: &[StateTensor],
        prob_matrixes: &[SquaredMatrix],
        scores: &[f32],
        lr: f32,
    ) -> PyResult<(f32, f32)> {
        unimplemented!()
    }

    fn predict(
        self: &Self,
        state_tensors: &[StateTensor],
        use_log_prob: bool, // true to return log probability
    ) -> PyResult<(SquaredMatrix<f32>, f32)> {
        assert!(!state_tensors.is_empty());
        assert_eq!(use_log_prob, false);
        let state_batch = Tensor::<f32>::new(&[
            state_tensors.len() as u64,
            state_tensors[0].len() as u64,
            state_tensors[0][0].len() as u64,
            state_tensors[0][0][0].len() as u64,
        ])
        .with_values(cast_slice(&state_tensors))
        .expect("Unable to create state batch tensor");

        let mut prediction = SessionRunArgs::new();
        prediction.add_feed(&self.predict_input, 0, &state_batch);
        prediction.add_target(&self.predict_output);
        let log_action_token = prediction.request_fetch(&self.predict_output, 0);
        let value_token = prediction.request_fetch(&self.predict_output, 1);
        self.bundle
            .session
            .run(&mut prediction)
            .expect("Unable to run prediction");

        // Check our results.
        let action_tensor: Tensor<f32> = prediction
            .fetch(log_action_token)
            .expect("Unable to retrieve action result");
        let value_tensor: Tensor<f32> = prediction
            .fetch(value_token)
            .expect("Unable to retrieve value");

        assert_eq!(
            action_tensor.dims(),
            [BOARD_SIZE as u64 * BOARD_SIZE as u64]
        );

        let mut matrix = SquaredMatrix::default();
        action_tensor.iter().enumerate().for_each(|(index, x)| {
            matrix[index / BOARD_SIZE][index % BOARD_SIZE] = *x;
        });

        Ok((matrix, value_tensor[0]))
    }

    fn export(self: &Self) -> PyResult<Bytes> {
        unimplemented!()
    }

    fn import(self: &Self, _: Bytes) -> PyResult<()> {
        unimplemented!()
    }

    fn random_choose_with_dirichlet_noice(self: &Self, _: &[f32]) -> PyResult<usize> {
        unimplemented!()
    }
}
