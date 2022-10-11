use bytemuck::cast_slice;
use bytes::Bytes;
use num_cpus;
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

use tflitec::interpreter::{Interpreter, Options};
use tflitec::tensor;

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

    pub fn get_latest() -> Self {
        let renju_model = PolicyValueModel::new("model.py").expect("Unable to load model");

        if let Ok(mut file) = OpenOptions::new()
            .read(true)
            .write(false)
            .truncate(false)
            .create(false)
            .open("latest.weights")
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

    pub fn predict_batch(
        self: &Self,
        state_tensors: Vec<StateTensor>,
    ) -> PyResult<Vec<(SquareMatrix<f32>, f32)>> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            py.check_signals()?;
            let predict_fn: Py<PyAny> = self.module.as_ref(py).getattr("predict_batch")?.into();

            let batch_size = state_tensors.len();
            let state_batch: &PyList = PyList::new(py, state_tensors);

            let args = PyTuple::new(py, &[state_batch]);
            let result = predict_fn.call1(py, args)?;
            let tuple = <PyTuple as PyTryFrom>::try_from(result.as_ref(py))?;

            let prob_matrix_list = <PyList as PyTryFrom>::try_from(tuple.get_item(0)?)?;
            let score_list = <PyList as PyTryFrom>::try_from(tuple.get_item(1)?)?;
            assert_eq!(prob_matrix_list.len(), batch_size);
            assert_eq!(score_list.len(), batch_size);

            let mut pairs = Vec::with_capacity(batch_size);
            for index in 0..batch_size {
                let probs = <PyList as PyTryFrom>::try_from(prob_matrix_list.get_item(index)?)?;
                assert_eq!(probs.len(), BOARD_SIZE * BOARD_SIZE);
                let mut prob_matrix: SquareMatrix<f32> = SquareMatrix::default();
                for (idx, prob) in probs.iter().enumerate() {
                    let probability = prob.extract::<f32>()?;
                    prob_matrix[idx / BOARD_SIZE][idx % BOARD_SIZE] = probability;
                }

                let score: f32 = score_list.get_item(index)?.get_item(0)?.extract()?;

                pairs.push((prob_matrix, score));
            }

            Ok(pairs)
        })
    }

    pub fn train(
        self: &Self,
        state_tensors: &[StateTensor],
        prob_matrixes: &[SquareMatrix],
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

    pub fn predict(
        self: &Self,
        state_tensors: &[StateTensor],
        use_log_prob: bool,
    ) -> PyResult<(SquareMatrix<f32>, f32)> {
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
            let mut log_prob_matrix: SquareMatrix<f32> = SquareMatrix::default();
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

    pub fn export(self: &Self) -> PyResult<Bytes> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let export_fn: Py<PyAny> = self.module.as_ref(py).getattr("export_parameters")?.into();

            let result: Py<PyAny> = export_fn.call0(py)?.into();
            let encoded_str = <PyBytes as PyTryFrom>::try_from(result.as_ref(py))?;

            let buffer = Bytes::copy_from_slice(encoded_str.as_bytes());
            Ok(buffer)
        })
    }

    pub fn import(self: &Self, buffer: Bytes) -> PyResult<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let import_fn: Py<PyAny> = self.module.as_ref(py).getattr("import_parameters")?.into();

            let bytes = PyBytes::new(py, &buffer);
            let args = PyTuple::new(py, &[bytes]);
            import_fn.call1(py, args)?;
            Ok(())
        })
    }

    pub fn save_quantized_model(self: &Self, file_path: &str) -> PyResult<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let save_fn: Py<PyAny> = self
                .module
                .as_ref(py)
                .getattr("save_quantized_model")?
                .into();

            let args = PyTuple::new(py, &[file_path]);
            save_fn.call1(py, args)?;
            Ok(())
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

    pub fn predict(
        self: &Self,
        state_tensors: &[StateTensor],
    ) -> PyResult<(SquareMatrix<f32>, f32)> {
        assert!(!state_tensors.is_empty());

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

        let mut matrix = SquareMatrix::default();
        action_tensor.iter().enumerate().for_each(|(index, x)| {
            matrix[index / BOARD_SIZE][index % BOARD_SIZE] = *x;
        });

        Ok((matrix, value_tensor[0]))
    }
}

pub struct TfLiteModel {
    interpreter: Interpreter,
}

impl TfLiteModel {
    pub fn load(tflite_model_path: &str) -> Result<Self, tflitec::Error> {
        // Create interpreter options
        let mut options = Options::default();
        options.thread_count = num_cpus::get() as i32 / 2;
        options.is_xnnpack_enabled = true;
        println!("is_xnnpack_enabled={}", options.is_xnnpack_enabled);

        let interpreter = Interpreter::with_model_path(tflite_model_path, Some(options))?;

        Ok(TfLiteModel {
            interpreter: interpreter,
        })
    }

    pub fn predict_batch(
        self: &Self,
        state_tensors: Vec<StateTensor>,
    ) -> Result<Vec<(SquareMatrix<f32>, f32)>, tflitec::Error> {
        assert!(!state_tensors.is_empty());

        // Resize input
        let input_shape = tensor::Shape::new(vec![state_tensors.len(), 4, BOARD_SIZE, BOARD_SIZE]);
        self.interpreter.resize_input(0, input_shape)?;
        self.interpreter.allocate_tensors()?;

        let input_tensor = self.interpreter.input(0)?;
        assert_eq!(input_tensor.data_type(), tensor::DataType::Float32);

        assert!(input_tensor.set_data(&state_tensors[..]).is_ok());

        // Invoke interpreter
        assert!(self.interpreter.invoke().is_ok());

        // Get output tensor
        let prob_matrix_tensor = self.interpreter.output(0)?;
        let score_tensor = self.interpreter.output(1)?;

        assert_eq!(
            prob_matrix_tensor.shape().dimensions(),
            &vec![
                state_tensors.len(),
                BOARD_SIZE as usize * BOARD_SIZE as usize
            ]
        );

        assert_eq!(
            score_tensor.shape().dimensions(),
            &vec![state_tensors.len(), 1usize]
        );

        let mut vector = Vec::new();

        let prob_matrix_data = prob_matrix_tensor.data::<f32>().to_vec();
        let score_data = score_tensor.data::<f32>().to_vec();

        for batch_index in 0..state_tensors.len() {
            let mut prob_matrix = SquareMatrix::default();

            for index in 0..BOARD_SIZE * BOARD_SIZE {
                let prob = prob_matrix_data[batch_index * BOARD_SIZE * BOARD_SIZE + index];
                prob_matrix[index / BOARD_SIZE][index % BOARD_SIZE] = prob;
            }

            let score = score_data[batch_index];

            vector.push((prob_matrix, score));
        }

        Ok(vector)
    }
}
