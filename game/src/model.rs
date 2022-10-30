/*
 * (C) Copyright 2022 Jerry.Wang (https://github.com/wangjia184).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
use bytes::Bytes;
use num_cpus;
#[cfg(feature="train")]
use pyo3::prelude::*;
#[cfg(feature="train")]
use pyo3::types::{PyBytes, PyList, PyTuple};
#[cfg(feature="train")]
use std::fs::OpenOptions;
#[cfg(feature="train")]
use std::io::prelude::*;

use tflitec::interpreter::{Interpreter, Options};
use tflitec::tensor;

use crate::game::*;

#[cfg(feature="train")]
pub struct PolicyValueModel {
    module: Py<PyModule>,
}

#[cfg(feature="train")]
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

pub struct TfLiteModel {
    interpreter: Interpreter,
}

impl TfLiteModel {
    pub fn load(tflite_model_path: &str) -> Result<Self, tflitec::Error> {
        // Create interpreter options
        let mut options = Options::default();
        options.thread_count = (num_cpus::get() as i32 - 2).max(1);
        options.is_xnnpack_enabled = true;
        println!("is_xnnpack_enabled={}", options.is_xnnpack_enabled);

        let interpreter = Interpreter::with_model_path(tflite_model_path, Some(options))?;

        // Resize input
        let input_shape = tensor::Shape::new(vec![1, 4, BOARD_SIZE, BOARD_SIZE]);
        interpreter.resize_input(0, input_shape)?;
        interpreter.allocate_tensors()?;

        Ok(TfLiteModel {
            interpreter: interpreter,
        })
    }

    pub fn predict_one(
        self: &Self,
        state_tensor: StateTensor,
    ) -> Result<(SquareMatrix<f32>, f32), tflitec::Error> {
        let input_tensor = self.interpreter.input(0)?;
        assert_eq!(input_tensor.data_type(), tensor::DataType::Float32);

        assert!(input_tensor.set_data(&state_tensor).is_ok());

        // Invoke interpreter
        assert!(self.interpreter.invoke().is_ok());

        // Get output tensor
        let prob_matrix_tensor = self.interpreter.output(0)?;
        let score_tensor = self.interpreter.output(1)?;

        assert_eq!(
            prob_matrix_tensor.shape().dimensions(),
            &vec![1, BOARD_SIZE as usize * BOARD_SIZE as usize]
        );

        assert_eq!(score_tensor.shape().dimensions(), &vec![1, 1]);

        let prob_matrix_data = prob_matrix_tensor.data::<f32>().to_vec();
        let score_data = score_tensor.data::<f32>().to_vec();

        let mut prob_matrix: SquareMatrix = SquareMatrix::default();

        for row in 0..BOARD_SIZE {
            prob_matrix[row] = prob_matrix_data[row * BOARD_SIZE..(row + 1) * BOARD_SIZE]
                .try_into()
                .unwrap();
        }

        let score = score_data[0];

        Ok((prob_matrix, score))
    }
}
