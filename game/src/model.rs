use bytemuck::cast_slice;
use ndarray::prelude::*;
use ndarray::{Array, OwnedRepr};
use std::fs::OpenOptions;
use std::io::prelude::*;
use tensorflow::Graph;
use tensorflow::Operation;
use tensorflow::SavedModelBundle;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Tensor;

use tensorflow::Status;

use tensorflow_sys as tf;

use std::sync::Once;

use crate::*;

static START: Once = Once::new();
/*
pub fn load_plugable_device(library_filename: &str) -> Result<(), Status> {
    use std::ffi::CString;
    let c_filename = CString::new(library_filename)?;

    let raw_lib = unsafe {
        let raw_status: *mut tf::TF_Status = tf::TF_NewStatus();
        let raw_lib = tf::TF_LoadPluggableDeviceLibrary(c_filename.as_ptr(), raw_status);
        if !raw_status.is_null() {
            tf::TF_DeleteStatus(raw_status)
        }
        raw_lib
    };

    if raw_lib.is_null() {
        Err(Status::new())
    } else {
        Ok(())
    }
}
 */
pub trait RenjuModel {
    fn predict(
        self: &Self,
        state_tensors: &[StateTensor],
    ) -> Result<(ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>>, f32), Status>;

    fn train(
        self: &Self,
        state_tensors: &[StateTensor],
        prob_matrixes: &[SquaredMatrix],
        scores: &[f32],
        lr: f32,
    ) -> Result<(f32 /*loss*/, f32 /*entropy*/), Status>;
}

pub struct PolicyValueModel {
    graph: Graph,
    bundle: SavedModelBundle,
    predict_input: Operation,
    predict_output: Operation,
    train_input_state_batch: Operation,
    train_input_prob_batch: Operation,
    train_input_score_batch: Operation,
    train_input_lr: Operation,
    train_output: Operation,
    save_input: Operation,
    save_output: Operation,
    restore_input: Operation,
    restore_output: Operation,
    random_choose_with_dirichlet_noice_input: Operation,
    random_choose_with_dirichlet_noice_output: Operation,
    export_output: Operation,
    import_input: Operation,
    import_output: Operation,
}

impl PolicyValueModel {
    pub fn load(export_dir: &str) -> Result<Self, Status> {
        START.call_once(|| {
            tf::library::load().expect("Unable to load libtensorflow");
            /*
            match load_plugable_device("libmetal_plugins.dylib") {
                Ok(_) => println!("Loaded libmetal_plugin.dylib successfully."),
                Err(_) => println!("WARNING: Unable to load plugin."),
            }; */
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

        // train_step
        let train_signature = bundle
            .meta_graph_def()
            .get_signature("train")
            .expect("Unable to find concreate function `train`");
        let input_state_batch_info = train_signature
            .get_input("state_batch")
            .expect("Unable to find `state_batch` in concreate function `train`");
        let input_prob_batch_info = train_signature
            .get_input("prob_batch")
            .expect("Unable to find `prob_batch` in concreate function `train`");
        let input_score_batch_info = train_signature
            .get_input("score_batch")
            .expect("Unable to find `score_batch` in concreate function `train`");
        let input_lr_info = train_signature
            .get_input("lr")
            .expect("Unable to find `lr` in concreate function `train`");
        let output_info = train_signature
            .get_output("output_0")
            .expect("Unable to find `loss` in concreate function `train`");
        let train_input_state_batch = graph
            .operation_by_name_required(&input_state_batch_info.name().name)
            .expect("Unable to find `state_batch` op in concreate function `train`");
        let train_input_prob_batch = graph
            .operation_by_name_required(&input_prob_batch_info.name().name)
            .expect("Unable to find `prob_batch` op in concreate function `train`");
        let train_input_score_batch = graph
            .operation_by_name_required(&input_score_batch_info.name().name)
            .expect("Unable to find `score_batch` op in concreate function `train`");
        let train_input_lr = graph
            .operation_by_name_required(&input_lr_info.name().name)
            .expect("Unable to find `lr` op in concreate function `train`");
        let train_output = graph
            .operation_by_name_required(&output_info.name().name)
            .expect("Unable to find `output_0` op in concreate function `train`");

        // save
        let save_signature = bundle
            .meta_graph_def()
            .get_signature("save")
            .expect("Unable to find concreate function `save`");
        let inputs_info = save_signature
            .get_input("checkpoint_path")
            .expect("Unable to find `checkpoint_path` in concreate function `save`");
        let outputs_info = save_signature
            .get_output("output_0")
            .expect("Unable to find `output_0` in concreate function `save`");
        let save_input_op = graph
            .operation_by_name_required(&inputs_info.name().name)
            .expect("Unable to find `checkpoint_path` op in concreate function `save`");
        let save_output_op = graph
            .operation_by_name_required(&outputs_info.name().name)
            .expect("Unable to find `output_0` in concreate function `save`");

        // restore
        let restore_signature = bundle
            .meta_graph_def()
            .get_signature("restore")
            .expect("Unable to find concreate function `restore`");
        let inputs_info = restore_signature
            .get_input("checkpoint_path")
            .expect("Unable to find `checkpoint_path` in concreate function `restore`");
        let outputs_info = restore_signature
            .get_output("output_0")
            .expect("Unable to find `output_0` in concreate function `restore`");
        let restore_input_op = graph
            .operation_by_name_required(&inputs_info.name().name)
            .expect("Unable to find `checkpoint_path` op in concreate function `restore`");
        let restore_output_op = graph
            .operation_by_name_required(&outputs_info.name().name)
            .expect("Unable to find `output_0` in concreate function `restore`");

        // random_choose_with_dirichlet_noice
        let choice_signature = bundle
            .meta_graph_def()
            .get_signature("random_choose_with_dirichlet_noice")
            .expect("Unable to find concreate function `random_choose_with_dirichlet_noice`");
        let inputs_info = choice_signature.get_input("probs").expect(
            "Unable to find `probs` in concreate function `random_choose_with_dirichlet_noice`",
        );
        let outputs_info = choice_signature.get_output("output_0").expect(
            "Unable to find `output_0` in concreate function `random_choose_with_dirichlet_noice`",
        );
        let random_choose_with_dirichlet_noice_input_op = graph
            .operation_by_name_required(&inputs_info.name().name)
            .expect("Unable to find `probs` op in concreate function `random_choose_with_dirichlet_noice`");
        let random_choose_with_dirichlet_noice_output_op = graph
            .operation_by_name_required(&outputs_info.name().name)
            .expect(
            "Unable to find `output_0` in concreate function `random_choose_with_dirichlet_noice`",
        );

        // export
        let export_signature = bundle
            .meta_graph_def()
            .get_signature("export_param")
            .expect("Unable to find concreate function `export_param`");
        let export_output_info = export_signature
            .get_output("output_0")
            .expect("Unable to find `output_0` in concreate function `export_param`");
        let export_output_op = graph
            .operation_by_name_required(&export_output_info.name().name)
            .expect("Unable to find `output_0` in concreate function `export_param`");

        // import
        let import_signature = bundle
            .meta_graph_def()
            .get_signature("import_param")
            .expect("Unable to find concreate function `import_param`");
        let import_inputs_info = import_signature
            .get_input("encoded_str")
            .expect("Unable to find `encoded_str` in concreate function `import_param`");
        let import_output_info = import_signature
            .get_output("output_0")
            .expect("Unable to find `output_0` in concreate function `import_param`");
        let import_input_op = graph
            .operation_by_name_required(&import_inputs_info.name().name)
            .expect("Unable to find `encoded_str` op in concreate function `import_param`");
        let import_output_op = graph
            .operation_by_name_required(&import_output_info.name().name)
            .expect("Unable to find `output_0` in concreate function `import_param`");

        let model = Self {
            graph: graph,
            bundle: bundle,
            predict_input: predict_input_op,
            predict_output: predict_output_op,
            train_input_state_batch: train_input_state_batch,
            train_input_prob_batch,
            train_input_score_batch,
            train_input_lr: train_input_lr,
            train_output: train_output,
            save_input: save_input_op,
            save_output: save_output_op,
            restore_input: restore_input_op,
            restore_output: restore_output_op,
            random_choose_with_dirichlet_noice_input: random_choose_with_dirichlet_noice_input_op,
            random_choose_with_dirichlet_noice_output: random_choose_with_dirichlet_noice_output_op,
            export_output: export_output_op,
            import_input: import_input_op,
            import_output: import_output_op,
        };

        Ok(model)
    }

    pub fn save(self: &Self, filename: &str) -> Result<String, std::io::Error> {
        let mut file = OpenOptions::new()
            .read(false)
            .write(true)
            .truncate(true)
            .create(true)
            .open(filename)?;

        let content = self.export().expect("Unable to export");
        file.write_all(content.as_bytes())?;
        /*
        let file_path_tensor: Tensor<String> = Tensor::from(String::from(filename));

        // Save the model.
        let mut step = SessionRunArgs::new();
        step.add_feed(&self.save_input, 0, &file_path_tensor);
        step.add_target(&self.save_output);
        self.bundle.session.run(&mut step)*/
        Ok(content)
    }

    pub fn restore(self: &Self, filename: &str) -> Result<(), std::io::Error> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(false)
            .truncate(false)
            .create(false)
            .open(filename)?;

        let mut content = String::new();
        file.read_to_string(&mut content)?;

        self.import(&content).expect("Unable to import");

        Ok(())

        // Save the model.
        //let mut step = SessionRunArgs::new();
        //step.add_feed(&self.restore_input, 0, &file_path_tensor);
        //step.add_target(&self.restore_output);
        //.bundle.session.run(&mut step)
    }

    pub fn random_choose_with_dirichlet_noice(
        self: &Self,
        probabilities: &Tensor<f32>,
    ) -> Result<usize, Status> {
        let shape = probabilities.shape();
        assert_eq!(shape.dims(), Some(1));

        let mut step = SessionRunArgs::new();
        step.add_feed(
            &self.random_choose_with_dirichlet_noice_input,
            0,
            &probabilities,
        );
        step.add_target(&self.random_choose_with_dirichlet_noice_output);
        let selected_index_token =
            step.request_fetch(&self.random_choose_with_dirichlet_noice_output, 0);
        self.bundle.session.run(&mut step)?;

        let tensor: Tensor<i64> = step
            .fetch(selected_index_token)
            .expect("Unable to retrieve selected result");
        let index = tensor[0];
        if index >= 0 && index < shape[0].unwrap() {
            return Ok(index as usize);
        }
        unreachable!("Index must be in the range of probabilities")
    }

    pub fn export(self: &Self) -> Result<String, Status> {
        // Save the model.
        let mut step = SessionRunArgs::new();
        step.add_target(&self.export_output);

        let fetch_token = step.request_fetch(&self.export_output, 0);

        self.bundle.session.run(&mut step)?;

        let tensor = step.fetch::<String>(fetch_token)?;
        let text = tensor.to_string();
        Ok(text.trim_matches('"').to_string())
    }

    pub fn import(self: &Self, encoded_str: &str) -> Result<(), Status> {
        let tensor: Tensor<String> = Tensor::from(String::from(encoded_str));

        // Save the model.
        let mut step = SessionRunArgs::new();
        step.add_feed(&self.import_input, 0, &tensor);
        step.add_target(&self.import_output);
        self.bundle.session.run(&mut step)
    }
}

impl RenjuModel for PolicyValueModel {
    fn predict(
        self: &Self,
        state_tensors: &[StateTensor],
    ) -> Result<(ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>>, f32), Status> {
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
        self.bundle.session.run(&mut prediction)?;

        // Check our results.
        let log_action: Tensor<f32> = prediction
            .fetch(log_action_token)
            .expect("Unable to retrieve action result");
        let value: Tensor<f32> = prediction
            .fetch(value_token)
            .expect("Unable to retrieve log result");

        assert_eq!(
            log_action.dims(),
            [
                state_tensors.len() as u64,
                1u64,
                game::BOARD_SIZE as u64 * game::BOARD_SIZE as u64
            ]
        );

        let dim = Dim([state_tensors.len(), 1, game::BOARD_SIZE * game::BOARD_SIZE]);
        let data: Vec<_> = log_action.iter().map(|x| x.clone()).collect();
        // We can safely unwrap this because we know that `data` will have the
        // correct number of elements to conform to `dim`.
        let x = Array::from_shape_vec(dim, data).unwrap();

        //println!("{:?}", log_action);
        //println!("{:?}", value);

        Ok((x, value[0]))
    }

    fn train(
        self: &Self,
        state_tensors: &[StateTensor],
        prob_matrixes: &[SquaredMatrix],
        scores: &[f32],
        lr: f32,
    ) -> Result<(f32 /*loss*/, f32 /*entropy*/), Status> {
        assert!(!state_tensors.is_empty());
        assert_eq!(state_tensors.len(), prob_matrixes.len());
        assert_eq!(state_tensors.len(), scores.len());

        let state_batch = Tensor::<f32>::new(&[
            state_tensors.len() as u64,
            state_tensors[0].len() as u64,
            state_tensors[0][0].len() as u64,
            state_tensors[0][0][0].len() as u64,
        ])
        .with_values(cast_slice(state_tensors))
        .expect("Unable to create state batch tensor");

        let probs_tensor = Tensor::<f32>::new(&[
            prob_matrixes.len() as u64,
            1,
            (prob_matrixes[0].len() * prob_matrixes[0][0].len()) as u64,
        ])
        .with_values(cast_slice(prob_matrixes))
        .expect("Unable to create probability tensor");

        let score_tensor = Tensor::<f32>::new(&[prob_matrixes.len() as u64, 1, 1])
            .with_values(cast_slice(scores))
            .expect("Unable to create score tensor");

        let lr_tensor = Tensor::<f32>::new(&[1])
            .with_values(&[lr])
            .expect("Unable to create lr tensor");

        let mut train_step = SessionRunArgs::new();
        train_step.add_feed(&self.train_input_state_batch, 0, &state_batch);
        train_step.add_feed(&self.train_input_prob_batch, 0, &probs_tensor);
        train_step.add_feed(&self.train_input_score_batch, 0, &score_tensor);
        train_step.add_feed(&self.train_input_lr, 0, &lr_tensor);
        train_step.add_target(&self.train_output);

        let loss_token = train_step.request_fetch(&self.train_output, 0);
        let entropy_token = train_step.request_fetch(&self.train_output, 1);
        self.bundle.session.run(&mut train_step)?;

        // Check our results.
        let loss: Tensor<f32> = train_step
            .fetch(loss_token)
            .expect("Unable to retrieve loss result");
        let entropy: Tensor<f32> = train_step
            .fetch(entropy_token)
            .expect("Unable to retrieve entropy result");

        Ok((loss[0], entropy[0]))
    }
}
