use tensorflow::Graph;
use tensorflow::Operation;
use tensorflow::SavedModelBundle;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Tensor;

use tensorflow::Status;

use tensorflow_sys as tf;

use std::sync::Once;

static START: Once = Once::new();

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

pub trait RenjuModel {
    fn predict(self: &Self, state_batch: &Tensor<f32>) -> Result<(Tensor<f32>, f32), Status>;

    fn train(
        self: &Self,
        state_batch: &Tensor<f32>,
        mcts_probs: &Tensor<f32>,
        winner_batch: &Tensor<f32>,
        lr: f32,
    ) -> Result<(f32 /*loss*/, f32 /*entropy*/), Status>;
}

pub struct PolicyValueModel {
    graph: Graph,
    bundle: SavedModelBundle,
    predict_input: Operation,
    predict_output: Operation,
    train_input_state_batch: Operation,
    train_input_mcts_probs: Operation,
    train_input_winner_batch: Operation,
    train_input_lr: Operation,
    train_output: Operation,
    export_input: Operation,
    export_output: Operation,
    restore_input: Operation,
    restore_output: Operation,
    random_choose_with_dirichlet_noice_input: Operation,
    random_choose_with_dirichlet_noice_output: Operation,
}

impl PolicyValueModel {
    pub fn load(export_dir: &str) -> Result<Self, Status> {
        START.call_once(|| {
            tf::library::load().expect("Unable to load libtensorflow");
            /*
            match load_plugable_device("libmetal_plugin.dylib") {
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
        let input_mcts_probs_info = train_signature
            .get_input("mcts_probs")
            .expect("Unable to find `mcts_probs` in concreate function `train`");
        let input_winner_batch_info = train_signature
            .get_input("winner_batch")
            .expect("Unable to find `winner_batch` in concreate function `train`");
        let input_lr_info = train_signature
            .get_input("lr")
            .expect("Unable to find `lr` in concreate function `train`");
        let output_info = train_signature
            .get_output("output_0")
            .expect("Unable to find `loss` in concreate function `train`");
        let train_input_state_batch = graph
            .operation_by_name_required(&input_state_batch_info.name().name)
            .expect("Unable to find `state_batch` op in concreate function `train`");
        let train_input_mcts_probs = graph
            .operation_by_name_required(&input_mcts_probs_info.name().name)
            .expect("Unable to find `mcts_probs` op in concreate function `train`");
        let train_input_winner_batch = graph
            .operation_by_name_required(&input_winner_batch_info.name().name)
            .expect("Unable to find `winner_batch` op in concreate function `train`");
        let train_input_lr = graph
            .operation_by_name_required(&input_lr_info.name().name)
            .expect("Unable to find `lr` op in concreate function `train`");
        let train_output = graph
            .operation_by_name_required(&output_info.name().name)
            .expect("Unable to find `output_0` op in concreate function `train`");

        // save
        let export_signature = bundle
            .meta_graph_def()
            .get_signature("save")
            .expect("Unable to find concreate function `save`");
        let inputs_info = export_signature
            .get_input("checkpoint_path")
            .expect("Unable to find `checkpoint_path` in concreate function `save`");
        let outputs_info = export_signature
            .get_output("output_0")
            .expect("Unable to find `output_0` in concreate function `save`");
        let export_input_op = graph
            .operation_by_name_required(&inputs_info.name().name)
            .expect("Unable to find `checkpoint_path` op in concreate function `save`");
        let export_output_op = graph
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

        Ok(Self {
            graph: graph,
            bundle: bundle,
            predict_input: predict_input_op,
            predict_output: predict_output_op,
            train_input_state_batch: train_input_state_batch,
            train_input_mcts_probs: train_input_mcts_probs,
            train_input_winner_batch: train_input_winner_batch,
            train_input_lr: train_input_lr,
            train_output: train_output,
            export_input: export_input_op,
            export_output: export_output_op,
            restore_input: restore_input_op,
            restore_output: restore_output_op,
            random_choose_with_dirichlet_noice_input: random_choose_with_dirichlet_noice_input_op,
            random_choose_with_dirichlet_noice_output: random_choose_with_dirichlet_noice_output_op,
        })
    }

    pub fn save(self: &Self, filename: &str) -> Result<(), Status> {
        let file_path_tensor: Tensor<String> = Tensor::from(String::from(filename));

        // Save the model.
        let mut step = SessionRunArgs::new();
        step.add_feed(&self.export_input, 0, &file_path_tensor);
        step.add_target(&self.export_output);
        self.bundle.session.run(&mut step)
    }

    pub fn restore(self: &Self, filename: &str) -> Result<(), Status> {
        let file_path_tensor: Tensor<String> = Tensor::from(String::from(filename));

        // Save the model.
        let mut step = SessionRunArgs::new();
        step.add_feed(&self.restore_input, 0, &file_path_tensor);
        step.add_target(&self.restore_output);
        self.bundle.session.run(&mut step)
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
}

impl RenjuModel for PolicyValueModel {
    fn predict(self: &Self, state_batch: &Tensor<f32>) -> Result<(Tensor<f32>, f32), Status> {
        let shape = state_batch.shape();
        assert_eq!(shape.dims(), Some(4));
        assert_eq!(shape[1], Some(4));
        assert_eq!(shape[2], Some(15));
        assert_eq!(shape[3], Some(15));
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

        Ok((log_action, value[0]))
    }

    fn train(
        self: &Self,
        state_batch: &Tensor<f32>,
        mcts_probs: &Tensor<f32>,
        winner_batch: &Tensor<f32>,
        lr: f32,
    ) -> Result<(f32 /*loss*/, f32 /*entropy*/), Status> {
        let lr_tensor = Tensor::<f32>::new(&[1])
            .with_values(&[lr])
            .expect("Unable to create lr tensor");
        let mut train_step = SessionRunArgs::new();
        train_step.add_feed(&self.train_input_state_batch, 0, &state_batch);
        train_step.add_feed(&self.train_input_mcts_probs, 0, &mcts_probs);
        train_step.add_feed(&self.train_input_winner_batch, 0, &winner_batch);
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
