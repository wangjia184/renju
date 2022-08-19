use tensorflow::Graph;
use tensorflow::Operation;
use tensorflow::SavedModelBundle;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Tensor;


use tensorflow::Status;
use tensorflow::ops::Save;
use tensorflow_sys as tf;

use std::ffi::{ CString };
use std::sync::{Once};


static START: Once = Once::new();


pub fn load_plugable_device(library_filename: &str) -> Result<(), Status> {
    use std::ffi::{ CString };
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


pub struct PolicyValueModel {
    graph : Graph,
    bundle : SavedModelBundle,
    predict_input : Operation,
    predict_output : Operation,
    train_input_state_batch : Operation,
    train_input_mcts_probs : Operation,
    train_input_winner_batch : Operation,
    train_output : Operation,
    export_input : Operation,
    export_output : Operation,
    restore_input : Operation,
    restore_output : Operation,
}

impl PolicyValueModel {
    pub fn load(export_dir : &str) -> Result<Self, Status> {

        START.call_once(|| {
            tf::library::load().expect("Unable to load libtensorflow");
    
            match load_plugable_device("libmetal_plugin.dylib") {
                Ok(_) => println!("Loaded plugin successfully."),
                Err(_) => println!("WARNING: Unable to load plugin."),
            };
        });

        let mut graph = Graph::new();
        let bundle = SavedModelBundle::load(&SessionOptions::new(),
            &["serve"],
            &mut graph,
            export_dir)?;

        // predict
        let predict_signature = bundle.meta_graph_def().get_signature("predict").expect("Unable to find concreate function `predict`");
        let inputs_info = predict_signature.get_input("inputs").expect("Unable to find `inputs` in concreate function `predict`");
        let act_output_info = predict_signature.get_output("output_0").expect("Unable to find `output_0` in concreate function `predict`");
        let predict_input_op = graph.operation_by_name_required(&inputs_info.name().name).expect("Unable to find `inputs` op in concreate function `predict`");
        let predict_output_op = graph.operation_by_name_required(&act_output_info.name().name).expect("Unable to find `output_0` in concreate function `predict`");

        // train_step
        let train_signature = bundle.meta_graph_def().get_signature("train").expect("Unable to find concreate function `train`");
        let input_state_batch_info = train_signature.get_input("state_batch").expect("Unable to find `state_batch` in concreate function `train`");
        let input_mcts_probs_info = train_signature.get_input("mcts_probs").expect("Unable to find `mcts_probs` in concreate function `train`");
        let input_winner_batch_info = train_signature.get_input("winner_batch").expect("Unable to find `winner_batch` in concreate function `train`");
        let output_info = train_signature.get_output("output_0").expect("Unable to find `loss` in concreate function `train`");
        let train_input_state_batch = graph.operation_by_name_required(&input_state_batch_info.name().name).expect("Unable to find `state_batch` op in concreate function `train`");
        let train_input_mcts_probs = graph.operation_by_name_required(&input_mcts_probs_info.name().name).expect("Unable to find `mcts_probs` op in concreate function `train`");
        let train_input_winner_batch = graph.operation_by_name_required(&input_winner_batch_info.name().name).expect("Unable to find `winner_batch` op in concreate function `train`");
        let train_output = graph.operation_by_name_required(&output_info.name().name).expect("Unable to find `output_0` op in concreate function `train`");

        // export
        let export_signature = bundle.meta_graph_def().get_signature("export").expect("Unable to find concreate function `export`");
        let inputs_info = export_signature.get_input("checkpoint_filename").expect("Unable to find `checkpoint_filename` in concreate function `export`");
        let outputs_info = export_signature.get_output("output_0").expect("Unable to find `output_0` in concreate function `export`");
        let export_input_op = graph.operation_by_name_required(&inputs_info.name().name).expect("Unable to find `checkpoint_filename` op in concreate function `export`");
        let export_output_op = graph.operation_by_name_required(&outputs_info.name().name).expect("Unable to find `output_0` in concreate function `export`");

        // restore
        let restore_signature = bundle.meta_graph_def().get_signature("restore").expect("Unable to find concreate function `restore`");
        let inputs_info = restore_signature.get_input("checkpoint_filename").expect("Unable to find `checkpoint_filename` in concreate function `restore`");
        let outputs_info = restore_signature.get_output("output_0").expect("Unable to find `output_0` in concreate function `restore`");
        let restore_input_op = graph.operation_by_name_required(&inputs_info.name().name).expect("Unable to find `checkpoint_filename` op in concreate function `restore`");
        let restore_output_op = graph.operation_by_name_required(&outputs_info.name().name).expect("Unable to find `output_0` in concreate function `restore`");

        Ok(Self{
            graph : graph,
            bundle : bundle,
            predict_input : predict_input_op,
            predict_output : predict_output_op,
            train_input_state_batch : train_input_state_batch,
            train_input_mcts_probs : train_input_mcts_probs,
            train_input_winner_batch : train_input_winner_batch,
            train_output : train_output,
            export_input : export_input_op,
            export_output : export_output_op,
            restore_input : restore_input_op,
            restore_output : restore_output_op
        })
    }

    pub fn train(self : &Self, state_batch : &Tensor<f32>, mcts_probs : &Tensor<f32>, winner_batch : &Tensor<f32>) -> Result<(f32/*loss*/, f32/*entropy*/), Status> {

        let mut train_step = SessionRunArgs::new();
        train_step.add_feed(&self.train_input_state_batch, 0, &state_batch);
        train_step.add_feed(&self.train_input_mcts_probs, 0, &mcts_probs);
        train_step.add_feed(&self.train_input_winner_batch, 0, &winner_batch);
        train_step.add_target(&self.train_output);

        let loss_token = train_step.request_fetch(&self.train_output, 0);
        let entropy_token = train_step.request_fetch(&self.train_output, 1);
        self.bundle.session.run(&mut train_step)?;


        // Check our results.
        let loss: Tensor<f32> = train_step.fetch(loss_token).expect("Unable to retrieve loss result");
        let entropy: Tensor<f32> = train_step.fetch(entropy_token).expect("Unable to retrieve entropy result");

        Ok( (loss[0], entropy[0]) )
    }

    pub fn predict(self : &Self, state_batch : &Tensor<f32>) -> Result<(Tensor<f32>, f32), Status> {

        let mut prediction = SessionRunArgs::new();
        prediction.add_feed(&self.predict_input, 0, &state_batch);
        prediction.add_target(&self.predict_output);
        let log_action_token = prediction.request_fetch(&self.predict_output, 0);
        let value_token = prediction.request_fetch(&self.predict_output, 1);
        self.bundle.session.run(&mut prediction)?;

        // Check our results.
        let log_action: Tensor<f32> = prediction.fetch(log_action_token).expect("Unable to retrieve action result");
        let value: Tensor<f32> = prediction.fetch(value_token).expect("Unable to retrieve log result");


        Ok((log_action, value[0]))
    }

    pub fn save(self : &Self, filename : &str) -> Result<(), Status> {

        let file_path_tensor: Tensor<String> =
        Tensor::from(String::from(filename));

        // Save the model.
        let mut step = SessionRunArgs::new();
        step.add_feed(&self.export_input, 0, &file_path_tensor);
        step.add_target(&self.export_output);
        self.bundle.session.run(&mut step)
    }


    pub fn restore(self : &Self, filename : &str) -> Result<(), Status> {

        let file_path_tensor: Tensor<String> =
        Tensor::from(String::from(filename));

        // Save the model.
        let mut step = SessionRunArgs::new();
        step.add_feed(&self.restore_input, 0, &file_path_tensor);
        step.add_target(&self.restore_output);
        self.bundle.session.run(&mut step)
    }
}

