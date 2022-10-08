use copy_to_output::copy_to_output;
use std::env;

fn main() {
    // Re-runs script if any files in res are changed
    println!("cargo:rerun-if-changed=saved_model/*");
    println!("cargo:rerun-if-changed=model.py");
    println!("cargo:rerun-if-changed=latest.weights");
    copy_to_output("saved_model", &env::var("PROFILE").unwrap()).expect("Could not copy");
    copy_to_output("model.py", &env::var("PROFILE").unwrap()).expect("Could not copy");
    copy_to_output("latest.weights", &env::var("PROFILE").unwrap()).expect("Could not copy");
    tauri_build::build()
}
