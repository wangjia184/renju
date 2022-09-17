use copy_to_output::copy_to_output;
use std::env;

fn main() {
    // Re-runs script if any files in res are changed
    println!("cargo:rerun-if-changed=renju_15x15_model/*");
    copy_to_output("renju_15x15_model", &env::var("PROFILE").unwrap()).expect("Could not copy");

    tauri_build::build()
}
