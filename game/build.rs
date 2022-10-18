use copy_to_output::copy_to_output;
use std::env;
use std::path::Path;

fn main() {
    let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let dll_path = Path::new(&dir)
        .join("libtensorflowlite_c.dylib")
        .display()
        .to_string();

    // set path for tflitec
    println!(
        "cargo:rustc-env=TFLITEC_PREBUILT_PATH_AARCH64_APPLE_DARWIN={}",
        dll_path
    );

    // Re-runs script if any files in res are changed
    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=include/wrapper.h");
    println!("cargo:rerun-if-changed=model.py");
    println!("cargo:rerun-if-changed=latest.weights");

    copy_to_output("model.py", &env::var("PROFILE").unwrap()).expect("Could not copy");
    copy_to_output("best.tflite", &env::var("PROFILE").unwrap()).expect("Could not copy");
    tauri_build::build()
    

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("include/wrapper.h")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
