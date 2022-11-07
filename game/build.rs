use copy_to_output::copy_to_output;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    let target = build_target::target_triple().unwrap().to_lowercase();
    let lib_dir = Path::new(&dir)
        .join("lib")
        .join(&target)
        .display()
        .to_string();

    let paths = fs::read_dir(&lib_dir).expect("Unable to access library folder");

    for path in paths {
        let entry = path.unwrap();
        let entry_type = entry.file_type().unwrap();

        if !entry_type.is_dir() {
            let filepath = entry.path();
            if let Some(os_filename) = filepath.file_name() {
                if let Some(filename) = os_filename.to_str() {
                    match (target.as_str(), filename) {
                        ("x86_64-pc-windows-msvc", "tensorflowlite_c.dll")
                        | ("aarch64-apple-darwin", "libtensorflowlite_c.dylib") => {
                            println!(
                                "cargo:rustc-env=TFLITEC_PREBUILT_PATH_{}={}",
                                target.to_uppercase().replace("-", "_"),
                                filepath.display().to_string()
                            );
                        }
                        _ => (),
                    };

                    if let Some(ext) = filepath.extension() {
                        let need_copy = match ext.to_str() {
                            Some("dylib") => true,
                            Some("so") => true,
                            Some("lib") => true,
                            Some("dll") => true,
                            _ => false,
                        };
                        if need_copy {
                            copy_to_output(
                                &filepath.display().to_string(),
                                &env::var("PROFILE").unwrap(),
                            )
                            .expect("Could not copy file");
                        }
                    }
                }
            }
        }
    }

    println!("cargo:rustc-link-lib=onnxruntime");
    println!("cargo:rustc-link-search=native={}", &lib_dir);
    // Re-runs script if any files in res are changed
    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=include/wrapper.h");
    println!("cargo:rerun-if-changed=model.py");
    //println!("cargo:rerun-if-changed=best.tflite");

    copy_to_output("model.py", &env::var("PROFILE").unwrap()).expect("Could not copy");
    //copy_to_output("best.tflite", &env::var("PROFILE").unwrap()).expect("Could not copy");

    tauri_build::build();

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header(&format!("include/{}/wrapper.h", &target))
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
