#/bin/zsh

export TFLITEC_PREBUILT_PATH_AARCH64_APPLE_DARWIN=$(pwd)/lib/aarch64-apple-darwin/libtensorflowlite_c.dylib
cargo build --release