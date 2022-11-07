# cargo install tauri-cli

& "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

# Unable to find libclang: "couldn't find any valid shared libraries matching: ['clang.dll', 'libclang.dll'], set the `LIBCLANG_PATH` environment variable to a path where one of these files can be found 
$env:LIBCLANG_PATH="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\bin"

$env:TFLITEC_PREBUILT_PATH_X86_64_PC_WINDOWS_MSVC=[System.Environment]::CurrentDirectory + "\tensorflowlite_c.dll"
$env:TFLITEC_PREBUILT_PATH_X86_64_PC_WINDOWS_GNU=[System.Environment]::CurrentDirectory + "\tensorflowlite_c.dll"

# Error occurred while downloading from https://raw.githubusercontent.com/tensorflow/tensorflow/v2.9.1/tensorflow/lite/c/c_api.h: Error { description: "Failure when receiving data from the peer", code: 56, extra: None }
$env:HTTPS_PROXY="http://127.0.0.1:7890"

$env:RUSTFLAGS="-C target-feature=+crt-static"
#cargo build
# cargo tauri build s