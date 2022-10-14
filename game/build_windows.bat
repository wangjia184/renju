CALL "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
SET LIBCLANG_PATH=C:\Program Files\Microsoft^ Visual^ Studio\2022\Enterprise\VC\Tools\Llvm\x64\bin

SET TFLITEC_PREBUILT_PATH_X86_64_PC_WINDOWS_MSVC=%CD%\tensorflowlite_c.dll
SET TFLITEC_PREBUILT_PATH_X86_64_PC_WINDOWS_GNU=%CD%\tensorflowlite_c.dll

SET RUSTFLAGS=-C target-feature=+crt-static
cargo build --release

