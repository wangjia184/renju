cd tensorflow


export TF_NEED_OPENCL=0
export TF_CUDA_CLANG=0
export TF_NEED_TENSORRT=0
export TF_DOWNLOAD_CLANG=0
export TF_NEED_MPI=0
export TF_NEED_ROCM=0
export TF_NEED_CUDA=0
export TF_OVERRIDE_EIGEN_STRONG_INLINE=1 # Windows only
export CC_OPT_FLAGS=/arch:AVX
export TF_CONFIGURE_IOS=0
export TF_SET_ANDROID_WORKSPACE=0
export TFLITEC_BAZEL_COPTS="-march=native" # for native optimized build
/c/Users/Jerry.Wang/AppData/Local/Programs/Python/Python310/python.exe configure.py

# Optimizes the generated code for machine's CPU
# --define tflite_with_xnnpack_qs8=true flag enables XNNPACK inference for quantized operators using signed quantization schema. 
#        This schema is used by models produced by Model Optimization Toolkit through either post-training integer quantization or quantization-aware training. 
#        Post-training dynamic range quantization is not supported in XNNPACK.


bazel  --output_base=$(pwd)/../out build  --features=static_link_msvcrt -c opt --define tflite_with_xnnpack=true --define tflite_with_xnnpack_qs8=false --define xnnpack_force_float_precision=fp16 --config=opt //tensorflow/lite/c:tensorflowlite_c


#TFLITEC_PREBUILT_PATH_AARCH64_APPLE_DARWIN=/Users/jerry/projects/renju/renju.git/game/libtensorflowlite_c.dylib