# Prerequisites (for trainning)


```
pip3 install onnxruntime
pip3 install numpy
pip3 install tensorflow
pip3 install tensorflow_probability
pip3 install git+ssh://git@github.com/onnx/tensorflow-onnx.git
```

Mac Arm64
```
pip3 install tensorflow-macos
pip3 install tensorflow-metal
```

# Train

```
cargo run --features train -- train
```



# How to build TFLite


https://github.com/tensorflow/tflite-support/issues/755#issuecomment-1060998000ls

```
bazel build -c opt tensorflow_lite_support/tools/pip_package:build_pip_package

./bazel-bin/tensorflow_lite_support/tools/pip_package/build_pip_package --dst wheels --nightly_flag

```