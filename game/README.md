

```
sudo mkdir -p /usr/local/lib/libtensorflow-cpu-darwin-arm64-2.9.0

sudo tar -C /usr/local/lib/libtensorflow-cpu-darwin-arm64-2.9.0 -xzf bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz
```

cd /usr/local/lib/libtensorflow-cpu-darwin-arm64-2.9.0/lib


export DYLD_LIBRARY_PATH=/opt/homebrew/opt/tensorflow/lib
export DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH":~/opt/anaconda3/envs/ml/lib/python3.9/site-packages/tensorflow-plugins



Mac M1

```
pip3 install tensorflow-macos
pip3 install tensorflow-metal
pip3 install tensorflow_probability

```

https://github.com/tensorflow/tflite-support/issues/755#issuecomment-1060998000ls

```
bazel build -c opt tensorflow_lite_support/tools/pip_package:build_pip_package

./bazel-bin/tensorflow_lite_support/tools/pip_package/build_pip_package --dst wheels --nightly_flag
```
