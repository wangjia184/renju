https://joshvarty.github.io/AlphaZero/
https://jonathan-hui.medium.com/alphago-zero-a-game-changer-14ef6e45eba5
https://jonathan-hui.medium.com/alphago-how-it-works-technically-26ddcc085319

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
pip3 install git+ssh://git@github.com/onnx/tensorflow-onnx.git
```

python3 -m tf2onnx.convert --saved-model saved_model/20221021 --opset 17 --output model.onnx
