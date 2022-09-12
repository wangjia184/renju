# 对弈生成器，数据用于初始化策略网络

https://zhuanlan.zhihu.com/p/59567014

https://gist.github.com/lnshi/eb3dea05d99daba5c932bbc786cc3701

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