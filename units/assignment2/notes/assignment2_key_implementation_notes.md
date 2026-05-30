# Assignment 2 关键实现笔记

## Batch Normalization 与 Layer Normalization

Batch Normalization 在训练时按 batch 维度统计每个特征的均值和方差，并维护 running mean / running variance 供测试阶段使用。反向传播保留了普通链式求导版本和简化版本，统一脚本中对两者的 `dx`、`dgamma`、`dbeta` 做了对比。

Layer Normalization 不依赖 batch 统计，而是在每个样本内部跨特征归一化，因此训练和测试阶段行为一致，也不需要 running statistics。实现上与 BatchNorm 的主要差异是归一化维度从 `axis=0` 变为 `axis=1`。

## Dropout

当前实现使用 inverted dropout。训练时随机保留神经元并除以 keep ratio，使输出期望保持不变；测试时直接返回输入。这样测试阶段不需要额外缩放。

## 卷积、池化与快速路径

naive 卷积和池化用于明确展示局部感受野、padding、stride 和梯度累加逻辑。三层 CNN 训练时优先调用 `fast_layers.py` 中的快速路径。

本机没有编译 Cython 扩展时，`fast_layers.py` 使用纯 NumPy im2col/col2im fallback。这样既保留课程中的 fast layer 接口，也避免公开归档依赖本地编译产物。

## Spatial BatchNorm 与 GroupNorm

Spatial BatchNorm 将 `(N, C, H, W)` 转换为 `(N*H*W, C)` 后复用普通 BatchNorm，使每个通道在所有样本和空间位置上统计均值与方差。

GroupNorm 将通道分为 `G` 组，在每个样本的每个组内统计均值和方差。它不依赖 batch 大小，因此在小 batch 场景下比 BatchNorm 更稳定。

## 全连接网络与 CNN

`FullyConnectedNet` 的隐藏层按 `{affine - norm - relu - dropout}` 组织，最后一层使用 affine 输出分类 score。反向传播按相反顺序读取 cache，权重加入 `0.5 * reg * sum(W^2)`，梯度中加入 `reg * W`。

`ThreeLayerConvNet` 使用 `conv - relu - 2x2 max pool - affine - relu - affine - softmax`。卷积层 padding 和 stride 保持输入空间尺寸，池化后进入全连接层。

## PyTorch RNN/LSTM Captioning

RNN/LSTM 部分使用 PyTorch 张量和 autograd，因此只实现 forward。训练 loss 的路径是 image features 投影为初始 hidden state，caption 输入做 word embedding，再经过 RNN/LSTM 得到每个时间步 hidden state，最后用 temporal affine 和 masked temporal softmax 计算 loss。

采样阶段从 `<START>` token 开始逐步生成，每个时间步用上一步预测词作为下一步输入，并取 vocabulary score 最大的词作为输出。
