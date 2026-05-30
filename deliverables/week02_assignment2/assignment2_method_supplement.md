# Assignment 2 方法补充

本文件补足 Assignment 2 报告中偏理论的部分，后续可按需要整合进 LaTeX 报告。

## Batch Normalization

Batch Normalization 解决深层网络训练中各层输入分布持续变化的问题。对一个 batch 中的特征矩阵 $X \in \mathbb{R}^{N \times D}$，每个特征独立计算：

$$
\mu = \operatorname{mean}(X, \operatorname{axis}=0)
$$

$$
\sigma^2 = \operatorname{mean}\left((X - \mu)^2, \operatorname{axis}=0\right)
$$

$$
\hat{x} = \frac{X - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

$$
\operatorname{out} = \gamma \hat{x} + \beta
$$

反向传播核心是把上游梯度依次传回 $\beta$、$\gamma$、归一化结果、方差、均值和输入。简化形式可写为：

$$
dx
=
\frac{1}{N}
\cdot
\frac{1}{\mathrm{std}}
\left(
N\,d\hat{x}
-
\sum d\hat{x}
-
\hat{x}\,\sum(d\hat{x}\odot\hat{x})
\right)
$$

naive backward 按计算图逐步展开，便于检查推导；vectorized / alternative backward 将同类项合并，计算更短也更适合工程实现。

## Layer Normalization

Layer Normalization 解决 batch size 较小或序列模型中 batch 统计不稳定的问题。它对每个样本内部的特征维度归一化：

$$
\mu_i = \operatorname{mean}(x_i)
$$

$$
\sigma_i^2 = \operatorname{mean}\left((x_i - \mu_i)^2\right)
$$

$$
\hat{x}_i = \frac{x_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}
$$

forward 与 BatchNorm 形式接近，但统计轴从 batch 维度换成特征维度。backward 可复用 BatchNorm 的简化公式，只需把 $N$ 换成每个样本的特征数 $D$，并按 $\operatorname{axis}=1$ 求和。

## Dropout

Dropout 通过随机屏蔽隐藏单元缓解共适应和过拟合。当前实现使用 inverted dropout：

$$
\operatorname{mask} = \frac{\operatorname{Bernoulli}(p)}{p}
$$

$$
\operatorname{out} = x \odot \operatorname{mask}
$$

训练时 forward 应用 $\operatorname{mask}$，backward 为：

$$
dx = dout \odot \operatorname{mask}
$$

测试时直接返回输入。与普通 dropout 相比，inverted dropout 把缩放放在训练阶段，测试阶段无需额外处理。

## Convolution

卷积层解决图像局部模式提取问题。对输入 $x \in \mathbb{R}^{N \times C \times H \times W}$ 和滤波器 $w \in \mathbb{R}^{F \times C \times HH \times WW}$，输出为：

$$
\operatorname{out}_{n,f,i,j}=
\sum
\left(
\operatorname{window}(x_n, i, j) \odot w_f
\right)
+
b_f
$$

backward 中，$db$ 是所有空间位置上游梯度求和；$dw_f$ 累加每个输入窗口乘以上游梯度；$dx$ 将每个滤波器按上游梯度分配回对应输入窗口。

naive 实现显式遍历样本、滤波器和空间位置，便于理解；vectorized 实现通过 im2col 把卷积转换为矩阵乘法，显著减少 Python 循环开销。

## Max Pooling

Max Pooling 通过局部最大值下采样降低空间分辨率并增强局部平移鲁棒性。forward 对每个 pooling window 取最大值：

$$
\operatorname{out}_{n,c,i,j}=
\max\left(
\operatorname{window}(x_{n,c}, i, j)
\right)
$$

backward 只把梯度传给 forward 中取得最大值的位置。naive 实现通过 mask 定位最大值；向量化实现通常借助 reshape 或 im2col 组织窗口。

## Spatial BatchNorm 与 GroupNorm

Spatial BatchNorm 面向卷积特征图，将 $(N, C, H, W)$ 变换为 $(N \cdot H \cdot W, C)$，对每个通道做 BatchNorm，再变回原形状。它解决 CNN 中每个通道激活尺度漂移的问题。

GroupNorm 将通道分为 $G$ 组，在每个样本的组内统计均值和方差：

$$
x_{\operatorname{group}} \in \mathbb{R}^{N \times G \times (C/G) \times H \times W}
$$

GroupNorm 不依赖 batch 维度，因此小 batch 或显存受限时更稳定。它的 backward 与 LayerNorm 类似，只是归一化范围变成每个样本的每个通道组。

## RNN 与 LSTM Captioning

Vanilla RNN 用隐藏状态递推建模序列：

$$
h_t = \tanh(x_t W_x + h_{t-1} W_h + b)
$$

LSTM 使用输入门、遗忘门、输出门和候选记忆缓解长序列梯度消失：

$$
i, f, o = \sigma(a_i), \sigma(a_f), \sigma(a_o)
$$

$$
g = \tanh(a_g)
$$

$$
c_t = f \odot c_{t-1} + i \odot g
$$

$$
h_t = o \odot \tanh(c_t)
$$

Captioning 模型中，图像特征先投影成初始 hidden state，caption token 经 embedding 后输入 RNN/LSTM，每个时间步用 temporal affine 输出词表 score，并用 masked temporal softmax 忽略 $\texttt{<NULL>}$ padding。PyTorch 版本由 autograd 处理 backward，本次实现重点是 forward 计算路径和采样流程。
