# Assignment 1 方法说明

本文件用于补充 Assignment 1 原报告中“方法介绍较少、偏实现”的不足。这里不重复代码细节，而是从方法目标、基本数学形式、forward/backward 计算和实现方式差异四个角度概括本次涉及的主要模型。

## 1. kNN

kNN 解决的是最基础的监督分类问题：给定已标注训练样本，对新的测试样本寻找距离最近的若干训练样本，并用邻居标签投票得到预测类别。它不显式学习参数，训练阶段主要是保存训练集。

常用距离为 $L_2$ 距离：

$$
d(x, x_i) = \lVert x - x_i \rVert_2
$$

预测时选取距离最小的 $k$ 个样本：

$$
\hat{y} = \operatorname{majority_vote}\left({y_i \mid x_i \in \mathcal{N}_k(x)}\right)
$$

kNN 没有神经网络意义上的 backward，因为它没有可训练参数。核心 forward 过程就是计算测试样本到训练样本的距离矩阵，并基于距离排序完成投票。

naive 实现通常使用双重循环，逐个测试样本、逐个训练样本计算距离；vectorized 实现使用矩阵乘法展开：

$$
\lVert x - y \rVert_2^2=
\lVert x \rVert_2^2+\lVert y \rVert_2^2-2x^\top y
$$

向量化版本避免 Python 层循环，速度更快，但需要更注意矩阵维度和内存占用。

## 2. Softmax 线性分类器

Softmax 解决多分类问题。它先用线性函数把输入映射到每个类别的 score，再通过 Softmax 转换为类别概率。

基本形式：

$$
s = XW + b
$$

$$
p_j = \frac{\exp(s_j)}{\sum_k \exp(s_k)}
$$

损失函数使用交叉熵，加上 $L_2$ 正则化：

$$
L = -\log p_y + \mathrm{reg} \cdot \sum W^2
$$

forward 的核心是计算 score、概率和损失。backward 的关键结论是对 score 的梯度：

$$
ds = p
$$

$$
ds_y \mathrel{-}= 1
$$

$$
dW = \frac{X^\top ds}{N} + 2 \cdot \mathrm{reg} \cdot W
$$

naive 实现按样本循环，逐个计算概率、损失和梯度；vectorized 实现一次性计算整个 batch 的 score 矩阵和概率矩阵，再用矩阵乘法得到梯度。两者数学等价，但 vectorized 实现更高效，也更适合后续神经网络训练。

## 3. TwoLayerNet

TwoLayerNet 解决的问题是：在单个线性分类器表达能力不足时，引入一个隐藏层学习非线性特征。其结构为：

$$
\mathrm{input}
\rightarrow
\mathrm{affine}
\rightarrow
\mathrm{ReLU}
\rightarrow
\mathrm{affine}
\rightarrow
\mathrm{Softmax}
$$

forward 形式：

$$
h = \operatorname{ReLU}(XW_1 + b_1)
$$

$$
s = hW_2 + b_2
$$

$$
L = \operatorname{softmax_loss}(s, y) + \operatorname{regularization}
$$

backward 按链式法则从 Softmax 损失开始反向传播：

$$
dW_2 = h^\top ds + \mathrm{reg} \cdot W_2
$$

$$
dh = ds W_2^\top
$$

$$
dz_1 = dh \odot \mathbf{1}(z_1 > 0)
$$

$$
dW_1 = X^\top dz_1 + \mathrm{reg} \cdot W_1
$$

其中 ReLU 的 backward 只允许正激活位置传递梯度，负激活位置梯度为 $0$。TwoLayerNet 的关键点是正确缓存 forward 中的中间变量，并按相反顺序计算梯度。

## 4. FullyConnectedNet

FullyConnectedNet 是 TwoLayerNet 的推广，用多个隐藏层堆叠 affine、ReLU、可选 normalization 和 dropout。它用于学习更复杂的非线性表示。

第 $l$ 层的基本 forward 可写为：

$$
z_l = h_{l-1}W_l + b_l
$$

$$
h_l = \operatorname{ReLU}(z_l)
$$

最后一层输出类别 score：

$$
s = h_{L-1}W_L + b_L
$$

backward 仍然基于链式法则逐层反传：

$$
dh_{l-1} = dz_l W_l^\top
$$

$$
dW_l = h_{l-1}^\top dz_l + \mathrm{reg} \cdot W_l
$$

$$
db_l = \sum dz_l
$$

如果使用 batch normalization、layer normalization 或 dropout，则 forward 需要额外保存均值、方差、mask 等 cache，backward 也要把梯度正确传回这些模块。

FullyConnectedNet 的 naive 思路是按层手写固定网络结构；更通用的实现方式是用循环根据层数动态组织参数名，例如 $W_1, b_1, W_2, b_2, \ldots$。这种实现更接近 vectorized 和模块化思想：每一层处理一个 batch 的矩阵计算，层与层之间通过 cache 连接，避免为不同深度重复写大量相似代码。

## 5. naive 与 vectorized 的总体区别

naive 实现强调直观性，通常使用显式循环，便于理解每个样本或每个类别的计算过程。它适合验证公式，但在 NumPy 中效率较低。

vectorized 实现把样本维度和类别维度组织成矩阵，用广播、矩阵乘法和批量求和完成同样计算。它的优点是速度快、代码更接近真实训练流程；难点是需要严格检查矩阵形状、广播方向、正则项系数和 batch 平均方式。

本次 Assignment 1 的实现重点不是只让代码跑通，而是通过 naive 到 vectorized 的对照，理解损失函数、梯度传播和矩阵化计算之间的关系。
