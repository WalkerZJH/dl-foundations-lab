# Task1 方法补充

## Linear 与 ReLU

Linear 层完成仿射变换 $Y=XW+b$。给定上游梯度 $G$，反向传播为 $dX=GW^T$、$dW=X^TG$、$db=\sum_i G_i$。ReLU 使用 $Y=\max(0,X)$ 引入非线性，梯度为 $dX=G\odot\mathbf{1}(X>0)$。

## Softmax Cross Entropy

对 logits $z$ 先减去行最大值以避免指数溢出，再计算 $p_k=\exp(z_k)/\sum_j\exp(z_j)$。单样本损失为 $-\log p_y$，batch 平均后的 logits 梯度为 $(p-\mathrm{onehot}(y))/N$。

## Convolution

二维卷积 forward 对输入局部窗口与卷积核做内积。naive 实现显式遍历样本、卷积核和空间位置，结构直观；vectorized 实现通过 im2col 将所有窗口展开为矩阵，再用一次矩阵乘法计算。backward 中 $dW$ 是上游梯度与窗口列矩阵的乘积，$dX$ 通过 col2im 将重叠窗口梯度累加回原空间。

## Batch Normalization

训练阶段先计算 batch 均值 $\mu$ 与方差 $\sigma^2$，再得到 $\hat{x}=(x-\mu)/\sqrt{\sigma^2+\epsilon}$ 和 $y=\gamma\hat{x}+\beta$。反向传播同时考虑均值、方差对每个样本的依赖。评估阶段使用训练中累计的 running mean/variance，不再读取当前 batch 统计量。

## Dropout

inverted Dropout 以 keep ratio $q$ 采样 mask：$m\sim\mathrm{Bernoulli}(q)/q$，训练阶段 $y=x\odot m$，反向传播 $dx=dy\odot m$；评估阶段直接使用 $y=x$。本实现的固定 seed 初始化独立随机序列，但不会在每次 forward 重置随机状态。

## 优化器

SGD 使用 $\theta\leftarrow\theta-\eta g$。Momentum 维护速度 $v\leftarrow\mu v-\eta g$ 并更新 $\theta\leftarrow\theta+v$。Adam 分别估计一阶矩与二阶矩，做偏差修正后按 $\hat{m}/(\sqrt{\hat{v}}+\epsilon)$ 更新参数。

## 数值梯度检查

对标量函数 $f$ 使用中心差分：

$$
\frac{\partial f}{\partial x_i}\approx\frac{f(x_i+h)-f(x_i-h)}{2h}.
$$

数值梯度与解析 backward 的相对误差用于定位符号、轴、广播和重叠窗口累加错误。Task1 对 Linear、BatchNorm 和 Conv2D 分别执行该检查，并额外比较 naive/vectorized 卷积的 forward、$dX$、$dW$、$db$。
