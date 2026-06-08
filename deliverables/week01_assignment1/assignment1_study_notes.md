# Assignment 1 学习笔记

## 本周学习主题

本周围绕 CS231n Assignment 1 展开，重点理解基础图像分类器和全连接神经网络的训练流程。

## 主要内容

* kNN：理解距离度量、交叉验证和超参数选择。
* Softmax：理解线性分类器、交叉熵损失、正则化和梯度计算。
* TwoLayerNet：理解隐藏层、ReLU、前向传播、反向传播和参数更新。
* FullyConnectedNet：理解多层网络的模块化实现方式。
* 优化器：比较 SGD、Momentum、RMSProp 和 Adam 的更新规则。
* 向量化实现：减少 Python 循环，提升计算效率。

## 当前总结

Assignment 1 的核心价值在于把损失函数、梯度、反向传播和优化器串联起来。后续 Task1 NumPy 神经网络框架会继续沿用这些基础概念。

## 实验观察

当前 full-data 实验使用 49000 张训练图像、1000 张验证图像和 10000 张测试图像。kNN 的 hard search 与 elbow 选择均落在 `k=1`，测试准确率为 0.3513。Softmax 搜索的最佳测试准确率为 0.3523，SVM 搜索达到 0.3760。TwoLayerNet 对学习率非常敏感，`1e-2` 发散为 NaN，`1e-4` 达到测试准确率 0.4802。

进一步探索后，优化器消融中 SGD 取得测试准确率 0.4855，优于同一学习率下的 Momentum 和 Adam；FullyConnectedNet 的归一化/Dropout 消融在修正固定 dropout seed 后重跑，无归一化且无 Dropout 的配置取得测试准确率 0.4637，Dropout 单独使用取得 0.4383。当前观察强化了两个判断：调参需要优先看验证集和 loss 曲线，模块效果需要放在相同训练配置下比较。
