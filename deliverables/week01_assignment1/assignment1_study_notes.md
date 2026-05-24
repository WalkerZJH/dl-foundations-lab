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

当前小规模实验中，kNN 已改为使用完整 2000/500/500 实验子集，最佳验证准确率为 0.2800，测试准确率为 0.2320。Softmax baseline 的测试准确率最高，TwoLayerNet 接近 Softmax。FullyConnectedNet 的训练准确率明显高于验证和测试准确率，说明更深模型在小数据设置下更容易过拟合，后续需要继续调整正则化、dropout、学习率和训练轮数。
