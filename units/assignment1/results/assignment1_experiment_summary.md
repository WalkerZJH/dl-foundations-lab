# Assignment 1 实验结果摘要

实验数据：CIFAR-10 子集，训练集 2000 张、验证集 500 张、测试集 500 张。kNN 使用同一子集的完整 2000 张训练样本、500 张验证样本和 500 张测试样本。

实验耗时：2.88 秒。

| 模型 | 设置 | 训练准确率 | 验证准确率 | 测试准确率 |
| --- | --- | ---: | ---: | ---: |
| kNN | k=7, train=2000, val=500, test=500 | - | 0.2800 | 0.2320 |
| Softmax | lr=1e-7, reg=2.5e4, iters=500 | 0.3410 | 0.3080 | 0.2960 |
| TwoLayerNet | sgd_momentum, lr=0.001, epochs=3 | 0.2560 | 0.2820 | 0.2760 |
| FullyConnectedNet | adam, lr=0.001, epochs=3 | 0.5700 | 0.2860 | 0.2320 |

## 图片输出

* `figures/assignment1_loss_curves.png`：Softmax、TwoLayerNet、FullyConnectedNet 的训练损失曲线。
* `figures/assignment1_accuracy_comparison_full_knn.svg`：使用完整 kNN 实验子集后的准确率对比图。

## 简要观察

* kNN 已使用完整的 2000/500/500 实验子集，不再为了降低运行时间额外采样。
* kNN 在完整子集上的最佳验证准确率为 0.2800，测试准确率为 0.2320；相比此前采样版本，验证准确率和测试准确率均有提升。
* Softmax 的测试准确率仍最高，为 0.2960。TwoLayerNet 测试准确率为 0.2760，FullyConnectedNet 在当前小数据设置下仍表现出过拟合。
