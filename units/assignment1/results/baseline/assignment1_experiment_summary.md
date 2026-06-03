# Assignment 1 Baseline 实验结果摘要

实验数据：CIFAR-10，训练集 49000 张、验证集 1000 张、测试集 10000 张。

实验耗时：90.74 秒。

| 模型 | 设置 | 训练准确率 | 验证准确率 | 测试准确率 |
| --- | --- | ---: | ---: | ---: |
| Softmax | lr=1e-7, reg=2.5e4, iters=1500 | 0.3314 | 0.3460 | 0.3312 |
| TwoLayerNet | sgd_momentum, lr=0.001, epochs=10 | 0.2204 | 0.2440 | 0.2218 |
| FullyConnectedNet | adam, lr=0.001, epochs=10 | 0.4670 | 0.4640 | 0.4360 |

## 图片输出

* `figures/assignment1_loss_curves.png`：训练损失曲线。
* `figures/assignment1_accuracy_comparison_full_data.png`：模型准确率对比图，供 LaTeX 报告引用。
* `figures/assignment1_accuracy_comparison_full_data.svg`：模型准确率对比图。
