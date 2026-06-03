# Assignment 2 消融实验记录

当前消融实验使用 CIFAR-10 完整训练设置：`train=49000`、`val=1000`、`test=10000`、`epochs=10`、`batch_size=100`。suite 配置维护在 `assignment2_exploration_suites.json`。

## FullyConnectedNet 归一化与 Dropout 消融

| 项目 | 内容 |
| --- | --- |
| 被替换的模块 | BatchNorm、LayerNorm、Dropout |
| 对照组 | 无 normalization、无 Dropout 的同结构全连接网络 |
| 实验目的 | 判断归一化和 Dropout 在当前浅层 FC 配置下是否改善稳定性或泛化 |
| 结果位置 | `../results/ablation/fc_normalization_dropout_ablation_20260601_210225/` |

| 配置 | Train Acc | Val Acc | Test Acc | 结论 |
| --- | ---: | ---: | ---: | --- |
| 无归一化 / 无 Dropout | 0.4429 | 0.4420 | 0.4394 | 测试准确率最高 |
| BatchNorm | 0.4138 | 0.4430 | 0.4118 | 验证准确率略高，测试不占优 |
| LayerNorm | 0.4212 | 0.4380 | 0.4144 | 接近 BatchNorm，但未超过对照组 |
| Dropout keep=0.8 | 0.2583 | 0.2870 | 0.2577 | 明显欠拟合 |
| BatchNorm + Dropout keep=0.8 | 0.1726 | 0.2700 | 0.1728 | 欠拟合最明显 |

## ThreeLayerConvNet L2 正则消融

| 项目 | 内容 |
| --- | --- |
| 被改变的因素 | L2 正则强度：`reg=0`、`reg=1e-3`、`reg=1e-2` |
| 对照组 | 当前 baseline 使用 `reg=1e-3` |
| 实验目的 | 判断 L2 正则是否降低 train-val gap 并改善泛化 |
| 结果位置 | `../results/ablation/conv_regularization_ablation_20260601_211157/` |

| 配置 | Train Acc | Val Acc | Test Acc | Train-Val Gap | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `reg=0` | 0.7702 | 0.6140 | 0.6091 | 0.1562 | 训练准确率最高 |
| `reg=1e-3` | 0.7572 | 0.6140 | 0.6006 | 0.1432 | baseline 配置 |
| `reg=1e-2` | 0.7012 | 0.6100 | 0.6157 | 0.0912 | gap 最小，测试略高 |

## 小结

FC 消融说明当前全连接网络更容易受到 Dropout 带来的容量下降影响。CNN 正则消融说明强 L2 可以压低 train-val gap，但验证准确率不一定同步提升。两组结果都提示：消融实验需要同时看训练准确率、验证准确率、测试准确率和 gap，而不是只比较单个最高测试值。
