# Assignment 1 消融实验记录

本文件记录 Assignment 1 的消融实验设计、结果位置和观察结论。当前正式结果使用 CIFAR-10 完整训练设置：训练集 49000 张、验证集 1000 张、测试集 10000 张。

## 初始化方式消融

| 项目 | 内容 |
| --- | --- |
| 改变的因素 | 权重初始化方式：零初始化、小随机初始化、He 初始化。 |
| 对照组 | 小随机初始化。 |
| 控制变量 | TwoLayerNet 结构、学习率、L2 正则、batch size 和训练轮数。 |
| 结果位置 | `../results/ablation/init_ablation_20260601_174349/` |
| 观察结果 | 零初始化 val acc 0.1050，接近随机水平；He 初始化 val acc 0.2520，但 final loss 偏大。 |

零初始化结果符合对称性破缺的预期。He 初始化在固定 `lr=1e-3` 和 momentum 设置下验证准确率最高，但它需要和学习率一起判断，不能只看单点准确率。

## 优化器消融

| 项目 | 内容 |
| --- | --- |
| 改变的因素 | 参数更新规则：SGD、SGD + Momentum、Adam。 |
| 对照组 | 同一 TwoLayerNet 结构和初始化设置。 |
| 控制变量 | 模型结构、初始化、学习率、L2 正则、batch size 和训练轮数。 |
| 结果位置 | `../results/ablation/optimizer_ablation_20260601_174556/` |
| 观察结果 | SGD 取得 val acc 0.4960、test acc 0.4855，高于 Momentum 和 Adam。 |

Momentum 在当前学习率下表现不稳定；Adam 稳定但测试准确率低于 SGD。该结果说明优化器效果依赖学习率，不应脱离配置单独比较。

## 归一化与 Dropout 消融

| 项目 | 内容 |
| --- | --- |
| 改变的因素 | FullyConnectedNet 中的 BatchNorm、LayerNorm、Dropout。 |
| 对照组 | 不使用 normalization/dropout 的同结构全连接网络。 |
| 控制变量 | 网络层数、隐藏维度、优化器、学习率、正则强度和训练轮数。 |
| 结果位置 | `../results/ablation/normalization_dropout_ablation_20260601_174826/` |
| 观察结果 | 无归一化且无 Dropout 的配置取得 val acc 0.4840、test acc 0.4637；BatchNorm 和 LayerNorm 未超过对照组；Dropout 单独使用明显降低表现。 |

当前全连接网络层数较浅、训练轮数有限，归一化没有带来稳定收益；Dropout 在该容量和数据设置下更像是降低有效容量。
