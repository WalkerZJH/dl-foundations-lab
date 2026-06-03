# Assignment 2 超参数调优记录

当前超参数调优实验使用 CIFAR-10 完整训练设置：`train=49000`、`val=1000`、`test=10000`、`epochs=10`、`batch_size=100`。suite 配置维护在 `assignment2_exploration_suites.json`。

## ThreeLayerConvNet 学习率搜索

| 项目 | 内容 |
| --- | --- |
| 调整的超参数 | `learning_rate` |
| 对照设置 | 固定 `num_filters=8`、`hidden_dim=100`、`reg=1e-3`、Adam、10 epochs |
| 结果位置 | `../results/hparam_tuning/conv_learning_rate_search_20260601_215042/` |
| 选中配置 | `lr=1e-3`，val acc `0.6140`，test acc `0.6006` |
| 观察 | `lr=5e-4` 的 final loss 更低、test acc 为 `0.6110`，但验证准确率低于 `lr=1e-3`；验证集选择原则下保留 `lr=1e-3`。 |

## ThreeLayerConvNet 容量与正则搜索

| 项目 | 内容 |
| --- | --- |
| 调整的超参数 | `num_filters`、`hidden_dim`、`reg` |
| 对照设置 | 固定 Adam、`learning_rate=1e-3`、10 epochs、batch size 100 |
| 结果位置 | `../results/hparam_tuning/conv_capacity_reg_search_20260601_223059/` |
| 选中配置 | `num_filters=16`、`hidden_dim=100`、`reg=1e-3`，val acc `0.6440`，test acc `0.6350` |
| 观察 | 增加卷积核数量带来更强表示能力；降低到 4 个卷积核会明显降低验证和测试表现。 |

## 结果表

| 实验 | 配置 | Train Acc | Val Acc | Test Acc | 观察 |
| --- | --- | ---: | ---: | ---: | --- |
| Learning rate | `lr=1e-3` | 0.7572 | 0.6140 | 0.6006 | 验证集最优 |
| Learning rate | `lr=5e-4` | 0.7795 | 0.6020 | 0.6110 | final loss 更低，测试略高 |
| Capacity/reg | `filters=16, hidden=100, reg=1e-3` | 0.7791 | 0.6440 | 0.6350 | 当前 CNN 搜索最优 |
| Capacity/reg | `filters=4, hidden=100, reg=1e-2` | 0.6604 | 0.5640 | 0.5651 | 容量降低后表现下降 |

## 后续方向

后续可在当前最优 `filters=16` 配置上继续尝试更长训练轮数、学习率衰减和轻量数据增强。
