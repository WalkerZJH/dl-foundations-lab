# ThreeLayerConvNet 泛化问题补充分析

## 快速结论

早期快速验证口径下，`ThreeLayerConvNet` 出现过“训练准确率明显高于测试准确率”的现象。重新使用 CIFAR-10 完整训练设置后，该问题显著缓解：

| 实验 | Train Acc | Val Acc | Test Acc | Train-Val Gap |
| --- | ---: | ---: | ---: | ---: |
| full-data baseline：filters=8, hidden=100, reg=1e-3 | 0.7571 | 0.6110 | 0.6151 | 0.1461 |
| capacity search：filters=16, hidden=100, reg=1e-3 | 0.7791 | 0.6440 | 0.6350 | 0.1351 |

结论是：此前看似严重的泛化不足，主要来自训练样本过少。完整训练设置下，CNN 的验证和测试表现都明显提升；进一步增加卷积核数量后，测试准确率达到 `0.6350`。

## 实验设置

正式实验使用：

```text
train = 49000
val = 1000
test = 10000
optimizer = adam
learning_rate = 1e-3
epochs = 10
batch_size = 100
filter_size = 3
weight_scale = 1e-2
```

baseline 使用 `num_filters=8, hidden_dim=100, reg=1e-3`。容量搜索额外比较了 `num_filters`、`hidden_dim` 和 `reg`。

## L2 正则消融

| 配置 | Train Acc | Val Acc | Test Acc | Train-Val Gap |
| --- | ---: | ---: | ---: | ---: |
| `reg=0` | 0.7702 | 0.6140 | 0.6091 | 0.1562 |
| `reg=1e-3` | 0.7572 | 0.6140 | 0.6006 | 0.1432 |
| `reg=1e-2` | 0.7012 | 0.6100 | 0.6157 | 0.0912 |

强 L2 正则降低了训练准确率，也缩小了 train-val gap。`reg=1e-2` 的测试准确率略高，但验证准确率没有超过 `reg=0` 与 `reg=1e-3`，因此不能简单说“更强正则就是更好”。更合理的解释是：正则项可以压低容量和 gap，但最终模型选择仍应以验证集表现为主。

## 容量搜索

| 配置 | Train Acc | Val Acc | Test Acc |
| --- | ---: | ---: | ---: |
| filters=8, hidden=100, reg=1e-3 | 0.7572 | 0.6140 | 0.6006 |
| filters=8, hidden=100, reg=1e-2 | 0.7012 | 0.6100 | 0.6157 |
| filters=4, hidden=100, reg=1e-2 | 0.6604 | 0.5640 | 0.5651 |
| filters=8, hidden=50, reg=1e-2 | 0.6969 | 0.6060 | 0.6159 |
| filters=16, hidden=100, reg=1e-3 | 0.7791 | 0.6440 | 0.6350 |

降低卷积核数量会明显降低表现；增加到 16 个卷积核后，验证和测试准确率同步提升。这说明在完整训练数据下，模型容量增加没有导致明显不可控的过拟合，反而带来了更好的表示能力。

## 分析

ThreeLayerConvNet 的结构为：

```text
conv - relu - 2x2 max pool - affine - relu - affine - softmax
```

该结构的主要参数集中在池化后的全连接层，但卷积层决定了模型能提取多少空间特征通道。在训练样本不足时，模型容易拟合偶然的局部纹理或背景模式；当训练样本扩大到完整训练设置后，样本多样性提高，测试表现随之改善。

当前仍存在一定 train-val gap，但它更像是三层 CNN 在无数据增强、无更深结构、10 epochs 设置下的正常容量差距。后续可以从三个方向继续验证：

- 在 `filters=16` 配置上延长训练并加入学习率衰减；
- 尝试轻量数据增强；
- 比较更深 CNN 或加入 Spatial BatchNorm 后的稳定性。
