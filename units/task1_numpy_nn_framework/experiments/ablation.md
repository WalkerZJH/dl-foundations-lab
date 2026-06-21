# Task1 Digits BatchNorm / Dropout 消融

## 设置

完整 Digits 数据集下的对照组为 64-128-10 MLP。三组共享数据划分、初始化 seed、batch 顺序、Adam、learning rate 0.001、batch size 64、L2 正则 $10^{-4}$ 和 40 epochs。

* `plain_mlp`：无 BatchNorm、无 Dropout；
* `with_batchnorm`：在第一层 Linear 与 ReLU 之间加入 BatchNorm；
* `with_dropout`：在 ReLU 后加入 inverted Dropout，keep ratio 0.8。

## Digits 结果

| 配置 | 最佳 epoch | 选择时训练准确率 | 最佳验证准确率 | 选择时验证 loss |
| --- | ---: | ---: | ---: | ---: |
| plain MLP | 40 | 0.9960 | 0.9814 | 0.1166 |
| + BatchNorm | 37 | 1.0000 | 0.9814 | **0.0869** |
| + Dropout | 40 | 0.9928 | **0.9851** | 0.1208 |

BatchNorm 更快提高前期准确率，并在相同最高 validation accuracy 下取得更低 validation loss，说明预测分布的置信质量有所改善；同时训练集达到 1.0，未在本设置中缩小训练与验证差距。

Dropout 降低选择时训练准确率，并将 validation accuracy 提高一个离散步长（269 张验证集上的 1 个样本约为 0.0037）。按验证集选中后，其 test accuracy 为 0.9742，低于 baseline 的 0.9815。该差异说明单次固定划分上的一个样本优势不足以证明稳定提升，后续结论需要多 seed 重复实验支持；本轮仍遵守验证集选择规则，不依据 test 回退选择。
