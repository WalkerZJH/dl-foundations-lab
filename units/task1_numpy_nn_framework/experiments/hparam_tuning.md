# Task1 Digits 学习率探索

## 设置

在完整 Digits 数据集下，固定 MLP（64-128-10）、He 初始化、Adam、batch size 64、L2 正则 $10^{-4}$、40 epochs、数据划分和随机种子，仅改变学习率：$5\times10^{-4}$、$10^{-3}$、$3\times10^{-3}$。

结果位于 `../results/hparam_tuning/metrics.csv`，每个 run 的逐 epoch 曲线和 trace 位于同目录子文件夹。

## Digits 结果

| 学习率 | 最佳 epoch | 选择时训练准确率 | 最佳验证准确率 | 选择时验证 loss |
| ---: | ---: | ---: | ---: | ---: |
| 0.0005 | 34 | 0.9753 | 0.9628 | 0.2053 |
| 0.0010 | 40 | 0.9960 | 0.9814 | 0.1166 |
| 0.0030 | 26 | 0.9992 | **0.9851** | 0.0936 |

$5\times10^{-4}$ 的 loss 下降稳定，但在 40 epochs 内仍明显慢于另外两组。$3\times10^{-3}$ 在前 10 个 epoch 更快进入高准确率区间，并在第 26 个 epoch 首次达到本组最高 validation accuracy。后续 validation accuracy 在离散样本上波动，训练准确率继续接近 1，说明更长训练主要继续压低训练误差。

按 validation accuracy 选中 0.003 后观察 test accuracy 为 0.9815。该测试结果只用于报告，不参与学习率选择。
