# Digits MLP Baseline 摘要

完整 Digits 分层划分下，64-128-10 MLP 使用 Adam、learning rate 0.001、batch size 64、L2 正则 $10^{-4}$ 训练 40 epochs。

训练 loss 从 1.9146 持续下降到 0.0508，训练准确率达到 0.9960。第 40 个 epoch 取得最高 validation accuracy 0.9814，并在完成选择后观察到 test accuracy 0.9815。训练和验证曲线无发散或异常尖峰，baseline 可作为后续控制变量实验的对照。

原始记录：`metrics.csv`、`baseline_mlp/epoch_metrics.csv`、`baseline_mlp/run_trace.txt` 和 `baseline_mlp/training_curves.png`。
