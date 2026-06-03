# Assignment 2 结果目录

本目录保存 Assignment 2 的轻量实验结果，包括指标表、运行日志、摘要和可视化图片。

## Baseline 结果

当前统一 baseline 结果保存在 `baseline/`，正式设置为 `train=49000`、`val=1000`、`test=10000`、`epochs=10`：

- `baseline/assignment2_experiment_metrics.csv`：模型指标表；
- `baseline/assignment2_experiment_log.txt`：实验运行日志；
- `baseline/assignment2_experiment_summary.md`：实验摘要；
- `baseline/assignment2_experiment_summary.json`：结构化结果；
- `baseline/run_trace.txt`：运行 trace；
- `baseline/figures/assignment2_loss_curves.png`：训练损失曲线；
- `baseline/figures/assignment2_accuracy_comparison.png`：模型准确率对比。

核心结果：

| 模型 | Train Acc | Val Acc | Test Acc |
| --- | ---: | ---: | ---: |
| FullyConnectedNet + BN + Dropout | 0.3464 | 0.4010 | 0.3450 |
| ThreeLayerConvNet | 0.7571 | 0.6110 | 0.6151 |

## 超参数调优结果

当前已归档的探索结果保存在 `hparam_tuning/`：

- `hparam_tuning/conv_learning_rate_search_20260601_215042/`
- `hparam_tuning/conv_capacity_reg_search_20260601_223059/`

核心结果：

| 实验 | 选中配置 | Val Acc | Test Acc |
| --- | --- | ---: | ---: |
| CNN learning rate | `lr=1e-3` | 0.6140 | 0.6006 |
| CNN capacity/reg | `filters=16, hidden=100, reg=1e-3` | 0.6440 | 0.6350 |

## 消融实验结果

当前已归档的消融结果保存在 `ablation/`：

- `ablation/fc_normalization_dropout_ablation_20260601_210225/`
- `ablation/conv_regularization_ablation_20260601_211157/`

核心结果：

| 实验 | 选中/观察配置 | Val Acc | Test Acc |
| --- | --- | ---: | ---: |
| FC normalization/dropout | 无归一化、无 Dropout | 0.4420 | 0.4394 |
| CNN L2 regularization | `reg=1e-2` | 0.6100 | 0.6157 |

实验解释和复盘写入 `../experiments/` 或 `../notes/`。
