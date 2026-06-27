# 超参数搜索记录

## 设计原则

以 `baseline_textcnn` 为对照，每次只改变一个主要因素。配置选择只看 validation accuracy；test accuracy 只作为最终观察。

## 搜索空间

| run_id | 改动 | 对照 |
| --- | --- | --- |
| hparam_lr_3e-4 | learning_rate=0.0003 | baseline learning_rate=0.001 |
| hparam_lr_3e-3 | learning_rate=0.003 | baseline learning_rate=0.001 |
| hparam_dropout_0.2 | dropout=0.2 | baseline dropout=0.5 |
| hparam_dropout_0.7 | dropout=0.7 | baseline dropout=0.5 |
| hparam_len_64 | max_seq_len=64 | baseline max_seq_len=128 |
| hparam_len_256 | max_seq_len=256 | baseline max_seq_len=128 |

## 结果摘要

| run_id | best_val_acc | test_acc | 观察 |
| --- | ---: | ---: | --- |
| baseline_textcnn | 0.9192 | 0.9149 | 稳定 baseline |
| hparam_lr_3e-4 | 0.8958 | 0.8899 | 学习率偏小，6 epoch 内收敛不足 |
| hparam_lr_3e-3 | 0.9177 | 0.9128 | 更快但未超过 baseline |
| hparam_dropout_0.2 | 0.9228 | 0.9167 | validation 最优，正则减弱后更适合本配置 |
| hparam_dropout_0.7 | 0.9153 | 0.9120 | dropout 偏强，出现轻微欠拟合 |
| hparam_len_64 | 0.9176 | 0.9128 | 截断较多，略低于 baseline |
| hparam_len_256 | 0.9188 | 0.9163 | test 观察接近最优，但 validation 未超过 dropout=0.2 |

当前选择 `hparam_dropout_0.2` 作为 validation 最优配置。
