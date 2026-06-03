# Assignment 1 结果目录

本目录保存 Assignment 1 的轻量实验结果，包括指标表、运行日志、结构化摘要、训练曲线和准确率对比图。当前正式结果使用 CIFAR-10 完整训练设置：训练集 49000 张、验证集 1000 张、测试集 10000 张。原始 CIFAR-10 数据、大体积日志和模型参数不提交。

## Baseline 结果

当前统一 baseline 结果保存在 `baseline/`：

- `baseline/assignment1_experiment_metrics.csv`：Softmax、TwoLayerNet、FullyConnectedNet 的实验指标；
- `baseline/assignment1_experiment_log.txt`：实验运行日志；
- `baseline/assignment1_experiment_summary.md`：实验摘要；
- `baseline/assignment1_experiment_summary.json`：结构化结果；
- `baseline/figures/assignment1_loss_curves.png`：训练损失曲线；
- `baseline/figures/assignment1_accuracy_comparison_full_data.png`：准确率对比图，供 LaTeX 报告引用；
- `baseline/figures/assignment1_accuracy_comparison_full_data.svg`：准确率对比图。

kNN 的 full-data 结果由 chunked 探索脚本生成，保存在 `hparam_tuning/`。

## 超参数调优结果

当前已完成的超参数调优结果保存在 `hparam_tuning/`：

- `hparam_tuning/knn_k_hard_search_20260601_172938/`：kNN 的 `k` 值硬搜索；
- `hparam_tuning/knn_k_elbow_search_20260601_173222/`：kNN 的 elbow 选择；
- `hparam_tuning/softmax_lr_reg_search_20260601_173447/`：Softmax 学习率与 L2 正则搜索；
- `hparam_tuning/svm_lr_reg_search_20260601_173521/`：SVM 学习率与 L2 正则搜索；
- `hparam_tuning/two_layer_lr_search_20260601_173552/`：TwoLayerNet 学习率搜索；
- `hparam_tuning/two_layer_l2_search_20260601_173815/`：TwoLayerNet L2 正则搜索；
- `hparam_tuning/batch_size_search_20260601_174037/`：batch size 搜索。

## 消融实验结果

当前已完成的消融实验结果保存在 `ablation/`：

- `ablation/init_ablation_20260601_174349/`：初始化方式消融；
- `ablation/optimizer_ablation_20260601_174556/`：优化器消融；
- `ablation/normalization_dropout_ablation_20260601_174826/`：归一化与 Dropout 消融。

每个 suite 输出目录包含：

- `*_metrics.csv`
- `*_summary.json`
- `*_loss_curves.png`（适用于有训练 loss 的实验）
- `*_accuracy.png`
- `run_trace.txt`

实验解释和复盘维护在 `../experiments/`。
