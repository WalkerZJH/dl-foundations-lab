# Task1 结果索引

* `dataset_summary.json`：完整 Digits 划分与类别计数。
* `sanity_checks/`：shape、数值梯度、train/eval、卷积一致性与效率。
* `baseline/`：Digits MLP baseline 指标、trace 和曲线。
* `model_comparison/`：Digits MLP、MLP+BN、CNN 对比。
* `hparam_tuning/`：Digits 学习率搜索。
* `ablation/`：Digits BatchNorm/Dropout 消融。
* `cifar10_full/`：完整 CIFAR-10 上的 baseline、模型对比、学习率搜索和模块消融。
* `final_analysis.md`：统一实验分析。

各 suite 的 `metrics.csv` 以 validation accuracy 标记选中项。test 指标只出现在选中行，未选中行保持为空。
