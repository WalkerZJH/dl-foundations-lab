# 2026.06.15-2026.06.21 第 5 周周报

时间范围：2026.06.15 - 2026.06.21

## 本周工作

* 将 Task1 按独立库重新设计，未复制 Assignment 代码；实现 Module/Parameter、状态管理和 train/eval 切换。
* 使用 NumPy 实现 Linear、ReLU、BatchNorm、Dropout、naive/vectorized Conv2D、MaxPool2D、Softmax Cross Entropy、SGD、Momentum 和 Adam 的核心路径。
* 使用完整 scikit-learn Digits 数据集，共 1797 张图像，分层划分为训练 1257、验证 269、测试 271。
* 完成 shape、数值梯度、train/eval 模式、naive/vectorized 卷积一致性与效率检查。
* 完成 MLP baseline、MLP/MLP+BN/CNN 对比、学习率搜索和 BatchNorm/Dropout 消融。
* 追加完整 CIFAR-10 对照，使用 45000/5000/10000 划分复用相同实验矩阵。
* 为正式训练加入逐 epoch CSV trace、随机状态和本地 checkpoint，支持相同配置中断恢复。

## 主要结果

Digits 数据集下，MLP baseline 的 validation accuracy 为 0.9814，选择完成后的 test accuracy 为 0.9815。学习率搜索中 0.003 取得最高 validation accuracy 0.9851。MLP+BN 与 MLP 的最高 validation accuracy 持平，但 validation loss 更低；CNN 在当前 8×8 输入和单卷积结构下为 0.9777。

Digits 数据集下，Dropout keep ratio 0.8 在验证集上取得 0.9851，但选择后的 test accuracy 为 0.9742。该结果按既定规则保留，不根据 test 反向调整；269 张验证集上的一个样本即可改变约 0.0037 accuracy，后续需要多 seed 重复判断稳定性。

数值梯度误差均低于 $10^{-6}$，naive/vectorized 卷积前后向误差低于 $10^{-14}$。vectorized forward 在固定 benchmark 中约快 300 倍。

完整 CIFAR-10 上，MLP baseline validation/test accuracy 为 0.4648/0.4627；模型对比选中 CNN，validation/test accuracy 为 0.6250/0.6089。学习率搜索选中 0.0005，模块消融选中 BatchNorm。该组结果提供了高于 Digits 的结构区分度，也表明学习率与 Dropout 结论不能直接跨数据集迁移。

## 问题与修正

训练器初版存在配置层级读取错误和聚合 CSV 字段不一致，均在正式结果生成前修正。第一次探索运行还发现 suite 内 seed 不一致会破坏控制变量，相关结果已废弃，并在固定 seed 后完整重跑模型对比、学习率搜索和消融。

## 结果入口

* 代码与复现：`../../units/task1_numpy_nn_framework/README.md`
* 正确性检查：`../../units/task1_numpy_nn_framework/results/sanity_checks/`
* 正式分析：`../../units/task1_numpy_nn_framework/results/final_analysis.md`
* 详细报告：`latex/week05_task1_report.pdf`
