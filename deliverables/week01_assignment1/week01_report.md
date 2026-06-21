# 第 1 周周报

时间范围：2026.05.18 - 2026.05.24

## 本周完成内容

* 建立深度学习与 NLP 入门任务的学习仓库结构。
* 整理 CS231n Assignment 1 的学习范围、代码目录和成果归档目录。
* 梳理 kNN、Softmax、SVM、TwoLayerNet、FullyConnectedNet、优化器和向量化实现的学习重点。
* 将原始 Jupyter Notebook 任务整理为纯 Python 模块和统一实验脚本。
* 补全 Assignment 1 相关框架代码，完成 CIFAR-10 完整训练设置 baseline 实验，并生成指标表、日志和可视化图片。
* 建立探索实验 suite，完成 kNN、线性分类器、TwoLayerNet 和 FullyConnectedNet 的超参数搜索与消融实验。
* 编写 Assignment 1 的研究型 LaTeX 报告，并归档 PDF。

## 本周重点

本周工作重点为 CS231n Assignment 1 的学习与代码补全，主要围绕基础分类器、全连接神经网络和优化器展开。阶段性成果保存在 `deliverables/week01_assignment1/`，过程代码、实现规划和实验结果保存在 `units/assignment1/`。

## 当前结果

当前实验使用 CIFAR-10 完整训练设置，训练集 49000 张、验证集 1000 张、测试集 10000 张。kNN 使用完整训练集，并通过 chunked 距离计算保留运行痕迹。

baseline 结果保存在 `units/assignment1/results/baseline/`；探索实验结果保存在 `units/assignment1/results/hparam_tuning/` 和 `units/assignment1/results/ablation/`。本轮探索中，SVM 搜索取得 test acc 0.3760，TwoLayerNet 学习率搜索取得 test acc 0.4802，优化器消融中的 SGD 取得 test acc 0.4855；归一化/Dropout 消融已在修正固定 dropout seed 后重跑，正式结果保存在 `units/assignment1/results/ablation/normalization_dropout_ablation_20260608_115235/`。报告 PDF 保存在 `deliverables/week01_assignment1/latex/week01_assignment1_report.pdf`。
