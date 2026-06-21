# 阶段成果时间线

本目录用于按阶段归档学习成果。

## 时间线索引

| 阶段 | 时间范围 | 主题 | 状态 | 成果目录 |
| --- | --- | --- | --- | --- |
| 第 1 周 | 2026.05.18 - 2026.05.24 | CS231n Assignment 1：基础分类器与全连接网络 | 已完成第一轮实现、实验和报告整理 | `week01_assignment1/` |
| 第 2 周 | 2026.05.25 - 2026.05.30 | CS231n Assignment 2：归一化、卷积网络与 RNN 前向路径 | 已完成第一轮实现、实验和报告整理 | `week02_assignment2/` |
| 第 3 周 | 2026.06.02 - 2026.06.08 | Assignment 1 Dropout 异常更正与报告修订 | 期末备考周，不单独设成果目录 | 本时间线记录 |
| 第 4 周 | 2026.06.08 - 2026.06.14 | 期末复习与考试准备 | 未形成独立 deliverable | 本时间线记录 |
| 第 5 周 | 2026.06.15 - 2026.06.21 | Task1：NumPy 神经网络框架 | 已完成第一轮实现、实验和报告整理 | `week05_task1/` |

## 第 1 周：CS231n Assignment 1

时间范围：2026.05.18 - 2026.05.24

本阶段围绕 CS231n Assignment 1 展开，重点是把课程中的交互式 Notebook 任务整理为可维护的 Python 模块、统一实验脚本和阶段性报告。当前成果目录为 `week01_assignment1/`，过程代码和实验结果位于 `../units/assignment1/`。

### 主要任务

* 整理 Assignment 1 的仓库结构，区分代码、学习笔记、实验结果和阶段成果归档。
* 补全 kNN、Softmax、SVM、LinearClassifier、TwoLayerNet、FullyConnectedNet、常用网络层和优化器相关实现。
* 将 Assignment 1 的实验流程统一到 `../units/assignment1/code/run_assignment1_experiments.py`。
* 使用 CIFAR-10 完整训练设置运行 baseline 实验，并生成指标表、实验日志和可视化图片。
* 建立探索实验 suite，完成 kNN、线性分类器、TwoLayerNet 和 FullyConnectedNet 的超参数搜索与消融实验。
* 编写学习笔记、项目结构说明、实现规划和研究型 LaTeX 报告。

### 当前成果

* 周报：`week01_assignment1/week01_report.md`
* 报告 PDF：`week01_assignment1/latex/week01_assignment1_report.pdf`
* 代码入口说明：`../units/assignment1/code/README.md`
* 实验脚本：`../units/assignment1/code/run_assignment1_experiments.py`
* 探索脚本：`../units/assignment1/code/run_assignment1_explorations.py`
* 实验结果：`../units/assignment1/results/`

### 实验概况

当前正式实验使用 CIFAR-10 完整训练设置：训练集 49000 张、验证集 1000 张、测试集 10000 张。kNN 使用完整训练集并采用 chunked 距离计算。

主要结果如下：

| 模型 | 设置 | 训练准确率 | 验证准确率 | 测试准确率 |
| --- | --- | ---: | ---: | ---: |
| kNN | k=1, train=49000, val=1000, test=10000 | - | 0.3570 | 0.3513 |
| Softmax | lr=1e-7, reg=2.5e4, iters=1500 | 0.3314 | 0.3460 | 0.3312 |
| TwoLayerNet | sgd_momentum, lr=0.001, epochs=10 | 0.2204 | 0.2440 | 0.2218 |
| FullyConnectedNet | adam, lr=0.001, epochs=10 | 0.4670 | 0.4640 | 0.4360 |

探索实验补充结果：

| 实验 | 选中配置 | 验证准确率 | 测试准确率 |
| --- | --- | ---: | ---: |
| kNN hard / elbow | k=1 | 0.3570 | 0.3513 |
| SVM lr/reg search | lr=1e-7, reg=1e4 | 0.3860 | 0.3760 |
| TwoLayerNet lr search | lr=1e-4 | 0.4920 | 0.4802 |
| Optimizer ablation | SGD | 0.4960 | 0.4855 |
| Normalization ablation | no norm / no dropout | 0.4840 | 0.4637 |

### 阶段小结

第 1 周完成了 Assignment 1 的工程化整理：从 Notebook 任务转为纯 Python 模块结构，并形成 baseline 与 full-data 探索实验入口。探索结果表明，TwoLayerNet 对学习率敏感，较大学习率会导致 loss 发散；在当前完整训练设置下，SGD 和较小学习率组合优于默认 momentum 配置。后续可围绕更充分的训练、正则化和 Task1 NumPy 神经网络框架继续推进。

## 第 2 周：CS231n Assignment 2

时间范围：2026.05.25 - 2026.05.30

本阶段围绕 CS231n Assignment 2 展开，重点是把归一化、dropout、卷积网络和 PyTorch RNN captioning 的 Notebook转化为.py并填充代码、完成统一实验脚本和阶段性报告。当前成果目录为 `week02_assignment2/`，过程代码和实验结果位于 `../units/assignment2/`。

### 主要任务

* 补全 BatchNorm、LayerNorm、Dropout、卷积、池化、Spatial BatchNorm 和 GroupNorm。
* 补全 Momentum、RMSProp、Adam、FullyConnectedNet、ThreeLayerConvNet 和 PyTorch RNN/LSTM captioning 前向路径。
* 将 Assignment 2 的实验流程统一到 `../units/assignment2/code/run_assignment2_experiments.py`。
* 使用 CIFAR-10 完整训练设置运行 baseline、超参数搜索和消融实验，并生成指标表、实验日志和可视化图片。
* 编写项目结构说明、关键实现笔记和 LaTeX 报告。

### 当前成果

* 报告 PDF：`week02_assignment2/latex/week2_assignment2_report.pdf`
* 代码入口说明：`../units/assignment2/code/README.md`
* 实验脚本：`../units/assignment2/code/run_assignment2_experiments.py`
* 实验结果：`../units/assignment2/results/`

### 实验概况

当前正式实验使用 CIFAR-10 完整训练设置：训练集 49000 张、验证集 1000 张、测试集 10000 张，训练 10 epochs。

主要结果如下：

| 模型 | 设置 | 训练准确率 | 验证准确率 | 测试准确率 |
| --- | --- | ---: | ---: | ---: |
| FullyConnectedNet + BN + Dropout | adam, lr=1e-3, epochs=10, hidden=[100,100], keep=0.8 | 0.3464 | 0.4010 | 0.3450 |
| ThreeLayerConvNet | adam, lr=1e-3, epochs=10, filters=8, filter_size=3 | 0.7571 | 0.6110 | 0.6151 |
| ThreeLayerConvNet capacity search | filters=16, hidden=100, reg=1e-3 | 0.7791 | 0.6440 | 0.6350 |

## 第 3 周：Assignment 1 Dropout 异常更正

时间范围：2026.06.02 - 2026.06.08

本周处于期末备考阶段，不新设阶段目录，也不新增独立成果报告。主要工作集中在 Assignment 1 的 Dropout 消融实验复核与更正：

* 定位 `run_assignment1_explorations.py` 中正式训练误传固定 `seed=123` 的问题。
* 确认旧 Dropout 曲线的周期性尖峰来自每次 `dropout_forward()` 重置全局随机数状态，而不是测试集 loss 混入。
* 修正探索脚本，使 dropout seed 只在 suite 显式配置时传入。
* 重跑 `normalization_dropout_ablation`，正式结果更新为 `../units/assignment1/results/ablation/normalization_dropout_ablation_20260608_115235/`。
* 新增错误归档笔记：`../units/assignment1/notes/assignment1_dropout_seed_error_archive.md`。
* 修订第 1 周 LaTeX 报告，移除发散型 baseline/optimizer loss 图，改用文字说明，并纳入修正后的 Dropout 消融图。

该周工作属于已有 Assignment 1 成果的质量修订，相关代码、结果和报告仍归入 `../units/assignment1/` 与 `week01_assignment1/`。

## 第 4 周：期末复习与考试准备

时间范围：2026.06.08 - 2026.06.14

本周主要进行期末复习与考试准备，没有形成独立 deliverable。

## 第 5 周：Task1 NumPy 神经网络框架

时间范围：2026.06.15 - 2026.06.21

本阶段将后续任务切换为独立 task 工作流。Task1 不复制 Assignment 代码，使用 NumPy 实现模块、前向传播、反向传播、优化器和训练器，并在完整 Digits 与 CIFAR-10 数据集上完成正式实验。

### 主要任务

* 实现 Linear、ReLU、BatchNorm、Dropout、naive/vectorized Conv2D、MaxPool2D 与 Softmax Cross Entropy。
* 实现 SGD、Momentum、Adam、显式 train/eval 模式和可恢复的逐 epoch checkpoint；断点统一保存到 `units/checkpoints/`，不混入结果目录。
* 完成 shape、数值梯度、模式切换、卷积一致性和效率验证。
* 完成 MLP baseline、MLP/MLP+BN/CNN 对比、学习率搜索和 BatchNorm/Dropout 消融。
* 依据 validation accuracy 选择配置，test set 仅用于选择完成后的观察。

### 当前成果

* 成果目录：`week05_task1/`
* 周报：`week05_task1/week05_report.md`
* 方法补充：`week05_task1/task1_method_supplement.md`
* 报告 PDF：`week05_task1/latex/week05_task1_report.pdf`
* 单元入口：`../units/task1_numpy_nn_framework/README.md`
* 正式实验入口：`../units/task1_numpy_nn_framework/code/run_task1_experiments.py`
* 统一结果分析：`../units/task1_numpy_nn_framework/results/final_analysis.md`

### 实验概况

Digits 共 1797 张 $8\times8$ 图像，分层划分为训练 1257、验证 269、测试 271。MLP baseline validation accuracy 为 0.9814，选择后的 test accuracy 为 0.9815；学习率 0.003 的 validation accuracy 为 0.9851。完整 CIFAR-10 对照使用 45000/5000/10000 划分，模型对比中 CNN validation/test accuracy 为 0.6250/0.6089。naive/vectorized 卷积前后向结果一致，固定 forward benchmark 中向量化实现约快 300 倍。
