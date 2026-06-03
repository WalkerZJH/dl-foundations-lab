# 阶段成果时间线

本目录用于按阶段归档学习成果。

## 时间线索引

| 阶段 | 时间范围 | 主题 | 状态 | 成果目录 |
| --- | --- | --- | --- | --- |
| 第 1 周 | 2026.05.18 - 2026.05.24 | CS231n Assignment 1：基础分类器与全连接网络 | 已完成第一轮实现、实验和报告整理 | `week01_assignment1/` |
| 第 2 周 | 2026.05.25 - 2026.05.30 | CS231n Assignment 2：归一化、卷积网络与 RNN 前向路径 | 已完成第一轮实现、实验和报告整理 | `week02_assignment2/` |

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
* 报告 PDF：`week01_assignment1/pdf/week01_assignment1_report.pdf`
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

* 报告 PDF：`week02_assignment2/pdf/week2_assignment2_report.pdf`
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
