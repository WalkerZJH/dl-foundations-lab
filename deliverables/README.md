# 阶段成果时间线

本目录用于按阶段归档学习成果。

## 时间线索引

| 阶段 | 时间范围 | 主题 | 状态 | 成果目录 |
| --- | --- | --- | --- | --- |
| 第 1 周 | 2026.05.18 - 2026.05.24 | CS231n Assignment 1：基础分类器与全连接网络 | 已完成第一轮实现、实验和报告整理 | `week01_assignment1/` |

## 第 1 周：CS231n Assignment 1

时间范围：2026.05.18 - 2026.05.24

本阶段围绕 CS231n Assignment 1 展开，重点是把课程中的交互式 Notebook 任务整理为可维护的 Python 模块、统一实验脚本和阶段性报告。当前成果目录为 `week01_assignment1/`，过程代码和实验结果位于 `../units/assignment1/`。

### 主要任务

* 整理 Assignment 1 的仓库结构，区分代码、学习笔记、实验结果和阶段成果归档。
* 补全 kNN、Softmax、SVM、LinearClassifier、TwoLayerNet、FullyConnectedNet、常用网络层和优化器相关实现。
* 将 Assignment 1 的实验流程统一到 `../units/assignment1/code/run_assignment1_experiments.py`。
* 使用 CIFAR-10 子集运行实验，并生成指标表、实验日志和可视化图片。
* 编写学习笔记、项目结构说明、实现规划和 LaTeX 报告。

### 当前成果

* 周报：`week01_assignment1/week01_report.md`
* 报告 PDF：`week01_assignment1/pdf/week01_assignment1_report.pdf`
* 单元入口：`../units/assignment1/README.md`
* 实验脚本：`../units/assignment1/code/run_assignment1_experiments.py`
* 实验结果：`../units/assignment1/results/`

### 实验概况

当前实验使用 CIFAR-10 子集：训练集 2000 张、验证集 500 张、测试集 500 张。kNN 使用完整实验子集进行距离计算，不再额外采样。

主要结果如下：

| 模型 | 设置 | 训练准确率 | 验证准确率 | 测试准确率 |
| --- | --- | ---: | ---: | ---: |
| kNN | k=7, train=2000, val=500, test=500 | - | 0.2800 | 0.2320 |
| Softmax | lr=1e-7, reg=2.5e4, iters=500 | 0.3410 | 0.3080 | 0.2960 |
| TwoLayerNet | sgd_momentum, lr=0.001, epochs=3 | 0.2560 | 0.2820 | 0.2760 |
| FullyConnectedNet | adam, lr=0.001, epochs=3 | 0.5700 | 0.2860 | 0.2320 |

### 阶段小结

第 1 周完成了 Assignment 1 的第一轮工程化整理：从 Notebook 任务转为纯 Python 模块结构，并形成可复现实验入口。实验结果表明 Softmax 在当前小规模 CIFAR-10 子集上测试准确率最高，FullyConnectedNet 已出现一定过拟合，后续可围绕更充分的训练、正则化和 Task1 NumPy 神经网络框架继续推进。
