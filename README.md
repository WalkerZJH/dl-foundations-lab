# DL Foundations Lab

本仓库用于记录近期深度学习与 NLP 入门任务的学习过程、代码实现、实验结果和阶段性成果归档。仓库按“任务单元”和“阶段成果”两条线组织，方便自己持续维护。

## 当前仓库结构

* `units/`：按任务单元组织代码、笔记和实验结果。
* `deliverables/`：阶段性成果归档，按周组织报告、学习笔记、LaTeX 源文件和 PDF。

公开查看阶段性成果时，建议优先查看 `deliverables/`；复现实验或检查实现细节时，建议查看 `units/`。

## 当前阶段成果

第 1 周：2026.05.18 - 2026.05.24

成果归档目录：

* `deliverables/week01_assignment1/`

第 2 周：2026.05.25 - 2026.05.30

成果归档目录：

* `deliverables/week02_assignment2/`

第 3 周：2026.06.02 - 2026.06.08

期末备考周，不单独新设成果目录或报告。本周主要完成 Assignment 1 Dropout 固定 seed 异常定位、消融实验重跑、错误归档和第 1 周报告修订，记录见 `deliverables/README.md`。

第 4 周：2026.06.08 - 2026.06.14

期末复习与考试准备，本周未形成独立 deliverable。

第 5 周：2026.06.15 - 2026.06.21

成果归档目录：

* `deliverables/week05_task1/`

主要文件：

* `deliverables/week01_assignment1/week01_report.md`
* `deliverables/week01_assignment1/assignment1_study_notes.md`
* `deliverables/week01_assignment1/latex/week01_assignment1_report.tex`
* `deliverables/week01_assignment1/latex/week01_assignment1_report.pdf`
* `units/assignment1/code/README.md`
* `units/assignment1/code/run_assignment1_experiments.py`
* `deliverables/week02_assignment2/latex/week2_assignment2_report.tex`
* `deliverables/week02_assignment2/latex/week2_assignment2_report.pdf`
* `units/assignment2/code/README.md`
* `units/assignment2/code/run_assignment2_experiments.py`
* `deliverables/week05_task1/latex/week05_task1_report.pdf`
* `units/task1_numpy_nn_framework/code/run_task1_experiments.py`

说明：

第 1 周重点为 CS231n Assignment 1 的学习与代码补全，主要围绕 kNN、Softmax 分类器、SVM、两层神经网络、多层全连接网络、常用网络层和优化器展开。Task1 简单神经网络框架会在 Assignment 1 的基础上继续推进。

Assignment 1 当前已补充 full-data 探索实验结构，包含 kNN 的 `k` 值搜索、线性分类器学习率与正则搜索、TwoLayerNet 超参数搜索，以及初始化、优化器、归一化和 Dropout 消融。

第 2 周重点为 CS231n Assignment 2 的学习与工程化整理，主要围绕归一化、dropout、卷积网络、PyTorch RNN/LSTM captioning 前向路径和统一实验归档展开。

第 3 周不新增阶段目录，重点是对 Assignment 1 既有成果做质量修订：修正 Dropout seed 误用、更新归一化/Dropout 消融结果，并同步周报时间线。

第 4 周主要用于期末复习与考试准备，不单独设置成果目录。第 5 周完成独立 Task1 NumPy 神经网络框架，并在完整 Digits 与 CIFAR-10 数据集上完成正确性验证、MLP baseline、模型对比、学习率搜索和 BatchNorm/Dropout 消融。

## 当前任务进度

* Assignment 1：已完成第一轮实现、实验和报告整理
* Assignment 2：已完成第一轮实现、实验和报告整理
* Task1 NumPy 神经网络框架：已完成第一轮实现、实验和报告整理
* Task2 NLP baseline：待开始

## 面向索引的目录说明

### Assignment 1

* 核心代码包：`units/assignment1/code/cs231n/`
* 代码入口说明：`units/assignment1/code/README.md`
* 实验脚本：`units/assignment1/code/run_assignment1_experiments.py`
* 探索脚本：`units/assignment1/code/run_assignment1_explorations.py`
* 超参数调优记录：`units/assignment1/experiments/hparam_tuning.md`
* 消融实验记录：`units/assignment1/experiments/ablation.md`
* 实现规划：`units/assignment1/notes/assignment1_implementation_plan.md`
* Dropout seed 错误归档：`units/assignment1/notes/assignment1_dropout_seed_error_archive.md`
* 实验结果：`units/assignment1/results/`

复现 baseline 时以 `run_assignment1_experiments.py` 为入口；运行超参数搜索和消融实验时以 `run_assignment1_explorations.py` 为入口。

Assignment 1 数据按课程脚本下载到 `units/assignment1/code/cs231n/datasets/`。该目录中的数据文件不提交到 GitHub，只保留 `get_datasets.sh`。

### Assignment 2

* 核心代码包：`units/assignment2/code/cs231n/`
* 代码入口说明：`units/assignment2/code/README.md`
* 实验脚本：`units/assignment2/code/run_assignment2_experiments.py`
* 探索脚本：`units/assignment2/code/run_assignment2_explorations.py`
* 超参数调优记录：`units/assignment2/experiments/hparam_tuning.md`
* 消融实验记录：`units/assignment2/experiments/ablation.md`
* 项目结构说明：`units/assignment2/notes/assignment2_project_structure.md`
* 关键实现笔记：`units/assignment2/notes/assignment2_key_implementation_notes.md`
* 实验结果：`units/assignment2/results/`

复现实验时以 `run_assignment2_experiments.py` 为统一入口，探索实验以 `run_assignment2_explorations.py` 为入口，当前 PyTorch 相关路径使用 conda `minimind` 环境运行。当前正式结果使用 CIFAR-10 完整训练设置；ThreeLayerConvNet baseline test acc 为 0.6151，容量搜索最佳 test acc 为 0.6350。

Assignment 2 数据按课程脚本下载到 `units/assignment2/code/cs231n/datasets/`。该目录中的数据文件不提交到 GitHub，只保留下载脚本。

### Task1 NumPy 神经网络框架

* 单元入口：`units/task1_numpy_nn_framework/README.md`
* 核心代码包：`units/task1_numpy_nn_framework/code/numpy_nn/`
* 正式实验入口：`units/task1_numpy_nn_framework/code/run_task1_experiments.py`
* 实验设计：`units/task1_numpy_nn_framework/experiments/experiment_plan.md`
* 统一结果分析：`units/task1_numpy_nn_framework/results/final_analysis.md`

Task1 使用完整 Digits 与 CIFAR-10 数据集。scikit-learn 仅负责读取内置 Digits 和分层划分；模型计算、反向传播和参数更新均由独立 NumPy 框架完成。配置按 validation accuracy 选择，test set 只用于选择后的最终观察。

### 阶段成果归档

* 第 1 周成果目录：`deliverables/week01_assignment1/`
* 第 1 周报告 LaTeX：`deliverables/week01_assignment1/latex/week01_assignment1_report.tex`
* 第 1 周报告 PDF：`deliverables/week01_assignment1/latex/week01_assignment1_report.pdf`
* 第 1 周周报：`deliverables/week01_assignment1/week01_report.md`
* 第 1 周学习笔记：`deliverables/week01_assignment1/assignment1_study_notes.md`
* 第 2 周成果目录：`deliverables/week02_assignment2/`
* 第 2 周报告 LaTeX：`deliverables/week02_assignment2/latex/week2_assignment2_report.tex`
* 第 2 周报告 PDF：`deliverables/week02_assignment2/latex/week2_assignment2_report.pdf`
* 第 2 周周报：`deliverables/week02_assignment2/week2_report.md`
* 第 5 周成果目录：`deliverables/week05_task1/`
* 第 5 周报告 LaTeX：`deliverables/week05_task1/latex/week05_task1_report.tex`
* 第 5 周报告 PDF：`deliverables/week05_task1/latex/week05_task1_report.pdf`
* 第 5 周周报：`deliverables/week05_task1/week05_report.md`

阶段性总结、报告和可公开查看的 PDF 优先放入 `deliverables/`，过程代码和实验资产保留在 `units/`。
