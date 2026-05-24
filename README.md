# DL Foundations Lab

本仓库用于记录近期深度学习与 NLP 入门任务的学习过程、代码实现、实验结果和阶段性成果归档。仓库按“任务单元”和“阶段成果”两条线组织，方便自己持续维护，也方便公开访问者快速定位代码、笔记和报告。

## 当前仓库结构

* `units/`：按任务单元组织代码、笔记和实验结果。
* `deliverables/`：阶段性成果归档，按周组织报告、学习笔记、LaTeX 源文件和 PDF。

公开查看阶段性成果时，建议优先查看 `deliverables/`；复现实验或检查实现细节时，建议查看 `units/`。

## 当前阶段成果

第 1 周：2026.05.18 - 2026.05.24

成果归档目录：

* `deliverables/week01_assignment1/`

主要文件：

* `deliverables/week01_assignment1/week01_report.md`
* `deliverables/week01_assignment1/assignment1_study_notes.md`
* `deliverables/week01_assignment1/latex/week01_assignment1_report.tex`
* `deliverables/week01_assignment1/pdf/week01_assignment1_report.pdf`
* `units/assignment1/README.md`
* `units/assignment1/code/run_assignment1_experiments.py`

说明：

第 1 周重点为 CS231n Assignment 1 的学习与代码补全，主要围绕 kNN、Softmax 分类器、SVM、两层神经网络、多层全连接网络、常用网络层和优化器展开。Task1 简单神经网络框架会在 Assignment 1 的基础上继续推进。

## 当前任务进度

* Assignment 1：已完成第一轮实现、实验和报告整理
* Assignment 2：待开始
* Assignment 3：待开始
* Task1 NumPy 神经网络框架：待开始
* Task2 NLP baseline：待开始

## 说明

本仓库不上传原始数据集、大体积模型参数文件、训练日志和 LaTeX 编译中间文件。可公开归档的报告 PDF 会保留在 `deliverables/` 目录中。未实际使用的目录暂不加入索引，后续启用时再补充说明。

## 面向索引的目录说明

本节用于帮助公开访问者和后续 agent 快速定位代码、笔记、实验结果和阶段性成果。

### Assignment 1

* 单元入口：`units/assignment1/README.md`
* 核心代码包：`units/assignment1/code/cs231n/`
* 实验脚本：`units/assignment1/code/run_assignment1_experiments.py`
* 项目结构说明：`units/assignment1/notes/assignment1_project_structure.md`
* 实现规划：`units/assignment1/notes/assignment1_implementation_plan.md`
* 实验结果：`units/assignment1/results/`

Assignment 1 已清理原始 notebook、notebook 转换产物和当前实验不再使用的辅助脚本；复现实验时以 `run_assignment1_experiments.py` 为统一入口。

Assignment 1 数据按课程脚本下载到 `units/assignment1/code/cs231n/datasets/`。该目录中的数据文件不提交到 GitHub，只保留 `get_datasets.sh`。

### 阶段成果归档

* 第 1 周成果目录：`deliverables/week01_assignment1/`
* 第 1 周报告 LaTeX：`deliverables/week01_assignment1/latex/week01_assignment1_report.tex`
* 第 1 周报告 PDF：`deliverables/week01_assignment1/pdf/week01_assignment1_report.pdf`
* 第 1 周周报：`deliverables/week01_assignment1/week01_report.md`
* 第 1 周学习笔记：`deliverables/week01_assignment1/assignment1_study_notes.md`

阶段性总结、报告和可公开查看的 PDF 优先放入 `deliverables/`，过程代码和实验资产保留在 `units/`。
