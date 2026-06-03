# 第 2 周阶段报告：CS231n Assignment 2

时间范围：2026.05.25 - 2026.05.30

本阶段围绕 CS231n Assignment 2 展开，重点是将 Notebook 形式的归一化、dropout、卷积网络和 PyTorch RNN captioning 任务整理为可复现的纯 Python 工作流。

## 阶段成果

* 补全 Assignment 2 核心实现，代码位于 `../../units/assignment2/code/cs231n/`。
* 建立统一实验入口 `../../units/assignment2/code/run_assignment2_experiments.py`。
* 使用 CIFAR-10 完整训练设置运行 baseline、超参数搜索和消融实验，结果位于 `../../units/assignment2/results/`。
* 编写项目结构、实现规划和关键实现笔记，位于 `../../units/assignment2/notes/`。
* 编写 LaTeX 报告，位于 `latex/`。

## 实验概况

正式实验使用训练集 49000 张、验证集 1000 张、测试集 10000 张，训练 10 epochs，batch size 为 100。

| 实验 | 代表配置 | Train Acc | Val Acc | Test Acc |
| --- | --- | ---: | ---: | ---: |
| Baseline FC | FullyConnectedNet + BN + Dropout | 0.3464 | 0.4010 | 0.3450 |
| Baseline CNN | ThreeLayerConvNet, filters=8, reg=1e-3 | 0.7571 | 0.6110 | 0.6151 |
| CNN 容量搜索 | filters=16, hidden=100, reg=1e-3 | 0.7791 | 0.6440 | 0.6350 |
| FC 消融 | 无归一化、无 Dropout | 0.4429 | 0.4420 | 0.4394 |

完整报告 PDF 位于 `pdf/week2_assignment2_report.pdf`。
