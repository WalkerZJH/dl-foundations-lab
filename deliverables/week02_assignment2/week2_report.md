# 第 2 周阶段报告：CS231n Assignment 2

时间范围：2026.05.25 - 2026.05.30

本阶段围绕 CS231n Assignment 2 展开，重点是将 Notebook 形式的归一化、dropout、卷积网络和 PyTorch RNN captioning 任务整理为可复现的纯 Python 工作流。

## 阶段成果

* 补全 Assignment 2 核心实现，代码位于 `../../units/assignment2/code/cs231n/`。
* 建立统一实验入口 `../../units/assignment2/code/run_assignment2_experiments.py`。
* 生成实验结果、日志和可视化，位于 `../../units/assignment2/results/`。
* 编写项目结构、实现规划和关键实现笔记，位于 `../../units/assignment2/notes/`。
* 编写 LaTeX 报告和方法补充，位于 `latex/` 与 `assignment2_method_supplement.md`。

## 实验概况

当前正式实验使用 CIFAR-10 子集：训练集 2000 张、验证集 500 张、测试集 500 张，训练 10 epochs。运行环境为 conda `minimind`。烟测只记录 NumPy layer 与 PyTorch/RNN captioning 路径是否通过，不进入正式性能比较。

| 模型 | 训练准确率 | 验证准确率 | 测试准确率 |
| --- | ---: | ---: | ---: |
| FullyConnectedNet + BN + Dropout | 0.5680 | 0.3420 | 0.3200 |
| ThreeLayerConvNet | 0.8520 | 0.4620 | 0.4000 |

当前结果表明，在相同子集和正式训练配置下，三层卷积网络比全连接网络更好地利用图像空间结构。
