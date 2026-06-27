# 项目结构说明

`units/task2_classification` 是 Task2 的正式单元目录。代码、配置、实验设计、运行结果和学习归档分开保存，便于公开查看和复现。

主要路径：

* `code/`: 数据集读取、模型、训练、评估和实验入口。
* `configs/`: baseline、超参数搜索、模型对比和消融配置。
* `experiments/`: 实验设计、超参数搜索记录和消融记录。
* `notes/`: 实现计划、关键实现笔记和调试记录。
* `results/`: correctness checks、正式 run 输出、汇总表、图表和最终分析。

共享数据不放在本单元下，而是统一放在 `units/data/ag_news/`；该目录由 `.gitignore` 排除。checkpoint 统一保留给 `units/checkpoints/`，本次正式结果没有提交模型权重。
