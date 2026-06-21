# Task1：NumPy 神经网络框架

本单元从零实现一个仅依赖 NumPy 完成模型计算与反向传播的轻量神经网络框架，并在完整 Digits 与 CIFAR-10 数据集上验证训练能力。代码不复用 CS231n Assignment 实现，按独立库、实验入口和结果归档组织。

## 主要入口

* 核心包：`code/numpy_nn/`
* 数据加载：`code/data_loading.py`
* 正式实验：`code/run_task1_experiments.py`
* baseline 配置：`configs/baseline.json`
* 探索 suite：`configs/experiment_suites.json`
* 实验设计：`experiments/experiment_plan.md`
* 统一结果分析：`results/final_analysis.md`
* 阶段报告：`../../deliverables/week05_task1/`

## 已实现内容

框架包含 Linear、ReLU、BatchNorm、Dropout、naive/vectorized Conv2D、MaxPool2D、Softmax Cross Entropy、SGD、Momentum 与 Adam。各核心层均显式实现 forward/backward，并通过 shape、数值梯度、train/eval 模式和卷积一致性检查。

Digits 共 1797 张 $8\times8$ 灰度图像，按 70%/15%/15% 分层划分为 1257/269/271。配置选择只依据 validation accuracy；test set 仅在选定配置后观察。

CIFAR-10 对照使用完整 60000 张图像，划分为训练 45000、验证 5000、测试 10000。结果保存在 `results/cifar10_full/`，沿用相同的 validation 选参和 test 最终观察规则。

## 复现

在仓库根目录使用 `minimind` 环境的解释器：

```powershell
& 'D:\anaconda\envs\minimind\python.exe' -m compileall -f units/task1_numpy_nn_framework/code
& 'D:\anaconda\envs\minimind\python.exe' units/task1_numpy_nn_framework/code/run_task1_experiments.py --suite all
& 'D:\anaconda\envs\minimind\python.exe' units/task1_numpy_nn_framework/code/run_task1_experiments.py --dataset cifar10 --suite all
```

正式训练按 epoch 保存 CSV trace，并把本地 checkpoint 写入 `units/checkpoints/task1_numpy_nn_framework/<dataset>/<suite>/<run>/`。checkpoint 与正式结果分离且由 `.gitignore` 排除；重复运行会从相同配置的断点恢复。
