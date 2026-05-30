# Assignment 2

主要内容包括 Batch Normalization、Layer Normalization、Dropout、卷积与池化、Spatial BatchNorm、GroupNorm、三层卷积网络，以及 PyTorch RNN/LSTM captioning 前向路径。

## 目录说明

* `code/`：核心实现与统一实验脚本。
* `notes/`：项目结构、实现规划和关键实现笔记。
* `results/`：轻量实验结果、运行日志和可视化图片。

## 数据说明

课程数据脚本位于 `code/cs231n/datasets/`：

* `get_datasets.sh`：下载 CIFAR-10 和 `imagenet_val_25.npz`。
* `get_coco_dataset.sh`：下载 COCO captioning 数据。
* `get_imagenet_val.sh`：单独下载 ImageNet 验证样例。

当前实验使用 CIFAR-10 子集，数据实际放在 `code/cs231n/datasets/` 下；原始数据、压缩包和大体积数据文件由 `.gitignore` 排除，不提交到 GitHub。

## 当前实现状态

当前已补全 Assignment 2 的 NumPy 核心层、优化器、`FullyConnectedNet` 的归一化与 dropout 路径、`ThreeLayerConvNet`、以及 PyTorch RNN/LSTM captioning 前向路径。

统一实验入口为：

```powershell
conda run -n minimind python units\assignment2\code\run_assignment2_experiments.py
```

实验结果集中保存在 `results/`，阶段报告保存在 `../../deliverables/week02_assignment2/`。
