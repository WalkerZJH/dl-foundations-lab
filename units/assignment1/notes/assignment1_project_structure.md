# Assignment 1 项目结构说明

本目录用于说明 CS231n Assignment 1 当前代码、笔记、数据和实验结果的位置。
## 代码入口

* `../code/cs231n/`：核心 Python 包，包含数据读取、分类器、网络层、优化器和训练器。
* `../code/cs231n/classifiers/`：分类器实现，包含 kNN、Softmax、SVM、TwoLayerNet 和 FullyConnectedNet。
* `../code/cs231n/layers.py`：全连接层、ReLU、BatchNorm、LayerNorm、Dropout、卷积、池化、SVM loss 和 Softmax loss。
* `../code/cs231n/optim.py`：SGD、Momentum、RMSProp、Adam 优化器。
* `../code/cs231n/solver.py`：神经网络训练循环。
* `../code/run_assignment1_experiments.py`：统一实验脚本，负责运行 kNN、Softmax、TwoLayerNet 和 FullyConnectedNet，并生成结果。

## 数据位置

`get_datasets.sh` 的数据下载位置为：

* `../code/cs231n/datasets/cifar-10-batches-py/`
* `../code/cs231n/datasets/imagenet_val_25.npz`

下载后的数据文件由 `.gitignore` 忽略，不提交到 GitHub。当前 Assignment 1 的实际数据读取目录仅为 `../code/cs231n/datasets/`。

## 实验结果

* `../results/assignment1_experiment_metrics.csv`：实验指标表。
* `../results/assignment1_experiment_log.txt`：实验运行日志。
* `../results/assignment1_experiment_summary.md`：实验结果摘要。
* `../results/figures/assignment1_loss_curves.png`：训练损失曲线。
* `../results/figures/assignment1_accuracy_comparison_full_knn.svg`：使用完整 kNN 实验子集后的模型准确率对比。
