# Assignment 2 项目结构说明

本目录用于说明 CS231n Assignment 2 当前代码、笔记、数据和实验结果的位置。

## 代码入口

* `../code/run_assignment2_experiments.py`：统一实验脚本，串联实现自检、RNN smoke test、CIFAR-10 训练、指标保存和图表生成。
* `../code/cs231n/`：核心 Python 包。
* `../code/cs231n/layers.py`：affine、ReLU、BatchNorm、LayerNorm、Dropout、卷积、池化、Spatial BatchNorm、GroupNorm 和 Softmax loss。
* `../code/cs231n/optim.py`：SGD、Momentum、RMSProp、Adam 优化器。
* `../code/cs231n/classifiers/fc_net.py`：支持 batch/layer normalization 与 dropout 的多层全连接网络。
* `../code/cs231n/classifiers/cnn.py`：三层卷积网络。
* `../code/cs231n/rnn_layers_pytorch.py`：PyTorch 版 RNN/LSTM 前向算子。
* `../code/cs231n/classifiers/rnn_pytorch.py`：图像 captioning RNN/LSTM 模型前向与采样逻辑。

## 数据位置

课程脚本的数据下载位置为 `../code/cs231n/datasets/`：

* `../code/cs231n/datasets/cifar-10-batches-py/`
* `../code/cs231n/datasets/imagenet_val_25.npz`
* `../code/cs231n/datasets/coco_captioning/`

这些原始数据目录和文件由 `.gitignore` 排除。仓库只保留下载脚本，不保留数据本体和压缩包。

## 实验结果

* `../results/assignment2_experiment_metrics.csv`：模型指标表。
* `../results/assignment2_experiment_summary.json`：实验配置、数值自检、训练历史和 PyTorch/RNN smoke test 结果。
* `../results/assignment2_experiment_log.txt`：实验运行日志。
* `../results/assignment2_experiment_summary.md`：中文实验摘要。
* `../results/figures/assignment2_loss_curves.png`：训练损失曲线。
* `../results/figures/assignment2_accuracy_comparison.png`：模型准确率对比。

## 阶段成果

Assignment 2 的阶段性报告位于 `../../../deliverables/week02_assignment2/`：

* `../../../deliverables/week02_assignment2/latex/week2_assignment2_report.tex`
* `../../../deliverables/week02_assignment2/latex/week2_assignment2_report.pdf`
* `../../../deliverables/week02_assignment2/assignment2_method_supplement.md`
