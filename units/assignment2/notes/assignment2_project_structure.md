# Assignment 2 项目结构说明

本文件记录 Assignment 2 当前代码、笔记、数据、实验设计和实验结果的位置。

## 代码入口

- `../code/run_assignment2_experiments.py`：统一 baseline 实验脚本，串联实现自检、RNN smoke test、CIFAR-10 训练、指标保存和图表生成；
- `../code/run_assignment2_explorations.py`：超参数搜索和消融实验入口；
- `../code/cs231n/`：核心 Python 包；
- `../code/cs231n/layers.py`：affine、ReLU、BatchNorm、LayerNorm、Dropout、卷积、池化、Spatial BatchNorm、GroupNorm 和 Softmax loss；
- `../code/cs231n/optim.py`：SGD、Momentum、RMSProp、Adam 优化器；
- `../code/cs231n/classifiers/fc_net.py`：支持 batch/layer normalization 与 dropout 的多层全连接网络；
- `../code/cs231n/classifiers/cnn.py`：ThreeLayerConvNet；
- `../code/cs231n/rnn_layers_pytorch.py`：PyTorch 版 RNN/LSTM 前向算子；
- `../code/cs231n/classifiers/rnn_pytorch.py`：图像 captioning RNN/LSTM 模型前向与采样逻辑。

## 探索实验设计

- `../experiments/assignment2_exploration_suites.json`：探索实验 suite 配置；
- `../experiments/hparam_tuning.md`：超参数调优设计；
- `../experiments/ablation.md`：消融实验设计。

## 数据位置

课程脚本的数据下载位置为 `../code/cs231n/datasets/`：

- `../code/cs231n/datasets/cifar-10-batches-py/`
- `../code/cs231n/datasets/imagenet_val_25.npz`
- `../code/cs231n/datasets/coco_captioning/`

这些原始数据目录和文件由 `.gitignore` 排除。仓库只保留下载脚本，不保留数据本体和压缩包。

## 实验结果

当前 baseline 结果位于：

- `../results/baseline/assignment2_experiment_metrics.csv`：模型指标表；
- `../results/baseline/assignment2_experiment_summary.json`：实验配置、训练历史和 smoke test 结果；
- `../results/baseline/assignment2_experiment_log.txt`：实验运行日志；
- `../results/baseline/assignment2_experiment_summary.md`：中文实验摘要；
- `../results/baseline/figures/assignment2_loss_curves.png`：训练损失曲线；
- `../results/baseline/figures/assignment2_accuracy_comparison.png`：模型准确率对比。

当前超参数探索结果位于：

- `../results/hparam_tuning/conv_learning_rate_search_20260601_215042/`
- `../results/hparam_tuning/conv_capacity_reg_search_20260601_223059/`

当前消融实验结果位于：

- `../results/ablation/fc_normalization_dropout_ablation_20260601_210225/`
- `../results/ablation/conv_regularization_ablation_20260601_211157/`

## 阶段成果

Assignment 2 的阶段性报告位于 `../../../deliverables/week02_assignment2/`：

- `../../../deliverables/week02_assignment2/latex/week2_assignment2_report.tex`
- `../../../deliverables/week02_assignment2/pdf/week2_assignment2_report.pdf`
- `../../../deliverables/week02_assignment2/assignment2_method_supplement.md`
