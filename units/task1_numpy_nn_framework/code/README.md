# Task1 代码入口

`numpy_nn/` 是独立 NumPy 框架，训练和实验逻辑不进入核心层文件。

| 路径 | 用途 |
| --- | --- |
| `numpy_nn/module.py` | Module、Parameter、Sequential 与状态管理 |
| `numpy_nn/layers.py` | Linear、ReLU、Flatten、BatchNorm、Dropout |
| `numpy_nn/convolution.py` | naive/vectorized Conv2D 与 MaxPool2D |
| `numpy_nn/losses.py` | Softmax Cross Entropy |
| `numpy_nn/optimizers.py` | SGD、Momentum、Adam |
| `numpy_nn/models.py` | MLP、MLP+BN、CNN 构建入口 |
| `data_loading.py` | Digits 与 CIFAR-10 加载、归一化及分层划分 |
| `training.py` | 训练、验证、断点和 best-validation 状态恢复 |
| `run_task1_experiments.py` | 正式 suite 统一入口 |

正式 suite：`baseline`、`model_comparison`、`learning_rate_search`、`normalization_dropout_ablation`、`all`。使用 `--force` 可清理所选 suite 的本地断点并重新运行。

核心包不导入 PyTorch、TensorFlow 或自动求导工具。scikit-learn 仅在 `data_loading.py` 中用于读取内置 Digits 和 `train_test_split(..., stratify=...)`。

实验入口默认运行 Digits；传入 `--dataset cifar10` 时读取 Assignment 1 下载目录中的 CIFAR-10 Python batches，并将正式结果写入 `results/cifar10_full/`。
