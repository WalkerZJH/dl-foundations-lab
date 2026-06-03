# Assignment 2 代码目录

本目录保存 CS231n Assignment 2 的可运行代码。

## 入口

- `run_assignment2_experiments.py`：统一 baseline 实验脚本，负责实现自检、PyTorch/RNN smoke test、CIFAR-10 实验、指标保存和可视化生成；
- `run_assignment2_explorations.py`：超参数探索和消融实验入口；
- `../experiments/assignment2_exploration_suites.json`：探索实验 suite 配置；
- `cs231n/`：核心 Python 包，包含 layers、optim、solver、分类器、CNN、RNN/LSTM captioning 等实现。

## Baseline 运行方式

当前 PyTorch 依赖使用本机 conda 环境 `minimind`：

```powershell
D:\anaconda\envs\minimind\python.exe units\assignment2\code\run_assignment2_experiments.py
```

baseline 结果写入 `../results/baseline/`，默认使用 CIFAR-10 完整训练设置：`train=49000`、`val=1000`、`test=10000`、`epochs=10`。

## 探索实验运行方式

```powershell
D:\anaconda\envs\minimind\python.exe units\assignment2\code\run_assignment2_explorations.py --list-suites
D:\anaconda\envs\minimind\python.exe units\assignment2\code\run_assignment2_explorations.py --suite conv_capacity_reg_search --dry-run
```

探索脚本从 `../experiments/assignment2_exploration_suites.json` 读取 suite，生成 CSV、JSON 和曲线图。

常用 suite：

- `conv_capacity_reg_search`：ThreeLayerConvNet 容量与 L2 正则搜索；
- `conv_learning_rate_search`：ThreeLayerConvNet 学习率搜索；
- `fc_normalization_dropout_ablation`：FullyConnectedNet 归一化与 Dropout 消融；
- `conv_regularization_ablation`：ThreeLayerConvNet L2 正则消融。

数据下载脚本保留在 `cs231n/datasets/`。下载后的 CIFAR-10、COCO、ImageNet 样例等原始数据不提交。
