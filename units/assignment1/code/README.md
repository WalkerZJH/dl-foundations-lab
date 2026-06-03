# Assignment 1 代码目录

本目录保存 CS231n Assignment 1 的核心实现和实验入口。

## 主要内容

- `cs231n/`：核心代码包；
- `cs231n/classifiers/`：kNN、Softmax、SVM、TwoLayerNet、FullyConnectedNet 等分类器；
- `cs231n/datasets/get_datasets.sh`：课程数据下载脚本；
- `run_assignment1_experiments.py`：阶段性 baseline 复现实验入口；
- `run_assignment1_explorations.py`：超参数探索、消融实验和诊断实验入口。
- `../experiments/assignment1_exploration_suites.json`：探索实验 suite 配置。

## Baseline 入口

```powershell
python units\assignment1\code\run_assignment1_experiments.py
```

该脚本用于复现当前阶段的统一 baseline，正式归档结果保存在 `../results/baseline/`。

## 探索实验入口

```powershell
python units\assignment1\code\run_assignment1_explorations.py --list-suites
python units\assignment1\code\run_assignment1_explorations.py --suite knn_k_hard_search
python units\assignment1\code\run_assignment1_explorations.py --suite knn_k_elbow_search
python units\assignment1\code\run_assignment1_explorations.py --suite two_layer_lr_search
```

探索脚本从 `../experiments/assignment1_exploration_suites.json` 读取 suite，适合修改数据规模、训练轮数、batch size 或选择不同实验组。脚本生成 CSV、JSON、曲线图和 `run_trace.txt`。

常用 suite：

- `knn_k_hard_search`：kNN 的 `k` 值硬搜索；
- `knn_k_elbow_search`：kNN 的 `k` 值 elbow 选择；
- `softmax_lr_reg_search`：Softmax 学习率与 L2 正则搜索；
- `svm_lr_reg_search`：SVM 学习率与 L2 正则搜索；
- `two_layer_lr_search`：TwoLayerNet 学习率搜索；
- `two_layer_l2_search`：TwoLayerNet L2 正则搜索；
- `batch_size_search`：batch size 搜索；
- `init_ablation`：初始化方式消融；
- `optimizer_ablation`：优化器消融；
- `normalization_dropout_ablation`：归一化与 Dropout 消融。

实验结果按 suite 类型写入 `../results/` 下的分组目录。实验动机、现象分析。
