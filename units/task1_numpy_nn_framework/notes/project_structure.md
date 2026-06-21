# Task1 项目结构

Task1 采用“核心库、运行入口、配置、实验设计、结果”五层结构：

```text
units/task1_numpy_nn_framework/
├── code/
│   ├── numpy_nn/                  # 仅负责模型计算与参数更新
│   ├── data_loading.py            # Digits 与共享 CIFAR-10 加载
│   ├── training.py                # 训练、验证和断点恢复
│   └── run_task1_experiments.py   # 正式 suite 入口
├── configs/                       # baseline 与探索参数
├── experiments/                   # 研究问题和控制变量设计
├── notes/                         # 实现与调试记录
└── results/                       # 轻量结果、曲线和分析
```

`numpy_nn/` 不读取数据、不生成图表，也不包含具体实验参数。`run_task1_experiments.py` 负责组装配置，并在 suite 全部完成后按 validation accuracy 选出配置。测试集评估发生在选择之后。

每个正式 run 的 `results/` 目录只保存 `config.json`、`epoch_metrics.csv`、`run_trace.txt`、`training_summary.json` 和曲线。`training_checkpoint.pkl` 统一存放在 `units/checkpoints/task1_numpy_nn_framework/<dataset>/<suite>/<run>/`，仅用于本地恢复并受 `.gitignore` 排除。
