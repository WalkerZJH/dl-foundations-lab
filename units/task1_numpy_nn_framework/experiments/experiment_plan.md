# Task1 实验计划

## 研究问题

| 编号 | 问题 | 控制方式 | 结果位置 |
| --- | --- | --- | --- |
| RQ1 | 手写 forward/backward 是否正确？ | 数值梯度、shape 和 train/eval 检查 | `../results/sanity_checks/` |
| RQ2 | im2col 向量化是否与 naive 卷积一致并提升效率？ | 固定输入、权重与上游梯度 | `../results/sanity_checks/` |
| RQ3 | MLP、MLP+BN、CNN 在 Digits 与 CIFAR-10 上的收敛和验证表现有何差异？ | 固定划分、epoch、优化器和 seed | `../results/model_comparison/`、`../results/cifar10_full/model_comparison/` |
| RQ4 | 学习率如何影响收敛速度和验证表现？ | 仅改变 learning rate | `../results/hparam_tuning/`、`../results/cifar10_full/hparam_tuning/` |
| RQ5 | BatchNorm 与 Dropout 如何改变训练过程和泛化差距？ | 以 plain MLP 为对照，仅改变模块 | `../results/ablation/`、`../results/cifar10_full/ablation/` |

## 数据与评价

使用完整 scikit-learn Digits 数据集，共 1797 张 $8\times8$ 图像。固定 seed 42，分层划分为训练 1257、验证 269、测试 271。像素由 $[0,16]$ 缩放到 $[0,1]$，不使用数据增强。

追加实验使用完整 CIFAR-10，共 60000 张 $32\times32$ 彩色图像。官方训练部分分层划分为训练 45000、验证 5000，官方测试集保留 10000 张。结果归档在 `../results/cifar10_full/`。

训练记录 cross-entropy、accuracy、运行时间和参数量。配置选择只依据 best validation accuracy；test set 在选择完成后用于最终观察。

## 正式实验矩阵

下列矩阵分别在完整 Digits 与完整 CIFAR-10 数据集上执行，结果按对应数据集目录独立归档。

* baseline：MLP，hidden dim 128，Adam，learning rate 0.001，batch size 64，40 epochs。
* 模型对比：MLP、MLP+BatchNorm、CNN。
* 学习率：0.0005、0.001、0.003。
* 消融：plain MLP、MLP+BatchNorm、MLP+Dropout（keep ratio 0.8）。
* 卷积效率：naive 与 vectorized forward，固定张量和重复次数。
