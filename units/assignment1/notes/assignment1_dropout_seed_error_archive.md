# Assignment 1 Dropout 固定随机种子错误归档

本文记录一次 FullyConnectedNet Dropout 消融实验中的实现失误。该记录用于保留调试过程和修正依据，旧结果不再作为正式消融结论使用。

## 问题来源

`units/assignment1/code/run_assignment1_explorations.py` 在构造 `FullyConnectedNet` 时固定传入了 `seed=123`：

```python
FullyConnectedNet(
    ...,
    dropout_keep_ratio=variant.get("dropout_keep_ratio", 1.0),
    ...
    seed=123,
)
```

在 CS231n 风格实现中，`dropout_forward` 会在检测到 `dropout_param["seed"]` 时执行 `np.random.seed(...)`。这个设计主要用于梯度检查，使 dropout mask 可复现；它不适合作为正式训练时每次 forward 都执行的固定随机种子。

## 失误发展的过程

1. 探索脚本为了让实验可复现，给所有 `FullyConnectedNet` 统一传入了固定 seed。
2. Dropout 消融中只有 `A1-AB-NORM-004` 启用了 dropout，因此只有这条曲线受到该实现影响。
3. 训练过程中，`loss_history` 记录的是 minibatch training loss；epoch 末尾的 `check_accuracy()` 不会把 validation/test loss 写入同一个 loss 列表。
4. 由于 `dropout_forward()` 每次 forward 都重置全局 `np.random` 状态，训练 step 的 minibatch 抽样和 dropout mask 被异常耦合。
5. epoch 末尾的评估路径进入 test mode，不生成 dropout mask，随机数消耗路径与普通训练 step 不一致，导致下一个 epoch 开头的 minibatch 抽样状态发生跳变。

## 导致的结果

旧结果目录：

`units/assignment1/results/ablation/normalization_dropout_ablation_20260601_174826/`

该结果中 `A1-AB-NORM-004` 出现异常现象：

| 指标 | 旧结果 |
| --- | ---: |
| val acc | 0.2830 |
| test acc | 0.2720 |
| final loss | 0.0762 |

loss 曲线表现为在 epoch 边界附近出现周期性尖峰，且 final loss 与分类准确率不匹配。该现象不应解释为 dropout 的正常训练行为。

## 修正方式

正式探索脚本改为只在 suite 配置显式提供 `seed` 时传入 dropout seed：

```python
seed=variant.get("seed")
```

这样保留了需要固定 dropout mask 的调试入口，同时避免正式训练时反复重置全局随机数状态。

## 修正后结果

新结果目录：

`units/assignment1/results/ablation/normalization_dropout_ablation_20260608_115235/`

修正后 `A1-AB-NORM-004` 的指标为：

| 指标 | 新结果 |
| --- | ---: |
| val acc | 0.4480 |
| test acc | 0.4383 |
| final loss | 1.7743 |

重跑后的 dropout loss 曲线不再出现旧结果中的周期性尖峰。正式文档和报告均以新结果目录为准。

## 后续注意

- 正式训练实验不要在每次 dropout forward 内固定全局随机种子。
- 如果需要梯度检查或单元测试复现 dropout mask，应显式在测试配置中传入 seed，并避免把该配置复用到正式训练。
