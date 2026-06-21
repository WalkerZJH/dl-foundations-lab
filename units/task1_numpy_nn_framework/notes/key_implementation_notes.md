# Task1 关键实现笔记

## 参数与状态

`Parameter` 同时保存 `data` 和 `grad`，优化器持有具名参数引用。`Module.state_dict()` 将参数与 BatchNorm running mean/variance 分开编码，因此 best-validation 快照能完整恢复推理状态。

## BatchNorm

训练阶段按 batch 计算均值和方差，并更新 running statistics；反向传播使用化简公式一次计算输入梯度。评估阶段仅使用 running statistics，且不保留 backward cache。

## Dropout

参数使用 keep ratio 语义，`0.8` 表示保留 80%、丢弃 20%。每个 Dropout 实例只在构造时创建独立 `np.random.Generator`，forward 不重置全局随机状态。inverted dropout 在训练阶段除以 keep ratio，评估阶段直接返回输入。

## 卷积

naive 版本通过显式窗口循环实现，便于核对索引与梯度累积。vectorized 版本将滑动窗口展开为列矩阵，forward 转为矩阵乘法，backward 通过 `col2im` 的 `np.add.at` 累加重叠窗口梯度。

## 训练与选择

每轮结束计算完整 train/validation 指标并写入 trace。训练器保存当前状态、优化器状态、随机生成器状态和 best-validation 快照。suite 完成后只根据 validation accuracy 选择 run，再加载该 run 的 best 状态观察 test；未选中行不写 test 指标。
