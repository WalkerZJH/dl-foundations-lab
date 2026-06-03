# Assignment 1 任务实现规划

## 检查范例代码发现的TODO工作

* kNN：两层循环、一层循环、无循环距离计算，以及投票预测。
* Softmax：朴素实现、向量化实现、梯度计算和数值稳定性处理。
* SVM：为 `LinearSVM` 补充朴素与向量化 loss，修复缺失模块导入。
* LinearClassifier：小批量采样、SGD 参数更新和预测函数。
* 基础层：Affine、ReLU、Softmax loss、SVM loss。
* 训练组件：TwoLayerNet、FullyConnectedNet、Solver、SGD Momentum、RMSProp、Adam。
* 进阶层：BatchNorm、LayerNorm、Dropout、卷积、池化、Spatial BatchNorm、Spatial GroupNorm。

## 实现顺序

1. 先补全 kNN、Softmax、SVM 和 LinearClassifier，保证线性 baseline 可运行。
2. 补全 Affine/ReLU/Softmax loss，支撑 TwoLayerNet。
3. 补全优化器和 Solver 依赖，支撑神经网络训练。
4. 补全 FullyConnectedNet 及 normalization/dropout 相关层。
5. 运行基础行为检查和 full-data 实验，确认模型可以完成训练和预测。
6. 生成实验日志、指标表和可视化图片。
7. 完成实验报告。

## 探索实验阶段

1. 建立 `experiments/assignment1_exploration_suites.json`，用 suite 配置管理超参数搜索和消融实验。
2. 使用 `run_assignment1_explorations.py` 运行 full-data kNN、Softmax、SVM、TwoLayerNet 和 FullyConnectedNet 的探索实验。
3. 每个 suite 输出 CSV、JSON、曲线图和 `run_trace.txt`，便于中断后从合适位置继续。
4. 将实验结论写入 `experiments/hparam_tuning.md`、`experiments/ablation.md` 和阶段报告。
