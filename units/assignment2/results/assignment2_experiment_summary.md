# Assignment 2 实验摘要

## 数据与配置

* CIFAR-10 子集：train=2000，val=500，test=500。
* 训练配置：epochs=10，batch_size=100，conv_filters=8。

## 烟测结果

| 检查项 | 结果 |
| --- | --- |
| NumPy 层梯度烟测 | passed |
| PyTorch RNN Captioning 烟测 | passed |

烟测只用于确认实现路径可运行，不参与正式模型性能比较。

## 正式模型结果

| 模型 | 设置 | 训练准确率 | 验证准确率 | 测试准确率 | 最终 loss |
| --- | --- | ---: | ---: | ---: | ---: |
| FullyConnectedNet + BN + Dropout | adam, lr=1e-3, epochs=10, hidden=[100,100], keep=0.8 | 0.5680 | 0.3420 | 0.3200 | 2.2012 |
| ThreeLayerConvNet | adam, lr=1e-3, epochs=10, filters=8, filter_size=3 | 0.8520 | 0.4620 | 0.4000 | 0.4084 |

## 说明

正式结果仅来自 CIFAR-10 训练实验；烟测结果只记录通过或失败。当前正式训练轮数为 10 epochs。
