# Assignment 2 实验摘要

## 数据与配置

* CIFAR-10 完整训练设置：train=49000，val=1000，test=10000。
* 训练配置：epochs=10，batch_size=100，conv_filters=8，num_train_samples=None。

## 烟测结果

| 检查项 | 结果 |
| --- | --- |
| NumPy 层梯度烟测 | passed |
| PyTorch RNN Captioning 烟测 | passed |

烟测只用于确认实现路径可运行，不参与正式模型性能比较。

## 正式模型结果

| 模型 | 设置 | 完整训练准确率 | 验证准确率 | 测试准确率 | 最终 loss |
| --- | --- | ---: | ---: | ---: | ---: |
| FullyConnectedNet + BN + Dropout | adam, lr=1e-3, epochs=10, hidden=[100,100], keep=0.8 | 0.3464 | 0.4010 | 0.3450 | 2.0812 |
| ThreeLayerConvNet | adam, lr=1e-3, epochs=10, filters=8, filter_size=3 | 0.7571 | 0.6110 | 0.6151 | 0.6619 |

## 说明

正式结果仅来自 CIFAR-10 训练实验；烟测结果只记录通过或失败。当前正式训练轮数为 10 epochs。
