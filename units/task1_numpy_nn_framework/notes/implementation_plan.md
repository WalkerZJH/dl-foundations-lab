# Task1 实现规划

## 实现顺序

1. 建立 `Module`、`Parameter`、状态字典和 train/eval 递归切换。
2. 实现 Linear、ReLU、Flatten、Softmax Cross Entropy。
3. 实现 BatchNorm 与 Dropout，隔离 running statistics 和随机生成器。
4. 实现 naive Conv2D，再用 im2col/col2im 完成 vectorized Conv2D，补充 MaxPool2D。
5. 实现 SGD、Momentum、Adam 和 MLP/CNN 构建器。
6. 接入完整 Digits 数据集、分层划分、训练器和逐 epoch 断点。
7. 依次执行 shape、数值梯度、train/eval、卷积一致性与效率检查。
8. 运行 baseline、模型对比、学习率搜索和 BN/Dropout 消融。
9. 依据 validation accuracy 选择配置，最后观察 test set。

## 验证门槛

* 核心 Python 文件通过 `compileall`。
* Linear、BatchNorm、Conv2D 数值梯度相对误差低于 $10^{-6}$。
* naive/vectorized 卷积前后向相对误差低于 $10^{-10}$。
* Dropout 在训练阶段随机、评估阶段恒等；BatchNorm 评估阶段确定且不更新 running statistics。
* 正式实验使用全部 1797 张样本的固定分层划分。
