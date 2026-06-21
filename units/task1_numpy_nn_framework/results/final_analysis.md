# Task1 正式实验分析

## 正确性与效率

Linear、BatchNorm、vectorized Conv2D 的最大数值梯度相对误差分别处于 $10^{-10}$、$10^{-11}$ 和 $10^{-9}$ 量级，均低于 $10^{-6}$ 门槛。naive/vectorized 卷积前后向最大相对误差低于 $10^{-14}$。当前固定 benchmark 中 naive forward 平均 0.2806 秒，vectorized 平均 0.0009 秒；短时计时存在波动，量级上约为 300 倍加速。

Dropout 训练输出随调用变化，评估阶段为确定的恒等映射；BatchNorm 评估阶段输出确定且 running mean 不更新。

## Digits Baseline

Digits 数据集下，MLP baseline 在第 40 个 epoch 达到训练准确率 0.9960、validation accuracy 0.9814。完成验证集选择后，test accuracy 为 0.9815。loss 曲线持续下降，训练与验证走势一致，未观察到实现错误对应的异常震荡。

## Digits 模型对比

| 模型 | 参数量 | 最佳 epoch | 训练准确率 | 验证准确率 | 运行时间/秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP | 9610 | 40 | 0.9960 | **0.9814** | 0.59 |
| MLP+BN | 9866 | 37 | 1.0000 | **0.9814** | 0.75 |
| CNN | 8986 | 39 | 0.9960 | 0.9777 | 2.62 |

Digits 数据集下，MLP+BN 在早期收敛更快，并以更低 validation loss 达到与 MLP 相同的最高准确率。CNN 没有在当前 8×8 输入和单卷积层结构下超过 MLP，且 NumPy 卷积训练耗时约为 MLP 的 4.4 倍。Digits 的低分辨率和已对齐输入削弱了局部结构建模带来的优势，CNN 结构还需要更针对性的通道数和学习率搜索。

## Digits 学习率

Digits 数据集下，学习率从 0.0005 增至 0.003 后，40 epochs 内的最佳 validation accuracy 从 0.9628 提高到 0.9851。0.0005 的稳定性没有转化为有限预算内的充分收敛；0.003 更快降低 loss，且未出现发散。选中 0.003 后 test accuracy 为 0.9815。

## Digits BatchNorm / Dropout

Digits 数据集下，BatchNorm 与 plain MLP 的最高 validation accuracy 均为 0.9814，但 BatchNorm 的选择时 validation loss 从 0.1166 降至 0.0869。Dropout keep ratio 0.8 将 validation accuracy 提高到 0.9851，并把选择时训练准确率降至 0.9928，表现出更强的训练扰动。

Dropout 选中后的 test accuracy 为 0.9742，没有复现验证集上的一个样本优势。由于验证集只有 269 张，accuracy 的最小变化约为 0.0037；当前结果支持“Dropout 改变了拟合程度”，但不足以支持“Dropout 稳定提升泛化”。更可靠的判断需要固定实验矩阵后进行多 seed 重复，并报告均值与标准差。

## 完整 CIFAR-10 对照

CIFAR-10 使用官方完整数据，分层划分为训练 45000、验证 5000、测试 10000，沿用 40 epochs、batch size 64 和 validation accuracy 选参规则。

| 实验 | 选中配置 | 最佳验证准确率 | 选中后测试准确率 |
| --- | --- | ---: | ---: |
| Baseline | MLP，lr=0.001 | 0.4648 | 0.4627 |
| 模型对比 | CNN | 0.6250 | 0.6089 |
| 学习率搜索 | MLP，lr=0.0005 | 0.4980 | 0.4931 |
| 模块消融 | MLP+BatchNorm | 0.5060 | 0.5085 |

模型对比中，MLP、MLP+BN、CNN 的最佳验证准确率分别为 0.4648、0.5060、0.6250。CNN 以 132010 个参数超过约 39.5 万参数的两种 MLP，说明更复杂图像上的局部连接与权重共享形成了可观察优势。学习率 0.0005 优于 0.001 和 0.003，与 Digits 的选择相反，表明学习率不能跨数据集直接复用。

CIFAR-10 训练曲线显示，MLP baseline 的验证指标在中后期波动；CNN 在约第 22 个 epoch 后训练指标继续改善，但验证 loss 上升且验证准确率趋于平台。正式指标使用各配置的 best-validation 快照，不使用最后一轮状态。

消融中，BatchNorm 将验证准确率从 0.4648 提高到 0.5060，并降低 validation loss；Dropout keep ratio 0.8 的验证准确率为 0.3826，同时训练准确率也降至 0.3900，当前训练预算下表现为优化不足。该结果不支持泛化提升结论，后续需要联合搜索 Dropout 强度与学习率。

## 结论

本轮结果验证了独立 NumPy 框架的前后向正确性、训练能力和断点恢复路径。Digits 上接近饱和的结果适合确认流程，但模型差异有限；完整 CIFAR-10 对照进一步区分了 CNN、BatchNorm、学习率和 Dropout 的影响。所有 test 结果均在 validation 选择完成后生成，没有用于反向修改配置。
