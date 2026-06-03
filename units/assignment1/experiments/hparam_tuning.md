# Assignment 1 超参数调优记录

本文件记录 Assignment 1 的超参数探索设计、结果位置和观察结论。当前正式结果使用 CIFAR-10 完整训练设置：训练集 49000 张、验证集 1000 张、测试集 10000 张。

## kNN hard search 与 elbow 选择

| 项目 | 内容 |
| --- | --- |
| 调整的超参数 | `k ∈ {1, 3, 5, 7, 9, 11, 13, 15}` |
| 设计问题 | hard search 选择验证准确率最高的 `k`；elbow 选择接近平台期的较小 `k`。 |
| 控制变量 | 固定 CIFAR-10 完整训练设置、L2 距离和预处理方式。 |
| 结果位置 | `../results/hparam_tuning/knn_k_hard_search_20260601_172938/`；`../results/hparam_tuning/knn_k_elbow_search_20260601_173222/` |
| 观察结果 | hard search 和 elbow 都选择 `k=1`，val acc 0.3570，test acc 0.3513。 |

kNN 采用 chunked 距离计算，避免一次性保存 10000 x 49000 的测试距离矩阵。

## Softmax / SVM 的 learning rate 与 L2 正则

| 项目 | Softmax | SVM |
| --- | --- | --- |
| 调整的超参数 | `learning_rate`、`reg` | `learning_rate`、`reg` |
| 控制变量 | 固定完整数据划分、迭代次数、batch size 和初始化方式 | 固定完整数据划分、迭代次数、batch size 和初始化方式 |
| 结果位置 | `../results/hparam_tuning/softmax_lr_reg_search_20260601_173447/` | `../results/hparam_tuning/svm_lr_reg_search_20260601_173521/` |
| 最佳验证配置 | `lr=1e-7, reg=1e4` | `lr=1e-7, reg=1e4` |
| 观察结果 | val acc 0.3780，test acc 0.3523 | val acc 0.3860，test acc 0.3760 |

完整训练集上，较低正则强度 `1e4` 优于 `2.5e4`，说明过强的 L2 约束会压低线性分类器容量。

## TwoLayerNet 学习率搜索

| 项目 | 内容 |
| --- | --- |
| 调整的超参数 | `learning_rate ∈ {1e-2, 1e-3, 1e-4}` |
| 控制变量 | 固定 TwoLayerNet 结构、初始化方式、L2 正则、batch size 和训练轮数。 |
| 结果位置 | `../results/hparam_tuning/two_layer_lr_search_20260601_173552/` |
| 观察结果 | `lr=1e-4` 取得 val acc 0.4920、test acc 0.4802；`lr=1e-2` 出现 NaN，表现为发散。 |

## TwoLayerNet L2 正则搜索

| 项目 | 内容 |
| --- | --- |
| 调整的超参数 | `reg ∈ {0, 1e-3, 0.1}` |
| 控制变量 | 固定模型结构、学习率、batch size 和训练轮数。 |
| 结果位置 | `../results/hparam_tuning/two_layer_l2_search_20260601_173815/` |
| 观察结果 | 固定 `lr=1e-3` 时三个配置均明显弱于学习率搜索中的 `lr=1e-4`，该结果主要说明学习率是当前主导因素。 |

## batch size 搜索

| 项目 | 内容 |
| --- | --- |
| 调整的超参数 | `batch_size ∈ {64, 128, 200}` |
| 控制变量 | 固定模型结构、学习率、正则强度和训练轮数。 |
| 结果位置 | `../results/hparam_tuning/batch_size_search_20260601_174037/` |
| 观察结果 | 固定 `lr=1e-3` 下 batch size=128 的 val acc 最高，为 0.2400；各配置均存在 loss 过大或不稳定现象。 |
