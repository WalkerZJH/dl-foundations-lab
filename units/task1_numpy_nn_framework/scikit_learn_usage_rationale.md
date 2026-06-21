# scikit-learn 使用边界说明

Task1 使用 scikit-learn 的原因只有两项：

1. `load_digits()` 提供稳定、可自动获取的内置手写数字数据集，不需要下载和提交原始数据文件；
2. `train_test_split(..., stratify=labels)` 保证 70%/15%/15% 划分中的类别比例基本一致，减少手写分层采样代码带来的数据泄漏和类别偏差风险。

scikit-learn 不参与模型定义、前向传播、反向传播、损失计算、参数更新和指标计算。上述计算均由 `code/numpy_nn/` 与 NumPy 完成。该依赖边界可通过以下命令复核：

```powershell
rg -n "sklearn" units/task1_numpy_nn_framework/code
```

Digits 共 1797 张 $8\times8$ 图像，规模足以运行完整数据训练，同时适合在合理时间内完成逐元素数值梯度检查和 naive/vectorized 卷积效率比较。数据由库直接加载到内存，不在仓库生成具体数据文件。
