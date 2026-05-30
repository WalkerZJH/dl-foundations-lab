# Assignment 2 代码目录

本目录保存 CS231n Assignment 2 的可运行代码。

## 入口

* `run_assignment2_experiments.py`：统一实验脚本，负责实现自检、PyTorch/RNN smoke test、CIFAR-10 实验、指标保存和可视化生成。
* `cs231n/`：核心 Python 包，包含 layers、optim、solver、分类器、CNN、RNN/LSTM captioning 等实现。

## 运行方式

当前 PyTorch 依赖使用本机 conda 环境 `minimind`：

```powershell
conda run -n minimind python units\assignment2\code\run_assignment2_experiments.py
```

数据下载脚本保留在 `cs231n/datasets/`。下载后的 CIFAR-10、COCO、ImageNet 样例等原始数据不提交。
