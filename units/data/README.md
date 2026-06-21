# 共享数据目录

本目录统一存放各 assignment 与 task 使用的本地数据。代码从仓库级 `units/data/` 解析数据路径，避免在不同单元中重复下载和维护副本。

预期结构：

```text
units/data/
├── cifar-10-batches-py/
├── imagenet_val_25.npz
└── coco_captioning/
```

运行本目录中的下载脚本可获取对应数据。原始数据、压缩包和解压产物均由 `.gitignore` 排除，不提交仓库。
