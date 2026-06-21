# 共享 Checkpoint 目录

本目录统一存放可恢复训练状态，与正式结果分离。各单元应使用：

```text
units/checkpoints/<unit>/<dataset>/<suite>/<run>/
```

checkpoint 只用于中断恢复。除本说明外，本目录内容均由 `.gitignore` 排除。
