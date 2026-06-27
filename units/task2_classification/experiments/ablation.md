# 模块消融记录

## 设计原则

以 `baseline_textcnn` 为主对照，保持数据 split、tokenizer、词表、优化器和训练上限一致，只改变一个结构或正则因素。

## 消融矩阵

| run_id | 改动 | 目的 |
| --- | --- | --- |
| ablation_no_dropout | dropout=0.0 | 检查 dropout 是否缓解过拟合 |
| ablation_kernel_3_only | kernel_sizes=[3] | 检查单一 3-gram 卷积是否足够 |
| ablation_kernel_2345 | kernel_sizes=[2,3,4,5] | 检查加入 2-gram 是否提升 |
| ablation_emb_64 | embedding_dim=64 | 检查表示容量不足 |
| ablation_emb_256 | embedding_dim=256 | 检查更大 embedding 的收益和成本 |

## 结果摘要

| run_id | best_val_acc | test_acc | 观察 |
| --- | ---: | ---: | --- |
| baseline_textcnn | 0.9192 | 0.9149 | 对照组 |
| ablation_no_dropout | 0.9179 | 0.9137 | 去除 dropout 后略降，说明仍有正则价值 |
| ablation_kernel_3_only | 0.9173 | 0.9146 | 单 kernel 可用但 validation 稍弱 |
| ablation_kernel_2345 | 0.9189 | 0.9149 | 增加 2-gram 未带来稳定收益 |
| ablation_emb_64 | 0.9157 | 0.9116 | embedding 容量偏小 |
| ablation_emb_256 | 0.9202 | 0.9109 | validation 略高但 test 观察较低，且参数和显存明显增加 |

消融结果支持保留多 kernel 与适度 dropout；embedding 扩大到 256 的收益不稳定。
