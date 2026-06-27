# Task2 方法补充

## AG-News 文本分类

Tokenization 使用 lowercase regex tokenizer，词表只从 train split 构建。Embedding 将 token id 映射为可训练向量；`padding_idx=0` 避免 padding token 影响有效表示。

MLP/FN 使用 masked mean pooling：

```text
z = sum(mask_i x_i) / sum(mask_i)
logits = W2 ReLU(W1 z + b1) + b2
```

TextCNN 使用多 kernel 一维卷积和 global max pooling：

```text
h_k = ReLU(Conv1d_k(X))
z_k = max_pool(h_k)
logits = W concat(z_k) + b
```

BiLSTM-Attention 在 BiLSTM 输出上学习 token 权重，用于聚合关键位置。RCNN 将 BiLSTM 上下文和 embedding 拼接后 max pooling。TransformerEncoder 使用自注意力，但从零训练时未超过 TextCNN，说明预训练知识比单纯自注意力结构更关键。

DistilBERT fine-tuning 使用预训练语言模型编码文本，再通过分类头输出四类 logits。本轮 DistilBERT 是 AG-News 最优模型。

## 评价协议

所有配置选择基于 validation accuracy。test accuracy、macro-F1、confusion matrix 和 per-class metrics 只用于最终观察和分析，不用于反向调参。
