# 关键实现笔记

* label 使用 Hugging Face `ag_news` 的 0-based 编码，可直接输入 `torch.nn.CrossEntropyLoss`。
* `<pad>` 固定为 0，`<unk>` 固定为 1；所有模型都以 `padding_idx=0` 创建 embedding。
* TextCNN 对 embedding 后的序列做 `Conv1d -> ReLU -> global max pooling`，不同 kernel 的结果拼接后分类。
* MLP 使用 masked mean pooling，避免 padding token 影响句向量。
* LSTM/BiLSTM 使用 `pack_padded_sequence`，不把 padding 后的最后位置误当成有效状态。
* dropout 的 train/eval 行为通过 correctness check 验证：train 下同一输入两次输出不同，eval 下两次输出一致。
* validation accuracy 是唯一配置选择依据；每个 run 的 test metrics 只是最终观察。
