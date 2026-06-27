# 实现计划

## 阶段

1. 使用 `units/data/download_ag_news.py` 导出固定 AG-News split。
2. 实现 regex tokenizer、词表构建、padding 和 DataLoader。
3. 实现 MLP、TextCNN、LSTM、BiLSTM。
4. 实现统一训练入口，保存 config、metrics、run log、summary、曲线和 test 观察。
5. 实现 correctness checks：split、label、tokenizer、padding、single batch loss、train/eval mode。
6. 运行 baseline、超参数、模型对比和消融矩阵。
7. 汇总结果、图表、final analysis 和阶段交付报告。

## 选择

TextCNN 作为 baseline，因为 AG-News 文本较短，n-gram 局部模式有较强可解释性，训练成本低，适合 8GB 显存和 24 小时预算。LSTM/BiLSTM 用于观察顺序建模收益；MLP 用于提供最简单的 mean pooling 对照。
