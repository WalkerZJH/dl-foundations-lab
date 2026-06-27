# Task2 实验设计

## 任务目标

在 AG-News 上完成一个轻量、可复现的文本分类实验闭环。实验目标不是堆叠大型预训练模型，而是在 8GB RTX 4060 Laptop 和 24 小时预算内，用受控变量解释模型结构、超参数和关键模块对分类效果的影响。

## 数据协议

使用 Hugging Face `ag_news`。官方 train/test 为 120000/7600；从官方 train 中按 label 分层切出 10% validation，seed=42。最终导出为 `train=108000`、`val=12000`、`test=7600`。所有配置选择基于 validation set，test set 只用于最终观察。

## 核心模块

* tokenizer: lowercase regex tokenizer
* vocab: train split 构建，`vocab_size=30000`，`min_freq=2`
* padding: right padding，`padding_idx=0`
* baseline: TextCNN
* 对比模型：MLP/FN、LSTM、BiLSTM
* 训练目标：cross entropy
* 评价指标：accuracy、macro-F1、per-class report、confusion matrix

## 研究问题

* RQ1: TextCNN baseline 是否能稳定完成 AG-News 分类，并接近 92% 参考准确率？
* RQ2: learning rate、dropout、max sequence length 如何影响收敛、泛化和 train-val gap？
* RQ3: MLP、TextCNN、LSTM、BiLSTM 在效果、成本和稳定性上有什么差异？
* RQ4: TextCNN 的 dropout、kernel size、embedding dimension 是否真正影响泛化？
* RQ5: 在 8GB 显存和 24 小时预算下，哪些实验最具性价比？

## 正式矩阵

* baseline: `baseline_textcnn`
* 超参数：`hparam_lr_3e-4`、`hparam_lr_3e-3`、`hparam_dropout_0.2`、`hparam_dropout_0.7`、`hparam_len_64`、`hparam_len_256`
* 模型对比：`model_mlp`、`baseline_textcnn`、`model_lstm`、`model_bilstm`
* 消融：`ablation_no_dropout`、`ablation_kernel_3_only`、`ablation_kernel_2345`、`ablation_emb_64`、`ablation_emb_256`

## 风险控制

不做完整笛卡尔积 grid search，不把 BERT/DistilBERT 作为必做项，不提交数据、cache、checkpoint 或模型权重。所有 run 保存配置、epoch metrics、日志、曲线和 summary，便于复核。
