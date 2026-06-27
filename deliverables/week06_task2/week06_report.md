# Week06 Report: Task2 AG-News 分类

## 本周目标

本周完成 Task2 的 AG-News 文本分类实验闭环，并在初始 lightweight suite 基础上继续强化实验。最终报告聚焦 AG-News：从轻量 scratch 模型、训练策略、预训练语言模型、ensemble 和错误分析五个角度形成完整实验叙事。

## 完成内容

AG-News 已完成：

* lightweight baseline、超参数搜索、模型对比和 TextCNN 消融；
* 20 epoch 训练预算复核；
* TextCNN label smoothing、wide TextCNN、AdamW/cosine；
* FastText-style pooling、BiLSTM-Attention、RCNN、TransformerEncoder；
* DistilBERT fine-tuning；
* probability ensemble；
* confusion matrix、per-class F1 和 hard examples。

## 关键结果

| dataset | selected run | best_val_acc | test_acc | note |
| --- | --- | ---: | ---: | --- |
| AG-News | ag_distilbert_finetune | 0.9483 | 0.9466 | validation-selected best |
| AG-News | ag_textcnn_label_smoothing | 0.9219 | 0.9186 | best scratch test observation |
| AG-News | ag_ensemble_top3 | 0.9483 | 0.9392 | probability-average ensemble |

DistilBERT 明确超过 AG-News 92% 目标。scratch 文本模型多数停留在 91%--92% 区间，说明预训练语言知识是主要增益来源。`ag_ensemble_top3` 已超过 92% 参考线，但等权概率平均低于 DistilBERT 单模，因此作为模型互补性验证而不是最终最优模型。

## 交付材料

* 最新分析：`units/task2_classification/results/final_analysis.md`
* Stage2 汇总：`units/task2_classification/results/task2_stage2_all_summary.csv`
* 图表：`units/task2_classification/results/figures/stage2/`
* 最新 PDF：`deliverables/week06_task2/latex/week06_task2_classification_report.pdf`

## 后续方向

后续可补充 AG-News 多 seed 复核、weighted ensemble 和 Business/Sci-Tech hard examples 的细粒度语义分析。
