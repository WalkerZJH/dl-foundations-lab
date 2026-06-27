# Week06 Task2 阶段交付

本目录归档 Task2 分类模型设计与实现的阶段成果。最终报告聚焦 AG-News 文本分类：保留完整 lightweight suite、强化 scratch 模型、DistilBERT fine-tuning、probability ensemble 和错误分析。

## 文件

* 周报：`week06_report.md`
* 方法补充：`task2_method_supplement.md`
* LaTeX 源文件：`latex/week06_task2_classification_report.tex`
* 最新 PDF：`latex/week06_task2_classification_report.pdf`

## 结果入口

* 单元 README：`../../units/task2_classification/README.md`
* 最新最终分析：`../../units/task2_classification/results/final_analysis.md`
* Stage2 汇总表：`../../units/task2_classification/results/task2_stage2_all_summary.csv`
* AG-News 强化汇总：`../../units/task2_classification/results/ag_news_strengthened/stage2_summary.md`
* AG-News 错误分析：`../../units/task2_classification/results/ag_news_error_analysis/error_analysis.md`
* Stage2 图表：`../../units/task2_classification/results/figures/stage2/`

## 当前结论

AG-News validation 最优模型为 `ag_distilbert_finetune`，best validation accuracy 为 0.9483，最终 test observation accuracy 为 0.9466。`ag_ensemble_top3` 的 test observation accuracy 为 0.9392，作为模型互补性验证保留。所有配置选择均基于 validation set。
