# Stage2 Error Analysis

## Analyzed Model

The hard-example analysis uses `ag_distilbert_finetune`, the validation-selected best AG-News model.

## Per-Class Pattern

| class | precision | recall | F1 |
| --- | ---: | ---: | ---: |
| World | 0.9602 | 0.9532 | 0.9567 |
| Sports | 0.9879 | 0.9863 | 0.9871 |
| Business | 0.9252 | 0.9116 | 0.9183 |
| Sci/Tech | 0.9136 | 0.9353 | 0.9243 |

Sports remains the easiest class. Business and Sci/Tech remain the hardest pair because company, product, market, and technology events overlap semantically.

## Files

* `results/ag_news_error_analysis/per_class_metrics.csv`
* `results/ag_news_error_analysis/hard_examples.csv`
* `results/figures/stage2/ag_best_confusion_matrix.png`
* `results/figures/stage2/ag_business_scitech_confusion.png`
