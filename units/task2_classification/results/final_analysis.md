# Task2 Final Analysis

This file is the latest final analysis for the AG-News Task2 classification workflow.

All configuration selection uses validation accuracy. Test metrics are final observations only.

## AG-News Lightweight Baseline

The initial 15-run lightweight suite remains archived under `baseline/`, `hparam_tuning/`, `model_comparison/`, and `ablation/`.

| run | best_val_acc | test_acc | note |
| --- | ---: | ---: | --- |
| baseline_textcnn | 0.9192 | 0.9149 | TextCNN baseline |
| hparam_dropout_0.2 | 0.9228 | 0.9167 | best lightweight validation result |
| model_mlp | 0.9198 | 0.9121 | strong topic-word signal |
| model_bilstm | 0.9183 | 0.9163 | higher cost, no validation advantage |

The lightweight suite showed that MLP, TextCNN, LSTM, and BiLSTM all reach roughly 91-92% accuracy. The strong MLP result indicates that AG-News contains strong topic-word cues.

## AG-News Strengthened Suite

Stage2 added longer training, scheduler, label smoothing, wide TextCNN, FastText-style pooling, BiLSTM-Attention, RCNN, TransformerEncoder, DistilBERT, and ensemble results.

| run | model | best_val_acc | test_acc | macro_f1 |
| --- | --- | ---: | ---: | ---: |
| ag_distilbert_finetune | DistilBERT | 0.9483 | 0.9466 | 0.9466 |
| ag_ensemble_top3 | probability ensemble | 0.9483 | 0.9392 | 0.9391 |
| ag_textcnn_label_smoothing | TextCNN | 0.9219 | 0.9186 | 0.9183 |
| ag_bilstm_20ep | BiLSTM | 0.9173 | 0.9184 | 0.9184 |
| ag_fasttext_bigram | FastText-style mean pooling | 0.9191 | 0.9154 | 0.9153 |
| ag_rcnn | RCNN | 0.9203 | 0.9117 | 0.9116 |
| ag_transformer_encoder_small | Transformer from scratch | 0.9148 | 0.9128 | 0.9128 |

The best AG-News model is DistilBERT fine-tuning. It crosses the 92% target and reaches 94.66% test accuracy. Scratch strengthened models improved individual observations but did not reliably exceed 92%. The ensemble reaches 93.92% test accuracy, so it is a valid model-complementarity experiment, but it remains below DistilBERT alone and is not selected as the final model.

## AG-News Error Analysis

DistilBERT improves all classes, but Business and Sci/Tech remain the hardest pair.

| class | precision | recall | F1 |
| --- | ---: | ---: | ---: |
| World | 0.9602 | 0.9532 | 0.9567 |
| Sports | 0.9879 | 0.9863 | 0.9871 |
| Business | 0.9252 | 0.9116 | 0.9183 |
| Sci/Tech | 0.9136 | 0.9353 | 0.9243 |

Sports is easiest because sports articles have clearer named entities, event vocabulary, and narrower topics. Business and Sci/Tech mix product launches, companies, markets, software, and hardware, so the semantic boundary remains less clean.

Hard examples are saved in `ag_news_error_analysis/hard_examples.csv`.

## Key Files

* `ag_news_stage2_summary.csv`
* `task2_stage2_all_summary.csv`
* `ag_news_strengthened/stage2_summary.md`
* `ag_news_error_analysis/error_analysis.md`
* `figures/stage2/`
