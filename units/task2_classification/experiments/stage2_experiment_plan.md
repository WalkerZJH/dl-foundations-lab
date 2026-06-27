# Task2 Stage2 Experiment Plan

## Goal

Stage2 continues from the AG-News lightweight baseline suite and strengthens the text-classification experiment story without adding an unbounded grid search.

## Required Stage2 Work

AG-News:

* longer training budget review: 20-epoch TextCNN/BiLSTM variants;
* stronger scratch models: TextCNN + AdamW/cosine, wide TextCNN, label smoothing, FastText-style pooling, BiLSTM-Attention, RCNN, TransformerEncoder;
* pretrained model: DistilBERT fine-tuning;
* ensemble: probability averaging of top models;
* error analysis: confusion matrix, per-class F1, hard examples.

## Selection Rule

Validation accuracy selects configurations. Test accuracy is reported only as final observation.
