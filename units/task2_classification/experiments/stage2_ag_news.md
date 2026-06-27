# Stage2 AG-News Results

## Strengthened Scratch Models

The strengthened suite added 11 runs under `results/ag_news_strengthened/`. The strongest scratch test observation was `ag_textcnn_label_smoothing` with validation accuracy 0.9219 and test accuracy 0.9186. This improved over the first TextCNN baseline test observation but still did not cross 92% test accuracy.

The 20-epoch review showed that longer training alone did not guarantee improvement. The earlier 6-epoch `hparam_dropout_0.2` remains the strongest validation-selected scratch baseline in this run family.

## Pretrained Model

`ag_distilbert_finetune` achieved validation accuracy 0.9483 and test accuracy 0.9466. This is the main AG-News strengthened result and confirms that pretrained language representations improve the Business/Sci-Tech boundary more than simply widening scratch TextCNN.

## Ensemble

`ag_ensemble_top3` averaged probabilities from DistilBERT, TextCNN label smoothing, and RCNN. It reached test accuracy 0.9392, below DistilBERT alone. This suggests that averaging weaker scratch models with a stronger pretrained model diluted the best single model rather than correcting its residual errors.

## Key Conclusion

Scratch models plateaued around 91-92% test accuracy. DistilBERT crossed the target margin and reached 94.66% test accuracy. The improvement is attributed to pretrained language knowledge, not only to deeper architecture.
