# AG-News Stage2 Summary

Validation-selected best run: `ag_distilbert_finetune` with best_val_acc=0.9483; test_acc=0.9466 is final observation only.

| suite | run_id | model | best_val_acc | test_acc | macro_f1 | params | peak_mem_mb |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ag_news_pretrained | ag_distilbert_finetune | distilbert-base-uncased | 0.9483 | 0.9466 | 0.9466 | 66956548 | 1713 |
| hparam_tuning | hparam_dropout_0.2 | textcnn | 0.9227 | 0.9167 | 0.9165 | 4038532 | 162 |
| ag_news_strengthened | ag_textcnn_label_smoothing | textcnn | 0.9219 | 0.9186 | 0.9183 | 4038532 | 162 |
| ag_news_strengthened | ag_rcnn | rcnn | 0.9203 | 0.9117 | 0.9116 | 4203780 | 289 |
| ablation | ablation_emb_256 | textcnn | 0.9202 | 0.9109 | 0.9106 | 8075140 | 258 |
| model_comparison | model_mlp | mlp | 0.9198 | 0.9121 | 0.9119 | 3874052 | 107 |
| ag_news_strengthened | ag_bilstm_attention | bilstm_attention | 0.9197 | 0.9138 | 0.9138 | 4105477 | 252 |
| baseline | baseline_textcnn | textcnn | 0.9192 | 0.9149 | 0.9146 | 4038532 | 162 |
| model_comparison | model_lstm | lstm | 0.9191 | 0.9100 | 0.9099 | 3972612 | 130 |
| ag_news_strengthened | ag_fasttext_bigram | fasttext | 0.9191 | 0.9154 | 0.9153 | 6000804 | 158 |
| ablation | ablation_kernel_2345 | textcnn | 0.9189 | 0.9149 | 0.9148 | 4071940 | 171 |
| hparam_tuning | hparam_len_256 | textcnn | 0.9187 | 0.9163 | 0.9160 | 4038532 | 208 |
| model_comparison | model_bilstm | bilstm | 0.9183 | 0.9163 | 0.9160 | 4105220 | 191 |
| ablation | ablation_no_dropout | textcnn | 0.9179 | 0.9137 | 0.9134 | 4038532 | 162 |
| ag_news_strengthened | ag_textcnn_wide | textcnn | 0.9179 | 0.9142 | 0.9141 | 6052356 | 334 |
| hparam_tuning | hparam_lr_3e-3 | textcnn | 0.9177 | 0.9128 | 0.9127 | 4038532 | 162 |
| hparam_tuning | hparam_len_64 | textcnn | 0.9176 | 0.9128 | 0.9124 | 4038532 | 130 |
| ablation | ablation_kernel_3_only | textcnn | 0.9173 | 0.9146 | 0.9143 | 3889796 | 135 |
| ag_news_strengthened | ag_bilstm_20ep | bilstm | 0.9173 | 0.9184 | 0.9184 | 4105220 | 193 |
| ag_news_strengthened | ag_textcnn_adamw_cosine | textcnn | 0.9159 | 0.9113 | 0.9111 | 4038532 | 162 |
| ablation | ablation_emb_64 | textcnn | 0.9157 | 0.9116 | 0.9115 | 2020228 | 100 |
| hparam_tuning | hparam_dropout_0.7 | textcnn | 0.9153 | 0.9120 | 0.9118 | 4038532 | 162 |
| ag_news_strengthened | ag_transformer_encoder_small | transformer_encoder | 0.9148 | 0.9128 | 0.9128 | 5985540 | 798 |
| ag_news_strengthened | ag_textcnn_d02_len256_20ep | textcnn | 0.9147 | 0.9093 | 0.9095 | 4038532 | 208 |
| ag_news_strengthened | ag_textcnn_baseline_20ep | textcnn | 0.9144 | 0.9072 | 0.9071 | 4038532 | 162 |
| ag_news_strengthened | ag_textcnn_d02_20ep | textcnn | 0.9141 | 0.9103 | 0.9103 | 4038532 | 162 |
| hparam_tuning | hparam_lr_3e-4 | textcnn | 0.8958 | 0.8899 | 0.8895 | 4038532 | 162 |