# hparam_tuning

| run_id | model | best_val_acc | test_acc | macro_f1 |
| --- | --- | ---: | ---: | ---: |
| baseline_textcnn | textcnn | 0.9192 | 0.9149 | 0.9146 |
| hparam_lr_3e-4 | textcnn | 0.8958 | 0.8899 | 0.8895 |
| hparam_lr_3e-3 | textcnn | 0.9177 | 0.9128 | 0.9127 |
| hparam_dropout_0.2 | textcnn | 0.9227 | 0.9167 | 0.9165 |
| hparam_dropout_0.7 | textcnn | 0.9153 | 0.9120 | 0.9118 |
| hparam_len_64 | textcnn | 0.9176 | 0.9128 | 0.9124 |
| hparam_len_256 | textcnn | 0.9187 | 0.9163 | 0.9160 |

Selection is based on validation accuracy; test metrics are final observations.