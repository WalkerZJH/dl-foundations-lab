# model_comparison

| run_id | model | best_val_acc | test_acc | macro_f1 |
| --- | --- | ---: | ---: | ---: |
| baseline_textcnn | textcnn | 0.9192 | 0.9149 | 0.9146 |
| model_mlp | mlp | 0.9198 | 0.9121 | 0.9119 |
| model_lstm | lstm | 0.9191 | 0.9100 | 0.9099 |
| model_bilstm | bilstm | 0.9183 | 0.9163 | 0.9160 |

Selection is based on validation accuracy; test metrics are final observations.