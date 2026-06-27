# ablation

| run_id | model | best_val_acc | test_acc | macro_f1 |
| --- | --- | ---: | ---: | ---: |
| baseline_textcnn | textcnn | 0.9192 | 0.9149 | 0.9146 |
| ablation_no_dropout | textcnn | 0.9179 | 0.9137 | 0.9134 |
| ablation_kernel_3_only | textcnn | 0.9173 | 0.9146 | 0.9143 |
| ablation_kernel_2345 | textcnn | 0.9189 | 0.9149 | 0.9148 |
| ablation_emb_64 | textcnn | 0.9157 | 0.9116 | 0.9115 |
| ablation_emb_256 | textcnn | 0.9202 | 0.9109 | 0.9106 |

Selection is based on validation accuracy; test metrics are final observations.