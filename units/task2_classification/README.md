# Task2: AG-News text classification

This unit contains the reproducible AG-News classification workflow: data export, tokenization, vocabulary construction, TextCNN baseline, controlled hyperparameter search, MLP/LSTM/BiLSTM comparison, TextCNN ablation, Stage2 strengthened scratch models, DistilBERT fine-tuning, probability ensemble, and error analysis.

## Data

Data export script:

```powershell
D:\anaconda\envs\minimind\python.exe units\data\download_ag_news.py --out-dir units\data\ag_news --cache-dir units\data\.hf_cache --val-ratio 0.1 --seed 42
```

The export protocol is fixed to Hugging Face `ag_news`:

* `train.csv`: 108000
* `val.csv`: 12000, stratified from the official train split with seed 42
* `test.csv`: 7600, final observation only
* labels: `0 World`, `1 Sports`, `2 Business`, `3 Sci/Tech`

Data CSV files, Hugging Face cache, and checkpoints are ignored by `.gitignore`.

## Entry points

* Data export: `units/data/download_ag_news.py`
* Training entry: `units/task2_classification/code/train.py`
* Evaluation reader: `units/task2_classification/code/evaluate.py`
* Core experiment entry: `units/task2_classification/code/run_task2_experiments.py`
* Stage2 AG-News entry: `units/task2_classification/code/stage2_ag_news.py`
* Stage2 DistilBERT entry: `units/task2_classification/code/stage2_distilbert.py`
* Stage2 summary entry: `units/task2_classification/code/summarize_stage2.py`
* Model implementations: `units/task2_classification/code/models/`

Common commands:

```powershell
D:\anaconda\envs\minimind\python.exe -m compileall units\data\download_ag_news.py units\task2_classification\code
D:\anaconda\envs\minimind\python.exe units\task2_classification\code\run_task2_experiments.py --smoke
D:\anaconda\envs\minimind\python.exe units\task2_classification\code\run_task2_experiments.py --correctness
D:\anaconda\envs\minimind\python.exe units\task2_classification\code\run_task2_experiments.py --suite core
D:\anaconda\envs\minimind\python.exe units\task2_classification\code\stage2_ag_news.py --suite ag_news_strengthened --config units\task2_classification\configs\stage2_ag_news.yaml
D:\anaconda\envs\minimind\python.exe units\task2_classification\code\stage2_distilbert.py --epochs 3 --batch-size 32 --learning-rate 2e-5
D:\anaconda\envs\minimind\python.exe units\task2_classification\code\summarize_stage2.py
```

`--suite core` runs the formal lightweight matrix. If interrupted and re-run, completed runs with `test_metrics.json` are skipped.

## Results

Configuration files live in `configs/`, experiment notes live in `experiments/`, and formal results live in `results/`:

* correctness checks: `results/correctness_checks/summary.md`
* baseline: `results/baseline/baseline_textcnn/summary.md`
* hyperparameter search: `results/hparam_tuning/hparam_summary.md`
* model comparison: `results/model_comparison/model_comparison_summary.md`
* ablation: `results/ablation/ablation_summary.md`
* AG-News Stage2 strengthened runs: `results/ag_news_strengthened/`
* AG-News pretrained model: `results/ag_news_pretrained/`
* AG-News ensemble: `results/ag_news_ensemble/`
* AG-News error analysis: `results/ag_news_error_analysis/`
* final analysis: `results/final_analysis.md`
* Stage2 summary table: `results/task2_stage2_all_summary.csv`
* figures: `results/figures/`

The Stage1 AG-News validation-best configuration is `hparam_dropout_0.2`, with validation accuracy 0.9228 and final test observation 0.9167. The Stage2 validation-best configuration is `ag_distilbert_finetune`, with validation accuracy 0.9483 and final test observation 0.9466. The best scratch test observation is `ag_textcnn_label_smoothing`, with test accuracy 0.9186. `ag_ensemble_top3` reaches test accuracy 0.9392 and is kept as a model-complementarity experiment.

## Deliverable

The phase deliverable is `deliverables/week06_task2/`, including the weekly report, method supplement, LaTeX source, and compiled PDF.
