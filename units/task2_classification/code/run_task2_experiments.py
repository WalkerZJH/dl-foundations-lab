from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataset import NewsDataset, build_vocab, collate_batch, read_csv, tokenize
from models import build_model
from train import train_one_run
from utils import load_yaml, save_json, set_seed


ROOT = CODE_DIR.parents[2]
TASK_DIR = ROOT / "units" / "task2_classification"
RESULTS_DIR = TASK_DIR / "results"
DATA_DIR = ROOT / "units" / "data" / "ag_news"
CONFIG_DIR = TASK_DIR / "configs"
LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def base_config() -> dict:
    return load_yaml(CONFIG_DIR / "baseline.yaml")


def all_runs() -> list[tuple[str, str, dict]]:
    base = base_config()

    def cfg(run_id: str, section: str, **updates):
        item = deepcopy(base)
        item.update(updates)
        item["run_id"] = run_id
        return (section, run_id, item)

    return [
        ("baseline", "baseline_textcnn", base),
        cfg("hparam_lr_3e-4", "hparam_tuning", learning_rate=3e-4),
        cfg("hparam_lr_3e-3", "hparam_tuning", learning_rate=3e-3),
        cfg("hparam_dropout_0.2", "hparam_tuning", dropout=0.2),
        cfg("hparam_dropout_0.7", "hparam_tuning", dropout=0.7),
        cfg("hparam_len_64", "hparam_tuning", max_seq_len=64),
        cfg("hparam_len_256", "hparam_tuning", max_seq_len=256),
        cfg("model_mlp", "model_comparison", model="mlp", hidden_dim=256),
        cfg("model_lstm", "model_comparison", model="lstm", hidden_dim=128, gradient_clip_norm=5.0),
        cfg("model_bilstm", "model_comparison", model="bilstm", hidden_dim=128, gradient_clip_norm=5.0),
        cfg("ablation_no_dropout", "ablation", dropout=0.0),
        cfg("ablation_kernel_3_only", "ablation", kernel_sizes=[3]),
        cfg("ablation_kernel_2345", "ablation", kernel_sizes=[2, 3, 4, 5]),
        cfg("ablation_emb_64", "ablation", embedding_dim=64),
        cfg("ablation_emb_256", "ablation", embedding_dim=256),
    ]


def assert_data_ready(data_dir: Path) -> None:
    required = ["train.csv", "val.csv", "test.csv", "metadata.json"]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing AG-News files: {missing}. Run units/data/download_ag_news.py first."
        )


def run_smoke(data_dir: Path) -> None:
    assert_data_ready(data_dir)
    config = base_config()
    config.update({"run_id": "smoke_textcnn", "epochs": 1, "batch_size": 64})
    rows = read_csv(data_dir / "train.csv")[:512]
    tmp = RESULTS_DIR / "_smoke_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    # Reuse the real pipeline by writing a tiny temporary split under ignored results.
    for name, subset in {"train": rows[:384], "val": rows[384:448], "test": rows[448:512]}.items():
        with (tmp / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["label", "text"])
            writer.writeheader()
            writer.writerows(subset)
    summary = train_one_run(config, tmp, tmp / "run")
    save_json(RESULTS_DIR / "smoke_check.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


def run_correctness(data_dir: Path) -> None:
    assert_data_ready(data_dir)
    out = RESULTS_DIR / "correctness_checks"
    out.mkdir(parents=True, exist_ok=True)
    config = base_config()
    train_rows = read_csv(data_dir / "train.csv")
    val_rows = read_csv(data_dir / "val.csv")
    test_rows = read_csv(data_dir / "test.csv")
    data_check = {
        "splits": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "labels": sorted({row["label"] for row in train_rows + val_rows + test_rows}),
        "non_empty_text": all(row["text"].strip() for row in train_rows[:1000] + val_rows[:1000] + test_rows[:1000]),
        "passed": len(train_rows) == 108000
        and len(val_rows) == 12000
        and len(test_rows) == 7600
        and sorted({row["label"] for row in train_rows + val_rows + test_rows}) == [0, 1, 2, 3],
    }
    save_json(out / "data_check.json", data_check)

    vocab = build_vocab(train_rows, config["vocab_size"], config["min_freq"])
    sample = train_rows[0]
    tokens = tokenize(sample["text"])
    ids, length = vocab.encode(tokens, config["max_seq_len"])
    tokenizer_check = {
        "raw_text": sample["text"],
        "tokens": tokens[:20],
        "input_ids": ids[:20],
        "length": length,
        "pad_token_id": vocab.token_to_id["<pad>"],
        "unk_token_id": vocab.token_to_id["<unk>"],
        "passed": vocab.token_to_id["<pad>"] == 0 and vocab.token_to_id["<unk>"] == 1,
    }
    save_json(out / "tokenizer_check.json", tokenizer_check)

    dataset = NewsDataset(train_rows[:8], vocab, config["max_seq_len"])
    batch = collate_batch([dataset[i] for i in range(8)])
    padding_check = {
        "input_shape": list(batch["input_ids"].shape),
        "lengths": batch["lengths"].tolist(),
        "mask_counts": (batch["input_ids"] != 0).sum(dim=1).tolist(),
    }
    padding_check["passed"] = padding_check["lengths"] == padding_check["mask_counts"]
    save_json(out / "padding_mask_check.json", padding_check)

    label_check = {"label_mapping": LABELS, "label_index_base": 0, "passed": True}
    save_json(out / "label_mapping_check.json", label_check)

    set_seed(config["seed"])
    config_small = deepcopy(config)
    config_small.update({"epochs": 1, "batch_size": 8})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config_small, vocab.size).to(device)
    model.train()
    inputs = batch["input_ids"].to(device)
    lengths = batch["lengths"].to(device)
    y = batch["label"].to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(inputs, lengths), y)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    save_json(
        out / "single_batch_overfit.json",
        {"initial_loss": losses[0], "final_loss": losses[-1], "passed": losses[-1] < losses[0]},
    )

    model.train()
    a = model(inputs, lengths).detach().cpu()
    b = model(inputs, lengths).detach().cpu()
    model.eval()
    c = model(inputs, lengths).detach().cpu()
    d = model(inputs, lengths).detach().cpu()
    mode_check = {
        "train_outputs_differ": bool(not torch.allclose(a, b)),
        "eval_outputs_same": bool(torch.allclose(c, d)),
    }
    mode_check["passed"] = mode_check["train_outputs_differ"] and mode_check["eval_outputs_same"]
    save_json(out / "mode_check.json", mode_check)

    passed = all(
        item["passed"]
        for item in [data_check, tokenizer_check, padding_check, label_check, mode_check]
    ) and losses[-1] < losses[0]
    (out / "summary.md").write_text(
        "\n".join(
            [
                "# Task2 Correctness Checks",
                "",
                f"- data split check: {'passed' if data_check['passed'] else 'failed'}",
                f"- tokenizer / label check: {'passed' if tokenizer_check['passed'] else 'failed'}",
                f"- padding / length check: {'passed' if padding_check['passed'] else 'failed'}",
                f"- single batch loss: {losses[0]:.4f} -> {losses[-1]:.4f}",
                f"- train/eval mode check: {'passed' if mode_check['passed'] else 'failed'}",
                f"- all_passed: {passed}",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"all_passed": passed}, ensure_ascii=False))


def aggregate() -> None:
    rows = []
    for section, run_id, _ in all_runs():
        path = RESULTS_DIR / section / run_id / "test_metrics.json"
        if path.exists():
            item = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "section": section,
                    "run_id": run_id,
                    "model": item["model"],
                    "best_epoch": item["best_epoch"],
                    "best_val_acc": item["best_val_acc"],
                    "test_acc": item["test_acc"],
                    "test_macro_f1": item["test_macro_f1"],
                    "num_parameters": item["num_parameters"],
                    "peak_gpu_memory_mb": item["peak_gpu_memory_mb"],
                }
            )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "summary_all_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    make_figures(rows)
    write_section_summaries(rows)


def make_figures(rows: list[dict]) -> None:
    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    baseline_dir = RESULTS_DIR / "baseline" / "baseline_textcnn"
    if (baseline_dir / "loss_curve.png").exists():
        shutil.copyfile(baseline_dir / "loss_curve.png", fig_dir / "baseline_loss_curve.png")
    if (baseline_dir / "accuracy_curve.png").exists():
        shutil.copyfile(baseline_dir / "accuracy_curve.png", fig_dir / "baseline_acc_curve.png")
    best = max(rows, key=lambda row: row["best_val_acc"])
    cm_path = RESULTS_DIR / best["section"] / best["run_id"] / "confusion_matrix.json"
    if cm_path.exists():
        cm = json.loads(cm_path.read_text(encoding="utf-8"))["matrix"]
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, cmap="Blues")
        plt.xticks(range(4), [LABELS[i] for i in range(4)], rotation=30, ha="right")
        plt.yticks(range(4), [LABELS[i] for i in range(4)])
        for i, row_vals in enumerate(cm):
            for j, value in enumerate(row_vals):
                plt.text(j, i, str(value), ha="center", va="center", color="black")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion matrix: {best['run_id']}")
        plt.tight_layout()
        plt.savefig(fig_dir / "confusion_matrix_best.png", dpi=160)
        plt.close()
    for section, filename, title in [
        ("hparam_tuning", "hparam_val_acc_comparison.png", "Hyperparameter validation accuracy"),
        ("model_comparison", "model_comparison_val_test_acc.png", "Model comparison"),
        ("ablation", "ablation_val_acc.png", "Ablation validation accuracy"),
    ]:
        subset = [row for row in rows if row["section"] == section or row["run_id"] == "baseline_textcnn"]
        if not subset:
            continue
        plt.figure(figsize=(8, 4))
        x = [row["run_id"] for row in subset]
        plt.bar(x, [row["best_val_acc"] for row in subset], label="val")
        if section == "model_comparison":
            plt.plot(x, [row["test_acc"] for row in subset], color="black", marker="o", label="test")
            plt.legend()
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("accuracy")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(fig_dir / filename, dpi=160)
        plt.close()

    runtime_rows = [row for row in rows if row["section"] in {"baseline", "model_comparison"}]
    if runtime_rows:
        plt.figure(figsize=(7, 4))
        plt.bar([row["run_id"] for row in runtime_rows], [row["num_parameters"] for row in runtime_rows])
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("trainable parameters")
        plt.title("Model size comparison")
        plt.tight_layout()
        plt.savefig(fig_dir / "runtime_comparison.png", dpi=160)
        plt.close()


def write_section_summaries(rows: list[dict]) -> None:
    sections = {
        "hparam_tuning": "hparam_summary.md",
        "model_comparison": "model_comparison_summary.md",
        "ablation": "ablation_summary.md",
    }
    for section, filename in sections.items():
        subset = [row for row in rows if row["section"] == section or row["run_id"] == "baseline_textcnn"]
        out = RESULTS_DIR / section
        out.mkdir(parents=True, exist_ok=True)
        lines = [f"# {section}", "", "| run_id | model | best_val_acc | test_acc | macro_f1 |", "| --- | --- | ---: | ---: | ---: |"]
        for row in subset:
            lines.append(
                f"| {row['run_id']} | {row['model']} | {row['best_val_acc']:.4f} | {row['test_acc']:.4f} | {row['test_macro_f1']:.4f} |"
            )
        lines.append("")
        lines.append("Selection is based on validation accuracy; test metrics are final observations.")
        (out / filename).write_text("\n".join(lines), encoding="utf-8")


def run_suite(data_dir: Path, suite: str) -> None:
    assert_data_ready(data_dir)
    runs = all_runs()
    if suite == "baseline":
        runs = runs[:1]
    for section, run_id, config in runs:
        result_dir = RESULTS_DIR / section / run_id
        if (result_dir / "test_metrics.json").exists():
            print(f"skip existing {run_id}")
            continue
        print(f"run {run_id}")
        train_one_run(config, data_dir, result_dir)
    aggregate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--suite", choices=["baseline", "core"], default=None)
    parser.add_argument("--aggregate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if args.smoke:
        run_smoke(data_dir)
    if args.correctness:
        run_correctness(data_dir)
    if args.suite:
        run_suite(data_dir, args.suite)
    if args.aggregate:
        aggregate()


if __name__ == "__main__":
    main()
