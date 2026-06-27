from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from train import train_one_run
from utils import load_yaml, save_json


ROOT = CODE_DIR.parents[2]
TASK_DIR = ROOT / "units" / "task2_classification"
CONFIG_DIR = TASK_DIR / "configs"
RESULTS_DIR = TASK_DIR / "results"
DATA_DIR = ROOT / "units" / "data" / "ag_news"
LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def base_config() -> dict:
    cfg = load_yaml(CONFIG_DIR / "baseline.yaml")
    cfg.update(
        {
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "early_stopping": {"monitor": "val_acc", "patience": 5},
        }
    )
    return cfg


def stage2_runs(config_path: Path) -> list[dict]:
    base = base_config()
    spec = load_yaml(config_path)
    runs = []
    for item in spec["runs"]:
        cfg = deepcopy(base)
        cfg.update(item)
        cfg.setdefault("run_id", item["run_id"])
        cfg.setdefault("scheduler", "none")
        cfg.setdefault("label_smoothing", 0.0)
        runs.append(cfg)
    return runs


def run_strengthened(config_path: Path, data_dir: Path) -> None:
    out_root = RESULTS_DIR / "ag_news_strengthened"
    out_root.mkdir(parents=True, exist_ok=True)
    for cfg in stage2_runs(config_path):
        out_dir = out_root / cfg["run_id"]
        if (out_dir / "test_metrics.json").exists():
            print(f"skip existing {cfg['run_id']}")
            continue
        print(f"run {cfg['run_id']}")
        train_one_run(cfg, data_dir, out_dir)
    summarize()


def load_all_rows() -> list[dict]:
    rows: list[dict] = []
    old_path = RESULTS_DIR / "summary_all_runs.csv"
    if old_path.exists():
        for row in csv.DictReader(old_path.open("r", encoding="utf-8")):
            row = dict(row)
            row["suite"] = row.pop("section")
            rows.append(row)
    for metrics_path in (RESULTS_DIR / "ag_news_strengthened").glob("*/test_metrics.json"):
        item = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "suite": "ag_news_strengthened",
                "run_id": item["run_id"],
                "model": item["model"],
                "best_epoch": item["best_epoch"],
                "best_val_acc": item["best_val_acc"],
                "test_acc": item["test_acc"],
                "test_macro_f1": item["test_macro_f1"],
                "num_parameters": item["num_parameters"],
                "peak_gpu_memory_mb": item["peak_gpu_memory_mb"],
            }
        )
    for metrics_path in (RESULTS_DIR / "ag_news_pretrained").glob("*/test_metrics.json"):
        item = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "suite": "ag_news_pretrained",
                "run_id": item["run_id"],
                "model": item["model"],
                "best_epoch": item["best_epoch"],
                "best_val_acc": item["best_val_acc"],
                "test_acc": item["test_acc"],
                "test_macro_f1": item["test_macro_f1"],
                "num_parameters": item["num_parameters"],
                "peak_gpu_memory_mb": item["peak_gpu_memory_mb"],
            }
        )
    for row in rows:
        for key in ["best_epoch", "best_val_acc", "test_acc", "test_macro_f1", "num_parameters"]:
            row[key] = float(row[key]) if key != "best_epoch" else int(float(row[key]))
        if row.get("peak_gpu_memory_mb") in {"", None, "None"}:
            row["peak_gpu_memory_mb"] = 0
        row["peak_gpu_memory_mb"] = float(row["peak_gpu_memory_mb"])
    return rows


def summarize() -> None:
    rows = load_all_rows()
    if not rows:
        return
    out_csv = RESULTS_DIR / "ag_news_stage2_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_figures(rows)
    best = max(rows, key=lambda r: r["best_val_acc"])
    lines = [
        "# AG-News Stage2 Summary",
        "",
        f"Validation-selected best run: `{best['run_id']}` with best_val_acc={best['best_val_acc']:.4f}; test_acc={best['test_acc']:.4f} is final observation only.",
        "",
        "| suite | run_id | model | best_val_acc | test_acc | macro_f1 | params | peak_mem_mb |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda r: r["best_val_acc"], reverse=True):
        lines.append(
            f"| {row['suite']} | {row['run_id']} | {row['model']} | {row['best_val_acc']:.4f} | {row['test_acc']:.4f} | {row['test_macro_f1']:.4f} | {int(row['num_parameters'])} | {row['peak_gpu_memory_mb']:.0f} |"
        )
    (RESULTS_DIR / "ag_news_strengthened" / "stage2_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_figures(rows: list[dict]) -> None:
    fig_dir = RESULTS_DIR / "figures" / "stage2"
    fig_dir.mkdir(parents=True, exist_ok=True)
    selected = sorted(rows, key=lambda r: r["best_val_acc"], reverse=True)[:14]
    plt.figure(figsize=(10, 5))
    plt.bar([r["run_id"] for r in selected], [r["best_val_acc"] for r in selected])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("validation accuracy")
    plt.tight_layout()
    plt.savefig(fig_dir / "ag_stage2_model_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter([r["num_parameters"] for r in rows], [r["test_acc"] for r in rows])
    for r in rows:
        if r["suite"] in {"ag_news_strengthened", "ag_news_pretrained"}:
            plt.annotate(r["run_id"], (r["num_parameters"], r["test_acc"]), fontsize=7)
    plt.xlabel("trainable parameters")
    plt.ylabel("test accuracy")
    plt.tight_layout()
    plt.savefig(fig_dir / "ag_params_vs_accuracy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter([r["peak_gpu_memory_mb"] for r in rows], [r["test_acc"] for r in rows])
    plt.xlabel("peak GPU memory MB")
    plt.ylabel("test accuracy")
    plt.tight_layout()
    plt.savefig(fig_dir / "ag_memory_vs_accuracy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter([r["best_val_acc"] for r in rows], [r["test_acc"] for r in rows])
    plt.xlabel("validation accuracy")
    plt.ylabel("test accuracy")
    plt.tight_layout()
    plt.savefig(fig_dir / "ag_val_test_comparison.png", dpi=160)
    plt.close()

    lightweight_best = max((r for r in rows if r["suite"] not in {"ag_news_strengthened", "ag_news_pretrained"}), key=lambda r: r["best_val_acc"], default=None)
    strengthened_best = max((r for r in rows if r["suite"] == "ag_news_strengthened"), key=lambda r: r["best_val_acc"], default=None)
    pretrained_best = max((r for r in rows if r["suite"] == "ag_news_pretrained"), key=lambda r: r["best_val_acc"], default=None)
    comp = [r for r in [lightweight_best, strengthened_best, pretrained_best] if r]
    if comp:
        plt.figure(figsize=(7, 4))
        plt.bar([r["run_id"] for r in comp], [r["test_acc"] for r in comp])
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("test accuracy")
        plt.tight_layout()
        plt.savefig(fig_dir / "ag_lightweight_strengthened_best_acc.png", dpi=160)
        plt.close()


def make_hard_examples(run_dir: Path) -> None:
    out_dir = RESULTS_DIR / "ag_news_error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    test_rows = pd.read_csv(DATA_DIR / "test.csv")
    pred = pd.read_csv(run_dir / "predictions.csv")
    pred["text"] = test_rows["text"]
    for k in range(4):
        pred[f"prob_{k}"] = pred[f"prob_{k}"].astype(float)
    pred["prob_true"] = pred.apply(lambda r: r[f"prob_{int(r.true_label)}"], axis=1)
    pred["prob_pred"] = pred.apply(lambda r: r[f"prob_{int(r.pred_label)}"], axis=1)
    sorted_probs = pred[[f"prob_{i}" for i in range(4)]].apply(lambda s: sorted(s, reverse=True), axis=1)
    pred["margin"] = sorted_probs.apply(lambda vals: vals[0] - vals[1])
    pred["true_name"] = pred["true_label"].map(LABELS)
    pred["pred_name"] = pred["pred_label"].map(LABELS)
    errors = pred[pred["true_label"] != pred["pred_label"]].copy()
    groups = []
    for true_label, pred_label, n in [(2, 3, 10), (3, 2, 10), (0, 2, 5), (2, 0, 5)]:
        groups.append(
            errors[(errors.true_label == true_label) & (errors.pred_label == pred_label)]
            .sort_values("prob_pred", ascending=False)
            .head(n)
        )
    groups.append(errors.sort_values("prob_pred", ascending=False).head(10))
    groups.append(errors.sort_values("margin", ascending=True).head(10))
    hard = pd.concat(groups).drop_duplicates(subset=["index"]).copy()
    hard.insert(0, "split", "test")
    hard[
        [
            "split",
            "index",
            "text",
            "true_label",
            "pred_label",
            "true_name",
            "pred_name",
            "prob_true",
            "prob_pred",
            "margin",
        ]
    ].to_csv(out_dir / "hard_examples.csv", index=False)

    report = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))[
        "classification_report"
    ]
    per_class = []
    for label, name in LABELS.items():
        item = report[str(label)]
        per_class.append({"label": label, "class": name, **item})
    pd.DataFrame(per_class).to_csv(out_dir / "per_class_metrics.csv", index=False)
    lines = [
        "# AG-News Error Analysis",
        "",
        f"Analyzed run: `{run_dir.name}`.",
        "",
        "Main confusion in the lightweight best model was Business <-> Sci/Tech. Stage2 keeps this as the key qualitative inspection target.",
        "",
        "Hard examples are saved in `hard_examples.csv`.",
    ]
    (out_dir / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def run_error_analysis() -> None:
    rows = load_all_rows()
    best = max(rows, key=lambda r: r["best_val_acc"])
    candidates = [
        RESULTS_DIR / "ag_news_strengthened" / best["run_id"],
        RESULTS_DIR / "ag_news_pretrained" / best["run_id"],
        RESULTS_DIR / best["suite"] / best["run_id"],
    ]
    for run_dir in candidates:
        if (run_dir / "predictions.csv").exists():
            make_hard_examples(run_dir)
            print(f"error analysis for {run_dir}")
            return
    raise FileNotFoundError(f"No predictions.csv found for best run {best['run_id']}")


def run_ensemble() -> None:
    rows = load_all_rows()
    pred_dirs = []
    for row in sorted(rows, key=lambda r: r["best_val_acc"], reverse=True):
        for root in ["ag_news_pretrained", "ag_news_strengthened", row["suite"]]:
            path = RESULTS_DIR / root / row["run_id"] / "predictions.csv"
            if path.exists():
                pred_dirs.append((row, path))
                break
        if len(pred_dirs) >= 3:
            break
    if len(pred_dirs) < 2:
        raise RuntimeError("Need at least two prediction files for ensemble.")
    frames = [pd.read_csv(path) for _, path in pred_dirs]
    labels = frames[0]["true_label"].to_numpy()
    avg = sum(frame[[f"prob_{i}" for i in range(4)]].to_numpy() for frame in frames) / len(frames)
    preds = avg.argmax(axis=1)
    acc = float((preds == labels).mean())
    from sklearn.metrics import f1_score, classification_report, confusion_matrix

    macro_f1 = float(f1_score(labels, preds, average="macro"))
    out_dir = RESULTS_DIR / "ag_news_ensemble" / "ag_ensemble_top3"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "index": range(len(labels)),
            "true_label": labels,
            "pred_label": preds,
            "prob_0": avg[:, 0],
            "prob_1": avg[:, 1],
            "prob_2": avg[:, 2],
            "prob_3": avg[:, 3],
        }
    ).to_csv(out_dir / "predictions.csv", index=False)
    summary = {
        "run_id": "ag_ensemble_top3",
        "model": "probability_average",
        "members": [row["run_id"] for row, _ in pred_dirs],
        "best_epoch": 0,
        "best_val_acc": max(row["best_val_acc"] for row, _ in pred_dirs),
        "test_acc": acc,
        "test_macro_f1": macro_f1,
        "num_parameters": sum(row["num_parameters"] for row, _ in pred_dirs),
        "peak_gpu_memory_mb": max(row["peak_gpu_memory_mb"] for row, _ in pred_dirs),
        "classification_report": classification_report(labels, preds, output_dict=True, zero_division=0),
    }
    save_json(out_dir / "test_metrics.json", summary)
    save_json(out_dir / "confusion_matrix.json", {"labels": [0, 1, 2, 3], "matrix": confusion_matrix(labels, preds).tolist()})
    (out_dir / "summary.md").write_text(
        f"# ag_ensemble_top3\n\nMembers: {', '.join(summary['members'])}\n\nTest accuracy: {acc:.4f}; macro-F1: {macro_f1:.4f}.\n",
        encoding="utf-8",
    )
    summarize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["ag_news_strengthened"])
    parser.add_argument("--config", default=str(CONFIG_DIR / "stage2_ag_news.yaml"))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--error-analysis", action="store_true")
    parser.add_argument("--ensemble", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.suite == "ag_news_strengthened":
        run_strengthened(Path(args.config), Path(args.data_dir))
    if args.summarize:
        summarize()
    if args.error_analysis:
        run_error_analysis()
    if args.ensemble:
        run_ensemble()


if __name__ == "__main__":
    main()
