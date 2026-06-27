from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "units" / "task2_classification" / "results"
FIG = RESULTS / "figures" / "stage2"


def load_rows() -> list[dict]:
    rows: list[dict] = []
    summary_path = RESULTS / "ag_news_stage2_summary.csv"
    if summary_path.exists():
        rows.extend(csv.DictReader(summary_path.open(encoding="utf-8")))
    for path in (RESULTS / "ag_news_ensemble").glob("*/test_metrics.json"):
        item = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "suite": "ag_news_ensemble",
                "run_id": item["run_id"],
                "model": item["model"],
                "best_epoch": item.get("best_epoch", 0),
                "best_val_acc": item["best_val_acc"],
                "test_acc": item["test_acc"],
                "test_macro_f1": item["test_macro_f1"],
                "num_parameters": item["num_parameters"],
                "peak_gpu_memory_mb": item["peak_gpu_memory_mb"],
            }
        )
    for row in rows:
        for key in ["best_val_acc", "test_acc", "test_macro_f1", "num_parameters", "peak_gpu_memory_mb"]:
            row[key] = float(row[key])
        row["best_epoch"] = int(float(row["best_epoch"]))
    return rows


def save_confusion_and_class_figures() -> None:
    labels = ["World", "Sports", "Business", "Sci/Tech"]
    best_cm_path = RESULTS / "ag_news_pretrained" / "ag_distilbert_finetune" / "confusion_matrix.json"
    cm = np.array(json.loads(best_cm_path.read_text(encoding="utf-8"))["matrix"])
    plt.figure(figsize=(5.8, 4.8))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(4), labels, rotation=25, ha="right")
    plt.yticks(range(4), labels)
    for i in range(4):
        for j in range(4):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("AG-News best confusion matrix")
    plt.tight_layout()
    plt.savefig(FIG / "ag_best_confusion_matrix.png", dpi=160)
    plt.close()

    per_class = pd.read_csv(RESULTS / "ag_news_error_analysis" / "per_class_metrics.csv")
    plt.figure(figsize=(6, 4))
    plt.bar(per_class["class"], per_class["f1-score"])
    plt.ylim(0.88, 1.0)
    plt.ylabel("F1")
    plt.title("AG-News per-class F1")
    plt.tight_layout()
    plt.savefig(FIG / "ag_per_class_f1.png", dpi=160)
    plt.close()

    light = json.loads(
        (RESULTS / "hparam_tuning" / "hparam_dropout_0.2" / "confusion_matrix.json").read_text(encoding="utf-8")
    )["matrix"]
    best = cm.tolist()
    values = [light[2][3] + light[3][2], best[2][3] + best[3][2]]
    plt.figure(figsize=(5, 4))
    plt.bar(["lightweight best", "DistilBERT"], values)
    plt.ylabel("Business <-> Sci/Tech errors")
    plt.tight_layout()
    plt.savefig(FIG / "ag_business_scitech_confusion.png", dpi=160)
    plt.close()


def save_metric_figures(rows: list[dict]) -> None:
    metric_dirs = []
    for base in ["ag_news_strengthened", "ag_news_pretrained", "hparam_tuning", "baseline", "model_comparison", "ablation"]:
        root = RESULTS / base
        if root.exists():
            metric_dirs.extend(path for path in root.glob("*") if (path / "metrics.csv").exists())

    gaps = []
    times = []
    for path in metric_dirs:
        metrics = pd.read_csv(path / "metrics.csv")
        if metrics.empty:
            continue
        best_i = metrics["val_acc"].idxmax()
        gaps.append((path.name, float(metrics.loc[best_i, "train_acc"] - metrics.loc[best_i, "val_acc"])))
        times.append((path.name, float(metrics["epoch_time_sec"].mean()), float(metrics["val_acc"].max())))

    gaps = sorted(gaps, key=lambda item: item[1], reverse=True)[:12]
    names = [item[0] for item in gaps][::-1]
    values = [item[1] for item in gaps][::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(names, values, color="#4c78a8")
    plt.axvline(0.02, color="#59a14f", linestyle="--", linewidth=1)
    plt.axvline(0.05, color="#e15759", linestyle="--", linewidth=1)
    plt.text(0.0205, -0.6, "small gap", color="#59a14f", fontsize=8)
    plt.text(0.0505, -0.6, "moderate gap", color="#e15759", fontsize=8)
    plt.xlim(0, 0.07)
    plt.xlabel("train accuracy - validation accuracy")
    plt.title("AG-News train-validation gap at best validation epoch")
    plt.tight_layout()
    plt.savefig(FIG / "ag_train_val_gap.png", dpi=160)
    plt.close()

    best_metrics = pd.read_csv(RESULTS / "ag_news_pretrained" / "ag_distilbert_finetune" / "metrics.csv")
    plt.figure(figsize=(6, 4))
    plt.plot(best_metrics["epoch"], best_metrics["train_acc"], marker="o", label="train")
    plt.plot(best_metrics["epoch"], best_metrics["val_acc"], marker="o", label="val")
    plt.xticks([1, 2, 3])
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "ag_best_acc_curve.png", dpi=160)
    plt.close()

    selected_scratch = [
        ("baseline_textcnn", "baseline"),
        ("hparam_dropout_0.2", "dropout 0.2"),
        ("model_mlp", "MLP"),
        ("model_bilstm", "BiLSTM"),
        ("ag_textcnn_label_smoothing", "TextCNN+LS"),
        ("ag_bilstm_20ep", "BiLSTM 20ep"),
        ("ag_fasttext_bigram", "FastText-style"),
        ("ag_rcnn", "RCNN"),
        ("ag_transformer_encoder_small", "Transformer scratch"),
    ]
    rows_by_id = {row["run_id"]: row for row in rows}
    offsets = {
        "baseline_textcnn": (8, -18),
        "hparam_dropout_0.2": (8, -2),
        "model_mlp": (8, -14),
        "model_bilstm": (8, 12),
        "ag_textcnn_label_smoothing": (8, 20),
        "ag_bilstm_20ep": (8, 6),
        "ag_fasttext_bigram": (8, 4),
        "ag_rcnn": (8, -12),
        "ag_transformer_encoder_small": (8, -16),
    }
    plt.figure(figsize=(9, 5.2))
    for run_id, label in selected_scratch:
        row = rows_by_id[run_id]
        x = row["num_parameters"] / 1_000_000
        y = row["test_acc"]
        plt.scatter(x, y, color="#4c78a8")
        plt.annotate(label, (x, y), fontsize=7, xytext=offsets[run_id], textcoords="offset points")
    plt.xlabel("trainable parameters (millions)")
    plt.ylabel("test accuracy")
    plt.xlim(3.65, 6.15)
    plt.ylim(0.908, 0.922)
    plt.title("AG-News scratch models: parameter count vs test accuracy")
    plt.tight_layout()
    plt.savefig(FIG / "ag_params_vs_accuracy.png", dpi=160)
    plt.close()


def save_zoomed_report_figures(rows: list[dict]) -> None:
    lightweight = [
        row
        for row in rows
        if row["suite"] in {"baseline", "hparam_tuning", "model_comparison"}
        and row["run_id"] in {"baseline_textcnn", "hparam_dropout_0.2", "model_mlp", "model_bilstm"}
    ]
    lightweight = sorted(lightweight, key=lambda row: row["best_val_acc"])
    plt.figure(figsize=(7, 4))
    y = np.arange(len(lightweight))
    vals = [row["best_val_acc"] for row in lightweight]
    plt.hlines(y, 0.88, vals, color="#bab0ab", linewidth=2)
    plt.scatter(vals, y, color="#4c78a8", zorder=3)
    for yi, row in zip(y, lightweight):
        plt.text(row["best_val_acc"] + 0.0007, yi, f"{row['best_val_acc']:.4f}", va="center", fontsize=8)
    plt.axvline(0.92, color="#e15759", linestyle="--", linewidth=1)
    plt.text(0.9205, len(lightweight) - 0.3, "92% reference", color="#e15759", fontsize=8)
    plt.yticks(y, [row["run_id"] for row in lightweight])
    plt.xlim(0.88, 0.93)
    plt.xlabel("best validation accuracy")
    plt.title("AG-News lightweight validation accuracy, zoomed scale")
    plt.tight_layout()
    plt.savefig(FIG / "hparam_val_acc_comparison.png", dpi=160)
    plt.close()

    stage2 = [
        row
        for row in rows
        if row["suite"] in {"ag_news_strengthened", "ag_news_pretrained", "ag_news_ensemble"}
        and row["run_id"]
        in {
            "ag_textcnn_label_smoothing",
            "ag_bilstm_20ep",
            "ag_fasttext_bigram",
            "ag_transformer_encoder_small",
            "ag_distilbert_finetune",
            "ag_ensemble_top3",
        }
    ]
    stage2 = sorted(stage2, key=lambda row: row["best_val_acc"])
    labels = {
        "ag_textcnn_label_smoothing": "TextCNN+label smoothing",
        "ag_bilstm_20ep": "BiLSTM 20ep",
        "ag_fasttext_bigram": "FastText-style pooling",
        "ag_transformer_encoder_small": "Transformer scratch",
        "ag_distilbert_finetune": "DistilBERT fine-tune",
        "ag_ensemble_top3": "probability ensemble",
    }
    colors = {
        "ag_news_strengthened": "#4c78a8",
        "ag_news_pretrained": "#f58518",
        "ag_news_ensemble": "#54a24b",
    }
    plt.figure(figsize=(8.5, 4.8))
    y = np.arange(len(stage2))
    for yi, row in zip(y, stage2):
        color = colors.get(row["suite"], "#4c78a8")
        plt.barh(yi, row["best_val_acc"] - 0.90, left=0.90, height=0.55, color=color, alpha=0.85)
        plt.scatter(row["test_acc"], yi, marker="D", color="black", s=24, zorder=3)
        plt.text(
            max(row["best_val_acc"], row["test_acc"]) + 0.0007,
            yi,
            f"val {row['best_val_acc']:.4f} / test {row['test_acc']:.4f}",
            va="center",
            fontsize=7,
        )
    plt.axvline(0.92, color="#e15759", linestyle="--", linewidth=1)
    plt.text(0.9205, -0.45, "92% reference", color="#e15759", fontsize=8)
    plt.yticks(y, [labels.get(row["run_id"], row["run_id"]) for row in stage2])
    plt.xlim(0.90, 0.955)
    plt.xlabel("accuracy")
    plt.title("AG-News strengthened model comparison on validation set")
    plt.tight_layout()
    plt.savefig(FIG / "ag_stage2_model_comparison.png", dpi=160)
    plt.close()


def save_stage2_summary(rows: list[dict]) -> None:
    with (RESULTS / "task2_stage2_all_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "suite",
                "run_id",
                "model",
                "best_epoch",
                "best_val_acc",
                "test_acc",
                "test_macro_f1",
                "num_parameters",
                "peak_gpu_memory_mb",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    save_confusion_and_class_figures()
    save_metric_figures(rows)
    save_zoomed_report_figures(rows)
    save_stage2_summary(rows)
    print(f"generated {len(list(FIG.glob('*.png')))} stage2 figures")


if __name__ == "__main__":
    main()
