from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_history(
    history: list[dict[str, float | int]],
    output_path: Path,
    dataset_label: str,
    run_label: str,
) -> None:
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="validation")
    axes[0].set(title="Cross-entropy Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="train")
    axes[1].plot(epochs, [row["validation_accuracy"] for row in history], label="validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0.0, 1.0))
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    display_run = run_label.replace("_", " ")
    figure.suptitle(f"{dataset_label} - {display_run} Training Curves")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_suite_comparison(
    rows: list[dict[str, object]],
    output_path: Path,
    dataset_label: str,
) -> None:
    labels = [str(row["run_id"]) for row in rows]
    values = [float(row["best_validation_accuracy"]) for row in rows]
    figure, axis = plt.subplots(figsize=(max(6.0, len(labels) * 1.8), 4.2))
    bars = axis.bar(labels, values, color="#2878B5")
    axis.set(
        title=f"{dataset_label} - Best Validation Accuracy",
        ylabel="Best validation accuracy",
        ylim=(0.0, 1.0),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=15)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.4f}", ha="center")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
