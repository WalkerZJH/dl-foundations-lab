from __future__ import annotations

import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def plot_metrics(metrics: list[dict], out_dir: Path, prefix: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in metrics]
    for field, ylabel, name in [
        ("loss", "Loss", "loss_curve.png"),
        ("acc", "Accuracy", "accuracy_curve.png"),
    ]:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, [row[f"train_{field}"] for row in metrics], marker="o", label="train")
        plt.plot(epochs, [row[f"val_{field}"] for row in metrics], marker="o", label="val")
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}{name}", dpi=160)
        plt.close()
