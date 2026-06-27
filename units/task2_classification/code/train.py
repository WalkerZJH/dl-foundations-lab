from __future__ import annotations

import argparse
import csv
import json
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn

from dataset import make_loaders
from models import build_model
from utils import count_parameters, load_yaml, plot_metrics, save_json, set_seed


def resolve_device(name: str) -> torch.device:
    if name == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader, criterion, device: torch.device, keep_probs: bool = False) -> dict:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    labels: list[int] = []
    preds: list[int] = []
    probs_out: list[list[float]] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        y = batch["label"].to(device)
        logits = model(input_ids, lengths)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)
        if keep_probs:
            probs_out.extend(F.softmax(logits, dim=1).cpu().tolist())
        total_loss += loss.item() * y.size(0)
        total += y.size(0)
        correct += (pred == y).sum().item()
        labels.extend(y.cpu().tolist())
        preds.extend(pred.cpu().tolist())
    return {
        "loss": total_loss / total,
        "acc": correct / total,
        "macro_f1": f1_score(labels, preds, average="macro"),
        "labels": labels,
        "preds": preds,
        "probs": probs_out,
    }


def build_optimizer(config: dict, model: nn.Module):
    name = config.get("optimizer", "adam").lower()
    kwargs = {
        "lr": config["learning_rate"],
        "weight_decay": config.get("weight_decay", 0.0),
    }
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(config: dict, optimizer, total_steps: int):
    name = config.get("scheduler", "none")
    if name in {None, "none"}:
        return None
    warmup_steps = int(total_steps * float(config.get("warmup_ratio", 0.1)))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        if name == "cosine_with_warmup":
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()
        if name == "linear_with_warmup":
            return max(0.0, 1.0 - progress)
        raise ValueError(f"Unsupported scheduler: {name}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_run(config: dict, data_dir: Path, result_dir: Path) -> dict:
    set_seed(config["seed"])
    result_dir.mkdir(parents=True, exist_ok=True)
    save_json(result_dir / "config.json", config)

    vocab_path = result_dir / "vocab.json"
    vocab, loaders = make_loaders(config, data_dir, vocab_path=vocab_path)
    device = resolve_device(config.get("device", "cuda_if_available"))
    model = build_model(config, vocab_size=vocab.size).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config.get("label_smoothing", 0.0)))
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(
        config,
        optimizer,
        total_steps=max(1, config["epochs"] * len(loaders["train"])),
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    best = {"val_acc": -1.0, "epoch": 0, "state": None}
    patience = config.get("early_stopping", {}).get("patience", 3)
    bad_epochs = 0
    metrics: list[dict] = []
    run_log: list[str] = []

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        wall_start = time.perf_counter()
        start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        if start:
            start.record()

        for batch in loaders["train"]:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, lengths)
            loss = criterion(logits, y)
            loss.backward()
            clip_norm = config.get("gradient_clip_norm")
            if clip_norm:
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            pred = logits.argmax(dim=1)
            total_loss += loss.item() * y.size(0)
            total += y.size(0)
            correct += (pred == y).sum().item()

        if end:
            end.record()
            torch.cuda.synchronize()
            epoch_time = start.elapsed_time(end) / 1000.0
        else:
            epoch_time = time.perf_counter() - wall_start

        val = evaluate_model(model, loaders["val"], criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total,
            "train_acc": correct / total,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_macro_f1": val["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time,
        }
        metrics.append(row)
        run_log.append(json.dumps(row, ensure_ascii=False))
        if row["val_acc"] > best["val_acc"]:
            best = {
                "val_acc": row["val_acc"],
                "epoch": epoch,
                "state": deepcopy(model.state_dict()),
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    test = evaluate_model(model, loaders["test"], criterion, device, keep_probs=True)
    report = classification_report(test["labels"], test["preds"], output_dict=True, zero_division=0)
    cm = confusion_matrix(test["labels"], test["preds"]).tolist()
    peak_memory = (
        int(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else None
    )
    summary = {
        "run_id": config["run_id"],
        "model": config["model"],
        "best_epoch": best["epoch"],
        "best_val_acc": best["val_acc"],
        "test_loss": test["loss"],
        "test_acc": test["acc"],
        "test_macro_f1": test["macro_f1"],
        "num_parameters": count_parameters(model),
        "peak_gpu_memory_mb": peak_memory,
        "epochs_ran": len(metrics),
    }

    with (result_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)
    (result_dir / "run_log.txt").write_text("\n".join(run_log) + "\n", encoding="utf-8")
    save_json(result_dir / "test_metrics.json", summary | {"classification_report": report})
    save_json(result_dir / "confusion_matrix.json", {"labels": [0, 1, 2, 3], "matrix": cm})
    with (result_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "index",
            "true_label",
            "pred_label",
            "prob_0",
            "prob_1",
            "prob_2",
            "prob_3",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (label, pred, probs) in enumerate(zip(test["labels"], test["preds"], test["probs"])):
            writer.writerow(
                {
                    "index": idx,
                    "true_label": label,
                    "pred_label": pred,
                    "prob_0": probs[0],
                    "prob_1": probs[1],
                    "prob_2": probs[2],
                    "prob_3": probs[3],
                }
            )
    plot_metrics(metrics, result_dir)
    (result_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# {config['run_id']}",
                "",
                f"- model: {config['model']}",
                f"- best_epoch: {summary['best_epoch']}",
                f"- best_val_acc: {summary['best_val_acc']:.4f}",
                f"- test_acc: {summary['test_acc']:.4f}",
                f"- test_macro_f1: {summary['test_macro_f1']:.4f}",
                f"- num_parameters: {summary['num_parameters']}",
                f"- peak_gpu_memory_mb: {summary['peak_gpu_memory_mb']}",
                "",
                "Test metrics are final observations only; configuration selection uses validation accuracy.",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", default="units/data/ag_news")
    parser.add_argument("--result-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(Path(args.config))
    summary = train_one_run(config, Path(args.data_dir), Path(args.result_dir))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
