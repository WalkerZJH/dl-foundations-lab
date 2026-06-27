from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parents[2]
RESULTS_DIR = ROOT / "units" / "task2_classification" / "results"
DATA_DIR = ROOT / "units" / "data" / "ag_news"


def load_split(data_dir: Path, split: str) -> pd.DataFrame:
    return pd.read_csv(data_dir / f"{split}.csv")


def make_loader(tokenizer, frame: pd.DataFrame, max_len: int, batch_size: int, shuffle: bool) -> DataLoader:
    encoded = tokenizer(
        frame["text"].astype(str).tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    labels = torch.tensor(frame["label"].astype(int).tolist(), dtype=torch.long)
    ds = TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool) -> dict:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    labels: list[int] = []
    preds: list[int] = []
    probs_out: list[list[float]] = []
    for input_ids, attention_mask, y in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        y = y.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=y)
        logits = out.logits
        pred = logits.argmax(dim=1)
        probs = torch.softmax(logits, dim=1)
        total_loss += float(out.loss.detach().cpu()) * y.size(0)
        total += y.size(0)
        correct += int((pred == y).sum().item())
        labels.extend(y.cpu().tolist())
        preds.extend(pred.cpu().tolist())
        probs_out.extend(probs.cpu().tolist())
    return {
        "loss": total_loss / total,
        "acc": correct / total,
        "macro_f1": f1_score(labels, preds, average="macro"),
        "labels": labels,
        "preds": preds,
        "probs": probs_out,
    }


def run_distilbert(data_dir: Path, out_dir: Path, epochs: int, batch_size: int, lr: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    failure_path = out_dir / "failure.json"
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=4
        ).to(device)
        train_df = load_split(data_dir, "train")
        val_df = load_split(data_dir, "val")
        test_df = load_split(data_dir, "test")
        train_loader = make_loader(tokenizer, train_df, 128, batch_size, True)
        val_loader = make_loader(tokenizer, val_df, 128, batch_size * 2, False)
        test_loader = make_loader(tokenizer, test_df, 128, batch_size * 2, False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = epochs * len(train_loader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        best = {"val_acc": -1.0, "epoch": 0, "state": None}
        metrics = []
        for epoch in range(1, epochs + 1):
            model.train()
            start = time.perf_counter()
            total_loss = 0.0
            total = 0
            correct = 0
            for input_ids, attention_mask, y in train_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                y = y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=y)
                scaler.scale(out.loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                pred = out.logits.detach().argmax(dim=1)
                total_loss += float(out.loss.detach().cpu()) * y.size(0)
                total += y.size(0)
                correct += int((pred == y).sum().item())
            val = evaluate(model, val_loader, device, use_amp=True)
            row = {
                "epoch": epoch,
                "train_loss": total_loss / total,
                "train_acc": correct / total,
                "val_loss": val["loss"],
                "val_acc": val["acc"],
                "val_macro_f1": val["macro_f1"],
                "lr": scheduler.get_last_lr()[0],
                "epoch_time_sec": time.perf_counter() - start,
            }
            metrics.append(row)
            if row["val_acc"] > best["val_acc"]:
                best = {
                    "val_acc": row["val_acc"],
                    "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                }
        if best["state"] is not None:
            model.load_state_dict(best["state"])
            model.to(device)
        test = evaluate(model, test_loader, device, use_amp=True)
        peak_memory = int(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else None
        summary = {
            "run_id": "ag_distilbert_finetune",
            "model": "distilbert-base-uncased",
            "best_epoch": best["epoch"],
            "best_val_acc": best["val_acc"],
            "test_loss": test["loss"],
            "test_acc": test["acc"],
            "test_macro_f1": test["macro_f1"],
            "num_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "peak_gpu_memory_mb": peak_memory,
            "classification_report": classification_report(
                test["labels"], test["preds"], output_dict=True, zero_division=0
            ),
        }
        with (out_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
            writer.writeheader()
            writer.writerows(metrics)
        with (out_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            fieldnames = ["index", "true_label", "pred_label", "prob_0", "prob_1", "prob_2", "prob_3"]
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
        (out_dir / "test_metrics.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out_dir / "confusion_matrix.json").write_text(
            json.dumps(
                {"labels": [0, 1, 2, 3], "matrix": confusion_matrix(test["labels"], test["preds"]).tolist()},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (out_dir / "summary.md").write_text(
            f"# ag_distilbert_finetune\n\nbest_val_acc={summary['best_val_acc']:.4f}; test_acc={summary['test_acc']:.4f}; macro_f1={summary['test_macro_f1']:.4f}.\n",
            encoding="utf-8",
        )
        if failure_path.exists():
            failure_path.unlink()
    except Exception as exc:
        failure_path.write_text(
            json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}, indent=2),
            encoding="utf-8",
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--result-dir", default=str(RESULTS_DIR / "ag_news_pretrained" / "ag_distilbert_finetune"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_distilbert(Path(args.data_dir), Path(args.result_dir), args.epochs, args.batch_size, args.learning_rate)


if __name__ == "__main__":
    main()
