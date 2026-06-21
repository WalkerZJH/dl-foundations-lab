from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data_loading import DatasetSplit
from io_utils import ensure_directory, write_csv, write_json
from numpy_nn.losses import SoftmaxCrossEntropy, softmax_cross_entropy
from numpy_nn.module import Module, copy_state
from numpy_nn.optimizers import Optimizer


@dataclass
class TrainingResult:
    history: list[dict[str, float | int]]
    best_epoch: int
    best_validation_accuracy: float
    runtime_seconds: float


def _config_digest(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _regularization_loss(model: Module, weight_decay: float) -> float:
    if weight_decay == 0.0:
        return 0.0
    return float(
        0.5
        * weight_decay
        * sum(
            np.sum(parameter.data**2)
            for name, parameter in model.named_parameters().items()
            if name.endswith("weight")
        )
    )


def _add_regularization_gradient(model: Module, weight_decay: float) -> None:
    if weight_decay == 0.0:
        return
    for name, parameter in model.named_parameters().items():
        if name.endswith("weight"):
            parameter.grad += weight_decay * parameter.data


def evaluate(model: Module, split: DatasetSplit, batch_size: int = 512) -> dict[str, float]:
    previous_mode = model.training
    model.eval()
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    for start in range(0, split.labels.size, batch_size):
        end = start + batch_size
        logits = model(split.images[start:end])
        losses.append(softmax_cross_entropy(logits, split.labels[start:end]) * logits.shape[0])
        predictions.append(np.argmax(logits, axis=1))
    prediction = np.concatenate(predictions)
    if previous_mode:
        model.train()
    return {
        "loss": float(sum(losses) / split.labels.size),
        "accuracy": float(np.mean(prediction == split.labels)),
    }


def train_model(
    model: Module,
    optimizer: Optimizer,
    train_split: DatasetSplit,
    validation_split: DatasetSplit,
    config: dict[str, Any],
    output_dir: Path,
    seed: int,
    resume: bool = True,
) -> TrainingResult:
    ensure_directory(output_dir)
    checkpoint_path = output_dir / "training_checkpoint.pkl"
    history_path = output_dir / "epoch_metrics.csv"
    trace_path = output_dir / "run_trace.txt"
    digest = _config_digest(config)
    training_config = config.get("training", config)
    epochs = int(training_config["epochs"])
    batch_size = int(training_config["batch_size"])
    weight_decay = float(training_config.get("weight_decay", 0.0))
    rng = np.random.default_rng(seed)
    history: list[dict[str, float | int]] = []
    best_epoch = 0
    best_validation_accuracy = -np.inf
    best_state = copy_state(model.state_dict())
    start_epoch = 1
    elapsed_before = 0.0

    if resume and checkpoint_path.exists():
        with checkpoint_path.open("rb") as handle:
            checkpoint = pickle.load(handle)
        if checkpoint["config_digest"] != digest:
            raise RuntimeError(
                f"Checkpoint config mismatch at {checkpoint_path}; use --force to restart."
            )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        rng.bit_generator.state = checkpoint["rng_state"]
        history = checkpoint["history"]
        best_epoch = checkpoint["best_epoch"]
        best_validation_accuracy = checkpoint["best_validation_accuracy"]
        best_state = checkpoint["best_state"]
        start_epoch = checkpoint["next_epoch"]
        elapsed_before = checkpoint["elapsed_seconds"]
        if checkpoint.get("completed", False):
            model.load_state_dict(best_state)
            return TrainingResult(
                history,
                best_epoch,
                best_validation_accuracy,
                elapsed_before,
            )

    started = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        permutation = rng.permutation(train_split.labels.size)
        batch_loss_sum = 0.0
        sample_count = 0
        for start in range(0, train_split.labels.size, batch_size):
            indices = permutation[start : start + batch_size]
            images = train_split.images[indices]
            labels = train_split.labels[indices]
            model.zero_grad()
            logits = model(images)
            criterion = SoftmaxCrossEntropy()
            loss = criterion.forward(logits, labels) + _regularization_loss(model, weight_decay)
            model.backward(criterion.backward())
            _add_regularization_gradient(model, weight_decay)
            optimizer.step()
            batch_loss_sum += loss * labels.size
            sample_count += labels.size

        train_metrics = evaluate(model, train_split)
        validation_metrics = evaluate(model, validation_split)
        epoch_seconds = time.perf_counter() - epoch_started
        row: dict[str, float | int] = {
            "epoch": epoch,
            "optimization_loss": float(batch_loss_sum / sample_count),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)
        if validation_metrics["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation_metrics["accuracy"]
            best_epoch = epoch
            best_state = copy_state(model.state_dict())

        write_csv(history_path, history)
        elapsed = elapsed_before + time.perf_counter() - started
        checkpoint = {
            "config_digest": digest,
            "model_state": copy_state(model.state_dict()),
            "optimizer_state": optimizer.state_dict(),
            "rng_state": rng.bit_generator.state,
            "history": history,
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_validation_accuracy,
            "best_state": best_state,
            "next_epoch": epoch + 1,
            "elapsed_seconds": elapsed,
            "completed": False,
        }
        with checkpoint_path.open("wb") as handle:
            pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)
        trace_line = (
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={validation_metrics['loss']:.6f} "
            f"val_acc={validation_metrics['accuracy']:.4f} seconds={epoch_seconds:.3f}"
        )
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(trace_line + "\n")
        print(trace_line, flush=True)

    runtime_seconds = elapsed_before + time.perf_counter() - started
    model.load_state_dict(best_state)
    checkpoint.update(
        {
            "model_state": copy_state(model.state_dict()),
            "next_epoch": epochs + 1,
            "elapsed_seconds": runtime_seconds,
            "completed": True,
        }
    )
    with checkpoint_path.open("wb") as handle:
        pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(
        output_dir / "training_summary.json",
        {
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_validation_accuracy,
            "runtime_seconds": runtime_seconds,
            "selection_metric": "validation_accuracy",
        },
    )
    return TrainingResult(history, best_epoch, best_validation_accuracy, runtime_seconds)
