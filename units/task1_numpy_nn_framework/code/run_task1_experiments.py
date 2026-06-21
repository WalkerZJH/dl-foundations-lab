from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path
from typing import Any

from data_loading import describe_splits, load_cifar10_splits, load_digits_splits
from io_utils import ensure_directory, read_json, write_csv, write_json
from numpy_nn import build_model, build_optimizer
from plotting import plot_suite_comparison, plot_training_history
from training import evaluate, train_model


TASK_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = TASK_ROOT / "configs"
RESULT_ROOT = TASK_ROOT / "results"
CHECKPOINT_ROOT = TASK_ROOT.parent / "checkpoints" / TASK_ROOT.name
SUITE_TO_DIRECTORY = {
    "baseline": "baseline",
    "model_comparison": "model_comparison",
    "learning_rate_search": "hparam_tuning",
    "normalization_dropout_ablation": "ablation",
}


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def suite_runs(name: str, baseline: dict[str, Any], suites: dict[str, Any]) -> list[dict[str, Any]]:
    if name == "baseline":
        config = copy.deepcopy(baseline)
        config["run_id"] = "baseline_mlp"
        return [config]
    return [merge_config(baseline, item) for item in suites[name]]


def load_experiment_data(baseline: dict[str, Any]):
    dataset_config = baseline["dataset"]
    dataset_name = str(dataset_config["name"])
    if dataset_name == "sklearn_digits":
        return load_digits_splits(
            seed=int(baseline["seed"]),
            train_ratio=float(dataset_config["train_ratio"]),
            validation_ratio=float(dataset_config["validation_ratio"]),
        )
    if dataset_name == "cifar10":
        repository_root = TASK_ROOT.parents[1]
        dataset_dir = repository_root / str(dataset_config["path"])
        return load_cifar10_splits(
            dataset_dir=dataset_dir,
            seed=int(baseline["seed"]),
            validation_ratio=float(dataset_config["validation_ratio"]),
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def execute_suite(name: str, dataset: str = "digits", force: bool = False) -> dict[str, object]:
    if dataset == "digits":
        dataset_label = "Digits"
        baseline = read_json(CONFIG_ROOT / "baseline.json")
        suites = read_json(CONFIG_ROOT / "experiment_suites.json")
        result_root = RESULT_ROOT
    else:
        dataset_label = "CIFAR-10"
        baseline = read_json(CONFIG_ROOT / "cifar10_baseline.json")
        suites = read_json(CONFIG_ROOT / "cifar10_experiment_suites.json")
        result_root = RESULT_ROOT / "cifar10_full"
    data = load_experiment_data(baseline)
    write_json(result_root / "dataset_summary.json", describe_splits(data))
    output_dir = ensure_directory(result_root / SUITE_TO_DIRECTORY[name])
    rows: list[dict[str, object]] = []
    configs = suite_runs(name, baseline, suites)
    for config in configs:
        run_id = str(config["run_id"])
        run_dir = output_dir / run_id
        checkpoint_dir = CHECKPOINT_ROOT / dataset / SUITE_TO_DIRECTORY[name] / run_id
        if force and run_dir.exists():
            shutil.rmtree(run_dir)
        if force and checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        ensure_directory(run_dir)
        write_json(run_dir / "config.json", config)
        model_seed = int(config["seed"])
        model = build_model(config["model"], model_seed)
        optimizer = build_optimizer(
            str(config["training"]["optimizer"]),
            model.named_parameters(),
            learning_rate=float(config["training"]["learning_rate"]),
        )
        result = train_model(
            model,
            optimizer,
            data.train,
            data.validation,
            config,
            run_dir,
            checkpoint_dir / "training_checkpoint.pkl",
            seed=model_seed + 1,
            resume=not force,
        )
        best_row = result.history[result.best_epoch - 1]
        row: dict[str, object] = {
            "run_id": run_id,
            "model": config["model"]["name"],
            "batchnorm": config["model"].get("batchnorm", False),
            "dropout_keep_ratio": config["model"].get("dropout_keep_ratio", 1.0),
            "learning_rate": config["training"]["learning_rate"],
            "best_epoch": result.best_epoch,
            "train_accuracy_at_selection": best_row["train_accuracy"],
            "best_validation_accuracy": result.best_validation_accuracy,
            "validation_loss_at_selection": best_row["validation_loss"],
            "test_accuracy": "",
            "test_loss": "",
            "selected_by_validation": False,
            "runtime_seconds": result.runtime_seconds,
            "parameter_count": model.parameter_count(),
        }
        rows.append(row)
        plot_training_history(
            result.history,
            run_dir / "training_curves.png",
            dataset_label=dataset_label,
            run_label=run_id,
        )

    selected = max(rows, key=lambda item: float(item["best_validation_accuracy"]))
    selected_index = next(index for index, row in enumerate(rows) if row["run_id"] == selected["run_id"])
    selected_config = configs[selected_index]
    selected_seed = int(selected_config["seed"])
    selected_model = build_model(selected_config["model"], selected_seed)
    selected_optimizer = build_optimizer(
        str(selected_config["training"]["optimizer"]),
        selected_model.named_parameters(),
        learning_rate=float(selected_config["training"]["learning_rate"]),
    )
    train_model(
        selected_model,
        selected_optimizer,
        data.train,
        data.validation,
        selected_config,
        output_dir / str(selected["run_id"]),
        CHECKPOINT_ROOT
        / dataset
        / SUITE_TO_DIRECTORY[name]
        / str(selected["run_id"])
        / "training_checkpoint.pkl",
        seed=selected_seed + 1,
        resume=True,
    )
    test_metrics = evaluate(selected_model, data.test)
    selected["test_accuracy"] = test_metrics["accuracy"]
    selected["test_loss"] = test_metrics["loss"]
    selected["selected_by_validation"] = True
    write_json(
        output_dir / "selected_test_observation.json",
        {
            "selected_run_id": selected["run_id"],
            "selection_metric": "validation_accuracy",
            "best_validation_accuracy": selected["best_validation_accuracy"],
            "test_metrics": test_metrics,
        },
    )
    write_csv(output_dir / "metrics.csv", rows)
    write_json(output_dir / "configs.json", configs)
    write_json(
        output_dir / "suite_summary.json",
        {
            "suite": name,
            "selected_run_id": selected["run_id"],
            "selection_metric": "validation_accuracy",
            "test_evaluated_after_selection": True,
            "runs": rows,
        },
    )
    plot_suite_comparison(
        rows,
        output_dir / "validation_accuracy_comparison.png",
        dataset_label=dataset_label,
    )
    return {"suite": name, "selected_run_id": selected["run_id"], "runs": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Task1 formal validation and experiment suites.")
    choices = [*SUITE_TO_DIRECTORY, "all"]
    parser.add_argument("--suite", choices=choices, default="all")
    parser.add_argument("--dataset", choices=["digits", "cifar10"], default="digits")
    parser.add_argument("--force", action="store_true", help="Restart selected training suites.")
    args = parser.parse_args()
    ensure_directory(RESULT_ROOT)
    selected_suites = list(SUITE_TO_DIRECTORY) if args.suite == "all" else [args.suite]
    for suite in selected_suites:
        summary = execute_suite(suite, dataset=args.dataset, force=args.force)
        print(f"suite complete: {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
