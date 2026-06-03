import argparse
import csv
import json
import os
import time
from copy import deepcopy
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cs231n.classifiers.cnn import ThreeLayerConvNet
from cs231n.classifiers.fc_net import FullyConnectedNet
from cs231n.data_utils import get_CIFAR10_data
from cs231n.solver import Solver


ASSIGNMENT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_ROOT = os.path.join(ASSIGNMENT_ROOT, "results")
DEFAULT_CONFIG = os.path.join(ASSIGNMENT_ROOT, "experiments", "assignment2_exploration_suites.json")


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_dict(base, override):
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        merged[key] = value
    return merged


def append_trace(out_dir, message):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "run_trace.txt")
    timestamp = datetime.now().isoformat(timespec="seconds")
    with open(path, "a", encoding="utf-8") as f:
        f.write("[%s] %s\n" % (timestamp, message))


def solver_train_sample_count(value, train_size):
    if value is None:
        return None
    return min(value, train_size)


def selected_suite_names(config, suite_name):
    names = sorted(config["suites"].keys())
    if suite_name == "all":
        return names
    if suite_name not in config["suites"]:
        raise ValueError("Unknown suite: %s" % suite_name)
    return [suite_name]


def variant_setting(variant):
    keys = [
        "kind",
        "model",
        "learning_rate",
        "reg",
        "num_filters",
        "filter_size",
        "hidden_dim",
        "hidden_dims",
        "batch_size",
        "epochs",
        "update_rule",
        "normalization",
        "dropout_keep_ratio",
        "weight_scale",
        "data",
    ]
    return ", ".join("%s=%s" % (key, variant[key]) for key in keys if key in variant)


def build_model(variant):
    if variant["model"] == "three_layer_convnet":
        return ThreeLayerConvNet(
            num_filters=variant.get("num_filters", 8),
            filter_size=variant.get("filter_size", 3),
            hidden_dim=variant.get("hidden_dim", 100),
            weight_scale=variant.get("weight_scale", 1e-2),
            reg=variant.get("reg", 1e-3),
        )
    if variant["model"] == "fully_connected":
        return FullyConnectedNet(
            variant.get("hidden_dims", [100, 100]),
            input_dim=3 * 32 * 32,
            num_classes=10,
            normalization=variant.get("normalization"),
            dropout_keep_ratio=variant.get("dropout_keep_ratio", 1.0),
            reg=variant.get("reg", 0.1),
            weight_scale=variant.get("weight_scale", 5e-2),
            seed=123,
        )
    raise ValueError("Unknown model: %s" % variant["model"])


def load_data(data_cfg, cache):
    key = (data_cfg["train_size"], data_cfg["val_size"], data_cfg["test_size"])
    if key not in cache:
        cache[key] = get_CIFAR10_data(
            num_training=data_cfg["train_size"],
            num_validation=data_cfg["val_size"],
            num_test=data_cfg["test_size"],
        )
    return cache[key]


def run_solver_variant(variant, data, solver_base):
    start = time.time()
    solver_cfg = merge_dict(solver_base, variant)
    model = build_model(variant)
    solver = Solver(
        model,
        data,
        update_rule=solver_cfg.get("update_rule", "adam"),
        optim_config={"learning_rate": solver_cfg.get("learning_rate", 1e-3)},
        lr_decay=solver_cfg.get("lr_decay", 0.95),
        num_epochs=solver_cfg.get("epochs", 10),
        batch_size=solver_cfg.get("batch_size", 100),
        num_train_samples=solver_train_sample_count(
            solver_cfg.get("num_train_samples"), data["X_train"].shape[0]
        ),
        num_val_samples=None,
        print_every=100,
        verbose=False,
    )
    solver.train()
    full_train_acc = solver.check_accuracy(
        data["X_train"], data["y_train"], num_samples=None
    )
    test_acc = solver.check_accuracy(data["X_test"], data["y_test"], num_samples=None)
    train_acc = float(full_train_acc)
    history_train_acc = float(solver.train_acc_history[-1])
    val_acc = float(solver.best_val_acc)
    return {
        "id": variant["id"],
        "model": variant["model"],
        "setting": variant_setting(solver_cfg),
        "train_acc": train_acc,
        "history_train_acc": history_train_acc,
        "val_acc": val_acc,
        "test_acc": float(test_acc),
        "train_val_gap": train_acc - val_acc,
        "final_loss": float(solver.loss_history[-1]) if solver.loss_history else "",
        "elapsed_seconds": time.time() - start,
        "loss_history": [float(x) for x in solver.loss_history],
        "train_acc_history": [float(x) for x in solver.train_acc_history],
        "val_acc_history": [float(x) for x in solver.val_acc_history],
        "variant": variant,
    }


def row_from_result(item):
    return {
        "id": item["id"],
        "model": item["model"],
        "setting": item["setting"],
        "train_acc": item["train_acc"],
        "history_train_acc": item.get("history_train_acc", item["train_acc"]),
        "val_acc": item["val_acc"],
        "test_acc": item["test_acc"],
        "train_val_gap": item["train_val_gap"],
        "final_loss": item["final_loss"],
        "elapsed_seconds": item["elapsed_seconds"],
    }


def best_by_val(results):
    return max(results, key=lambda item: item["val_acc"])


def load_resume_results(out_dir, suite_name):
    if not out_dir:
        return []
    summary_path = os.path.join(out_dir, "%s_summary.json" % suite_name)
    if not os.path.exists(summary_path):
        return []
    with open(summary_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("results", [])


def run_suite(config, suite_name, args, out_dir):
    suite = config["suites"][suite_name]
    default_data = merge_dict(config["defaults"]["data"], suite.get("data"))
    cli_data = {
        key: value
        for key, value in {
            "train_size": args.train_size,
            "val_size": args.val_size,
            "test_size": args.test_size,
        }.items()
        if value is not None
    }
    default_data = merge_dict(default_data, cli_data)
    solver_base = merge_dict(config["defaults"]["solver"], suite.get("solver"))
    if args.epochs is not None:
        solver_base["epochs"] = args.epochs
    if args.batch_size is not None:
        solver_base["batch_size"] = args.batch_size

    cache = {}
    results = load_resume_results(out_dir, suite_name)
    completed_ids = {item["id"] for item in results}
    append_trace(
        out_dir,
        "suite started: %s completed=%d" % (suite_name, len(completed_ids)),
    )
    variants = suite["variants"]
    if args.max_variants is not None:
        variants = variants[: args.max_variants]
    for variant in variants:
        if variant["id"] in completed_ids:
            append_trace(out_dir, "skip completed variant %s" % variant["id"])
            continue
        np.random.seed(args.seed)
        data_cfg = merge_dict(default_data, variant.get("data"))
        append_trace(
            out_dir,
            "start variant %s data=%s" % (variant["id"], json.dumps(data_cfg)),
        )
        data = load_data(data_cfg, cache)
        result = run_solver_variant(variant, data, solver_base)
        result["data"] = data_cfg
        results.append(result)
        payload = {
            "suite": suite_name,
            "suite_meta": {key: value for key, value in suite.items() if key != "variants"},
            "default_data": default_data,
            "results": results,
            "selected": row_from_result(best_by_val(results)),
        }
        save_outputs_to_dir(out_dir, payload)
        append_trace(
            out_dir,
            "finish variant %s val_acc=%.4f test_acc=%.4f"
            % (variant["id"], result["val_acc"], result["test_acc"]),
        )

    selected = best_by_val(results)
    append_trace(out_dir, "suite completed selected=%s" % selected["id"])
    return {
        "suite": suite_name,
        "suite_meta": {key: value for key, value in suite.items() if key != "variants"},
        "default_data": default_data,
        "results": results,
        "selected": row_from_result(selected),
    }


def output_dir(suite_name, suite, args):
    if args.resume_dir:
        path = os.path.abspath(args.resume_dir)
        os.makedirs(path, exist_ok=True)
        return path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.abspath(args.output_root or RESULTS_ROOT)
    path = os.path.join(root, suite["result_group"], "%s_%s" % (suite_name, timestamp))
    os.makedirs(path, exist_ok=True)
    return path


def save_csv(path, results):
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "id",
            "model",
            "setting",
            "train_acc",
            "history_train_acc",
            "val_acc",
            "test_acc",
            "train_val_gap",
            "final_loss",
            "elapsed_seconds",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow(row_from_result(item))


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_plots(out_dir, suite_name, results):
    plt.figure(figsize=(8, 5))
    for item in results:
        if item["loss_history"]:
            plt.plot(item["loss_history"], label=item["id"])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("%s Loss Curves" % suite_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "%s_loss_curves.png" % suite_name), dpi=160)
    plt.close()

    labels = [item["id"] for item in results]
    val_acc = [item["val_acc"] for item in results]
    test_acc = [item["test_acc"] for item in results]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, val_acc, width, label="Validation")
    plt.bar(x + width / 2, test_acc, width, label="Test")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylim(0, max(max(val_acc), max(test_acc), 0.1) + 0.05)
    plt.ylabel("Accuracy")
    plt.title("%s Accuracy" % suite_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "%s_accuracy.png" % suite_name), dpi=160)
    plt.close()


def save_outputs_to_dir(out_dir, payload):
    save_csv(os.path.join(out_dir, "%s_metrics.csv" % payload["suite"]), payload["results"])
    save_json(os.path.join(out_dir, "%s_summary.json" % payload["suite"]), payload)
    save_plots(out_dir, payload["suite"], payload["results"])
    return out_dir


def save_outputs(payload, args):
    out_dir = output_dir(payload["suite"], payload["suite_meta"], args)
    return save_outputs_to_dir(out_dir, payload)


def print_suite_list(config):
    for name in sorted(config["suites"].keys()):
        suite = config["suites"][name]
        print("%s\t%s\t%s" % (name, suite["type"], suite["title"]))


def print_dry_run(config, suite_name):
    suite = config["suites"][suite_name]
    print("[%s] %s" % (suite_name, suite["title"]))
    print("type=%s, result_group=%s" % (suite["type"], suite["result_group"]))
    for variant in suite["variants"]:
        print("  - %s: %s" % (variant["id"], variant_setting(variant)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run configurable Assignment 2 exploration suites."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--suite", default="conv_capacity_reg_search")
    parser.add_argument("--list-suites", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default=RESULTS_ROOT)
    parser.add_argument("--resume-dir", default=None)
    parser.add_argument("--max-variants", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.list_suites:
        print_suite_list(config)
        return

    selected = selected_suite_names(config, args.suite)
    if args.dry_run:
        for suite_name in selected:
            print_dry_run(config, suite_name)
        return

    for suite_name in selected:
        start = time.time()
        out_dir = output_dir(suite_name, config["suites"][suite_name], args)
        payload = run_suite(config, suite_name, args, out_dir)
        payload["elapsed_seconds"] = time.time() - start
        save_outputs_to_dir(out_dir, payload)
        chosen = payload["selected"]
        print(
            "%s: selected=%s val_acc=%.4f test_acc=%.4f output=%s"
            % (suite_name, chosen["id"], chosen["val_acc"], chosen["test_acc"], out_dir)
        )


if __name__ == "__main__":
    main()
