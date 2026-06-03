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

from cs231n.classifiers.fc_net import FullyConnectedNet, TwoLayerNet
from cs231n.classifiers.k_nearest_neighbor import KNearestNeighbor
from cs231n.classifiers.linear_classifier import LinearSVM, Softmax
from cs231n.data_utils import get_CIFAR10_data
from cs231n.solver import Solver


ASSIGNMENT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_ROOT = os.path.join(ASSIGNMENT_ROOT, "results")
DEFAULT_CONFIG = os.path.join(ASSIGNMENT_ROOT, "experiments", "assignment1_exploration_suites.json")


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_data(data):
    return {
        "X_train": data["X_train"].reshape(data["X_train"].shape[0], -1),
        "y_train": data["y_train"],
        "X_val": data["X_val"].reshape(data["X_val"].shape[0], -1),
        "y_val": data["y_val"],
        "X_test": data["X_test"].reshape(data["X_test"].shape[0], -1),
        "y_test": data["y_test"],
    }


def accuracy(y_pred, y_true):
    return float(np.mean(y_pred == y_true))


def merge_dict(base, override):
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        merged[key] = value
    return merged


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
        "classifier",
        "k",
        "learning_rate",
        "reg",
        "hidden_dim",
        "hidden_dims",
        "batch_size",
        "num_iters",
        "epochs",
        "update_rule",
        "init",
        "normalization",
        "dropout_keep_ratio",
    ]
    return ", ".join("%s=%s" % (key, variant[key]) for key in keys if key in variant)


def apply_initialization(model, init_name):
    if init_name in (None, "normal"):
        return
    for key, value in list(model.params.items()):
        if not key.startswith("W"):
            continue
        if init_name == "zero":
            model.params[key] = np.zeros_like(value)
        elif init_name == "he":
            fan_in = value.shape[0]
            model.params[key] = np.random.randn(*value.shape) * np.sqrt(2.0 / fan_in)
        elif init_name == "xavier":
            fan_in, fan_out = value.shape[0], value.shape[1]
            model.params[key] = np.random.randn(*value.shape) * np.sqrt(
                2.0 / (fan_in + fan_out)
            )
        else:
            raise ValueError("Unknown init mode: %s" % init_name)


def build_model(variant, data):
    input_dim = int(np.prod(data["X_train"].shape[1:]))
    if variant["model"] == "two_layer":
        model = TwoLayerNet(
            input_dim=input_dim,
            hidden_dim=variant.get("hidden_dim", 100),
            num_classes=10,
            weight_scale=variant.get("weight_scale", 1e-2),
            reg=variant.get("reg", 0.0),
        )
    elif variant["model"] == "fully_connected":
        model = FullyConnectedNet(
            variant.get("hidden_dims", [100, 100]),
            input_dim=input_dim,
            num_classes=10,
            dropout_keep_ratio=variant.get("dropout_keep_ratio", 1.0),
            normalization=variant.get("normalization"),
            reg=variant.get("reg", 0.0),
            weight_scale=variant.get("weight_scale", 1e-2),
            seed=123,
        )
    else:
        raise ValueError("Unknown model: %s" % variant["model"])
    apply_initialization(model, variant.get("init"))
    return model


def row_from_result(item):
    return {
        "id": item["id"],
        "model": item["model"],
        "setting": item["setting"],
        "train_acc": item["train_acc"],
        "val_acc": item["val_acc"],
        "test_acc": item["test_acc"],
        "train_val_gap": item["train_val_gap"],
        "final_loss": item["final_loss"],
        "elapsed_seconds": item["elapsed_seconds"],
    }


def best_by_val(results):
    return max(results, key=lambda item: item["val_acc"])


def elbow_by_val(results, tolerance):
    best = best_by_val(results)
    threshold = best["val_acc"] - tolerance
    eligible = [item for item in results if item["val_acc"] >= threshold]
    return min(eligible, key=lambda item: item["variant"].get("k", 0))


def choose_result(results, selection):
    method = (selection or {}).get("method", "best_val")
    if method == "best_val":
        return best_by_val(results)
    if method == "elbow":
        return elbow_by_val(results, selection.get("tolerance", 0.01))
    raise ValueError("Unknown selection method: %s" % method)


def run_knn_variant(variant, model, flat, val_dists, test_dists):
    start = time.time()
    val_pred = model.predict_labels(val_dists, k=variant["k"])
    test_pred = model.predict_labels(test_dists, k=variant["k"])
    return {
        "id": variant["id"],
        "model": "kNN",
        "setting": variant_setting(variant),
        "train_acc": "",
        "val_acc": accuracy(val_pred, flat["y_val"]),
        "test_acc": accuracy(test_pred, flat["y_test"]),
        "train_val_gap": "",
        "final_loss": "",
        "elapsed_seconds": time.time() - start,
        "loss_history": [],
        "train_acc_history": [],
        "val_acc_history": [],
        "variant": variant,
    }


def run_knn_suite_chunked(suite, flat, out_dir, args):
    start = time.time()
    model = KNearestNeighbor()
    model.train(flat["X_train"], flat["y_train"])
    variants = suite["variants"]
    states = {
        variant["id"]: {
            "variant": variant,
            "val_correct": 0,
            "val_seen": 0,
            "test_correct": 0,
            "test_seen": 0,
        }
        for variant in variants
    }

    def consume_split(split_name, X, y):
        total = X.shape[0]
        chunk_size = args.knn_chunk_size
        for start_idx in range(0, total, chunk_size):
            end_idx = min(start_idx + chunk_size, total)
            trace_event(
                out_dir,
                "kNN %s chunk %d:%d / %d" % (split_name, start_idx, end_idx, total),
            )
            dists = model.compute_distances_no_loops(X[start_idx:end_idx])
            for variant in variants:
                pred = model.predict_labels(dists, k=variant["k"])
                state = states[variant["id"]]
                correct = int(np.sum(pred == y[start_idx:end_idx]))
                state["%s_correct" % split_name] += correct
                state["%s_seen" % split_name] += end_idx - start_idx

    consume_split("val", flat["X_val"], flat["y_val"])
    consume_split("test", flat["X_test"], flat["y_test"])

    elapsed = time.time() - start
    results = []
    for variant in variants:
        state = states[variant["id"]]
        val_acc = state["val_correct"] / float(state["val_seen"])
        test_acc = state["test_correct"] / float(state["test_seen"])
        results.append(
            {
                "id": variant["id"],
                "model": "kNN",
                "setting": variant_setting(variant),
                "train_acc": "",
                "val_acc": val_acc,
                "test_acc": test_acc,
                "train_val_gap": "",
                "final_loss": "",
                "elapsed_seconds": elapsed,
                "loss_history": [],
                "train_acc_history": [],
                "val_acc_history": [],
                "variant": variant,
            }
        )
    return results


def run_linear_variant(variant, flat):
    start = time.time()
    model = Softmax() if variant["classifier"] == "softmax" else LinearSVM()
    loss_history = model.train(
        flat["X_train"],
        flat["y_train"],
        learning_rate=variant.get("learning_rate", 1e-7),
        reg=variant.get("reg", 2.5e4),
        num_iters=variant.get("num_iters", 500),
        batch_size=variant.get("batch_size", 200),
        verbose=False,
    )
    train_acc = accuracy(model.predict(flat["X_train"]), flat["y_train"])
    val_acc = accuracy(model.predict(flat["X_val"]), flat["y_val"])
    test_acc = accuracy(model.predict(flat["X_test"]), flat["y_test"])
    return {
        "id": variant["id"],
        "model": "Softmax" if variant["classifier"] == "softmax" else "LinearSVM",
        "setting": variant_setting(variant),
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "train_val_gap": train_acc - val_acc,
        "final_loss": float(loss_history[-1]) if loss_history else "",
        "elapsed_seconds": time.time() - start,
        "loss_history": [float(x) for x in loss_history],
        "train_acc_history": [],
        "val_acc_history": [],
        "variant": variant,
    }


def run_solver_variant(variant, data, solver_base):
    start = time.time()
    model = build_model(variant, data)
    solver_cfg = merge_dict(solver_base, variant)
    solver = Solver(
        model,
        data,
        update_rule=solver_cfg.get("update_rule", "sgd_momentum"),
        optim_config={"learning_rate": solver_cfg.get("learning_rate", 1e-3)},
        lr_decay=solver_cfg.get("lr_decay", 0.95),
        num_epochs=solver_cfg.get("epochs", 3),
        batch_size=solver_cfg.get("batch_size", 100),
        num_train_samples=solver_cfg.get("num_train_samples", 500),
        num_val_samples=None,
        print_every=100,
        verbose=False,
    )
    solver.train()
    test_acc = solver.check_accuracy(data["X_test"], data["y_test"], num_samples=None)
    train_acc = float(solver.train_acc_history[-1])
    val_acc = float(solver.best_val_acc)
    return {
        "id": variant["id"],
        "model": variant["model"],
        "setting": variant_setting(solver_cfg),
        "train_acc": train_acc,
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


def limit_suite_variants(suite, args):
    suite = deepcopy(suite)
    if args.max_variants is not None:
        suite["variants"] = suite["variants"][: args.max_variants]
    return suite


def trace_event(out_dir, message):
    timestamp = datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(out_dir, "run_trace.txt"), "a", encoding="utf-8") as f:
        f.write("[%s] %s\n" % (timestamp, message))


def load_resume_results(out_dir, suite_name):
    summary_path = os.path.join(out_dir, "%s_summary.json" % suite_name)
    if not os.path.exists(summary_path):
        return []
    with open(summary_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("results", [])


def build_payload(suite_name, suite, data_cfg, results, status, started_at, elapsed_seconds):
    selected = row_from_result(choose_result(results, suite.get("selection"))) if results else None
    return {
        "suite": suite_name,
        "status": status,
        "started_at": started_at,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed_seconds,
        "suite_meta": {key: value for key, value in suite.items() if key != "variants"},
        "data": data_cfg,
        "results": results,
        "selected": selected,
    }


def save_progress(out_dir, payload):
    suite_name = payload["suite"]
    save_csv(os.path.join(out_dir, "%s_metrics.csv" % suite_name), payload["results"])
    save_json(os.path.join(out_dir, "%s_summary.json" % suite_name), payload)
    save_plots(out_dir, suite_name, payload["results"])


def run_suite(config, suite_name, args, out_dir):
    suite = limit_suite_variants(config["suites"][suite_name], args)
    started_at = datetime.now().isoformat(timespec="seconds")
    cli_data = {
        key: value
        for key, value in {
            "train_size": args.train_size,
            "val_size": args.val_size,
            "test_size": args.test_size,
        }.items()
        if value is not None
    }
    data_cfg = merge_dict(config["defaults"]["data"], suite.get("data"))
    data_cfg = merge_dict(data_cfg, cli_data)
    solver_cfg = merge_dict(config["defaults"]["solver"], suite.get("solver"))
    if args.epochs is not None:
        solver_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        solver_cfg["batch_size"] = args.batch_size

    np.random.seed(args.seed)
    data = get_CIFAR10_data(
        num_training=data_cfg["train_size"],
        num_validation=data_cfg["val_size"],
        num_test=data_cfg["test_size"],
    )
    flat = flatten_data(data)
    trace_event(
        out_dir,
        "loaded data train=%s val=%s test=%s"
        % (data_cfg["train_size"], data_cfg["val_size"], data_cfg["test_size"]),
    )

    results = load_resume_results(out_dir, suite_name) if args.resume_dir else []
    completed = {item["id"] for item in results}
    if completed:
        trace_event(out_dir, "resume found completed variants: %s" % ", ".join(sorted(completed)))

    suite_start = time.time()
    if suite["variants"][0]["kind"] == "knn":
        if completed:
            trace_event(out_dir, "kNN resume uses existing completed results")
        else:
            trace_event(out_dir, "start chunked kNN suite")
            results = run_knn_suite_chunked(suite, flat, out_dir, args)
    else:
        for variant in suite["variants"]:
            if variant["id"] in completed:
                continue
            trace_event(out_dir, "start variant %s" % variant["id"])
            np.random.seed(args.seed)
            if variant["kind"] == "linear":
                result = run_linear_variant(variant, flat)
            elif variant["kind"] == "solver":
                result = run_solver_variant(variant, data, solver_cfg)
            else:
                raise ValueError("Unknown variant kind: %s" % variant["kind"])
            results.append(result)
            payload = build_payload(
                suite_name,
                suite,
                data_cfg,
                results,
                "running",
                started_at,
                time.time() - suite_start,
            )
            save_progress(out_dir, payload)
            trace_event(
                out_dir,
                "finish variant %s val_acc=%.4f test_acc=%.4f"
                % (result["id"], result["val_acc"], result["test_acc"]),
            )

    if suite["variants"][0]["kind"] == "knn" and not completed:
        payload = build_payload(
            suite_name,
            suite,
            data_cfg,
            results,
            "running",
            started_at,
            time.time() - suite_start,
        )
        save_progress(out_dir, payload)
        trace_event(out_dir, "finish chunked kNN suite")

    return build_payload(
        suite_name,
        suite,
        data_cfg,
        results,
        "completed",
        started_at,
        time.time() - suite_start,
    )


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
    loss_items = [item for item in results if item["loss_history"]]
    if loss_items:
        plt.figure(figsize=(8, 5))
        for item in loss_items:
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


def save_outputs(payload, args):
    out_dir = args.current_output_dir
    save_progress(out_dir, payload)
    trace_event(out_dir, "suite completed selected=%s" % payload["selected"]["id"])
    return out_dir


def print_suite_list(config):
    for name in sorted(config["suites"].keys()):
        suite = config["suites"][name]
        print("%s\t%s\t%s" % (name, suite["type"], suite["title"]))


def print_dry_run(config, suite_name, args):
    suite = limit_suite_variants(config["suites"][suite_name], args)
    print("[%s] %s" % (suite_name, suite["title"]))
    print("type=%s, result_group=%s" % (suite["type"], suite["result_group"]))
    print("selection=%s" % suite.get("selection", {"method": "best_val"}))
    for variant in suite["variants"]:
        print("  - %s: %s" % (variant["id"], variant_setting(variant)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run configurable Assignment 1 exploration suites."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--suite", default="two_layer_lr_search")
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
    parser.add_argument("--knn-chunk-size", type=int, default=200)
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
            print_dry_run(config, suite_name, args)
        return

    for suite_name in selected:
        start = time.time()
        suite = limit_suite_variants(config["suites"][suite_name], args)
        args.current_output_dir = output_dir(suite_name, suite, args)
        trace_event(args.current_output_dir, "suite started: %s" % suite_name)
        payload = run_suite(config, suite_name, args, args.current_output_dir)
        payload["elapsed_seconds"] = time.time() - start
        out_dir = save_outputs(payload, args)
        chosen = payload["selected"]
        print(
            "%s: selected=%s val_acc=%.4f test_acc=%.4f output=%s"
            % (suite_name, chosen["id"], chosen["val_acc"], chosen["test_acc"], out_dir)
        )


if __name__ == "__main__":
    main()
