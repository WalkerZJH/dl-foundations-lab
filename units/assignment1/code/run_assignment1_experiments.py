import csv
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cs231n.classifiers.fc_net import FullyConnectedNet, TwoLayerNet
from cs231n.classifiers.k_nearest_neighbor import KNearestNeighbor
from cs231n.classifiers.linear_classifier import Softmax
from cs231n.data_utils import get_CIFAR10_data
from cs231n.solver import Solver


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "_experiment_outputs")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def ensure_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)


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


def run_knn(flat):
    train_n = flat["X_train"].shape[0]
    val_n = flat["X_val"].shape[0]
    test_n = flat["X_test"].shape[0]

    model = KNearestNeighbor()
    model.train(flat["X_train"], flat["y_train"])
    val_dists = model.compute_distances_no_loops(flat["X_val"])
    test_dists = model.compute_distances_no_loops(flat["X_test"])

    best = {
        "model": "kNN",
        "k": None,
        "train_n": train_n,
        "val_n": val_n,
        "test_n": test_n,
        "val_acc": 0.0,
        "test_acc": 0.0,
    }
    rows = []
    for k in [1, 3, 5, 7]:
        val_pred = model.predict_labels(val_dists, k=k)
        val_acc = accuracy(val_pred, flat["y_val"])
        test_pred = model.predict_labels(test_dists, k=k)
        test_acc = accuracy(test_pred, flat["y_test"])
        rows.append(
            {
                "model": "kNN",
                "setting": f"k={k}, train={train_n}, val={val_n}, test={test_n}",
                "train_acc": "",
                "val_acc": val_acc,
                "test_acc": test_acc,
            }
        )
        if val_acc >= best["val_acc"]:
            best.update({"k": k, "val_acc": val_acc, "test_acc": test_acc})
    return best, rows


def run_softmax(flat):
    model = Softmax()
    loss_history = model.train(
        flat["X_train"],
        flat["y_train"],
        learning_rate=1e-7,
        reg=2.5e4,
        num_iters=500,
        batch_size=200,
        verbose=False,
    )
    train_acc = accuracy(model.predict(flat["X_train"]), flat["y_train"])
    val_acc = accuracy(model.predict(flat["X_val"]), flat["y_val"])
    test_acc = accuracy(model.predict(flat["X_test"]), flat["y_test"])
    return {
        "model": "Softmax",
        "setting": "lr=1e-7, reg=2.5e4, iters=500",
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "loss_history": loss_history,
    }


def run_solver_model(model, data, name, update_rule="sgd_momentum", learning_rate=1e-3):
    solver = Solver(
        model,
        data,
        update_rule=update_rule,
        optim_config={"learning_rate": learning_rate},
        lr_decay=0.95,
        num_epochs=3,
        batch_size=100,
        num_train_samples=500,
        num_val_samples=None,
        print_every=100,
        verbose=False,
    )
    solver.train()
    test_acc = solver.check_accuracy(data["X_test"], data["y_test"], num_samples=None)
    return {
        "model": name,
        "setting": f"{update_rule}, lr={learning_rate}, epochs=3",
        "train_acc": float(solver.train_acc_history[-1]),
        "val_acc": float(solver.best_val_acc),
        "test_acc": float(test_acc),
        "loss_history": [float(x) for x in solver.loss_history],
        "train_acc_history": [float(x) for x in solver.train_acc_history],
        "val_acc_history": [float(x) for x in solver.val_acc_history],
    }


def save_metrics(rows, summary):
    csv_path = os.path.join(RESULTS_DIR, "assignment1_experiment_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "setting", "train_acc", "val_acc", "test_acc"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    json_path = os.path.join(RESULTS_DIR, "assignment1_experiment_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def save_plots(results):
    plt.figure(figsize=(8, 5))
    for item in results:
        losses = item.get("loss_history")
        if losses:
            plt.plot(losses, label=item["model"])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Assignment 1 Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "assignment1_loss_curves.png"), dpi=160)
    plt.close()

    names = [item["model"] for item in results]
    val_acc = [item.get("val_acc", 0.0) for item in results]
    test_acc = [item.get("test_acc", 0.0) for item in results]
    x = np.arange(len(names))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, val_acc, width, label="Validation")
    plt.bar(x + width / 2, test_acc, width, label="Test")
    plt.xticks(x, names, rotation=15)
    plt.ylim(0, max(max(val_acc), max(test_acc), 0.1) + 0.05)
    plt.ylabel("Accuracy")
    plt.title("Assignment 1 Accuracy Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "assignment1_accuracy_comparison_full_knn.png"), dpi=160)
    plt.close()

def main():
    ensure_dirs()
    start = time.time()
    np.random.seed(42)

    data = get_CIFAR10_data(num_training=2000, num_validation=500, num_test=500)
    flat = flatten_data(data)

    log_lines = [
        "Assignment 1 experiment log",
        f"X_train={data['X_train'].shape}, X_val={data['X_val'].shape}, X_test={data['X_test'].shape}",
        "kNN uses full current subset: train=2000, val=500, test=500",
    ]

    knn_best, knn_rows = run_knn(flat)
    softmax = run_softmax(flat)
    two_layer = run_solver_model(
        TwoLayerNet(weight_scale=1e-2, reg=0.1),
        data,
        "TwoLayerNet",
        update_rule="sgd_momentum",
        learning_rate=1e-3,
    )
    full_net = run_solver_model(
        FullyConnectedNet([100, 100], weight_scale=5e-2, reg=0.1),
        data,
        "FullyConnectedNet",
        update_rule="adam",
        learning_rate=1e-3,
    )

    results = [
        {
            "model": "kNN",
            "setting": (
                f"k={knn_best['k']}, train={knn_best['train_n']}, "
                f"val={knn_best['val_n']}, test={knn_best['test_n']}"
            ),
            "train_acc": "",
            "val_acc": knn_best["val_acc"],
            "test_acc": knn_best["test_acc"],
        },
        softmax,
        two_layer,
        full_net,
    ]

    rows = []
    rows.extend(knn_rows)
    for item in results[1:]:
        rows.append(
            {
                "model": item["model"],
                "setting": item["setting"],
                "train_acc": item["train_acc"],
                "val_acc": item["val_acc"],
                "test_acc": item["test_acc"],
            }
        )

    elapsed = time.time() - start
    save_metrics(rows, {"results": results, "elapsed_seconds": elapsed})
    save_plots(results)

    for item in results:
        log_lines.append(
            "{model}: setting={setting}, val_acc={val_acc:.4f}, test_acc={test_acc:.4f}".format(
                **item
            )
        )
    log_lines.append(f"elapsed_seconds={elapsed:.2f}")
    with open(os.path.join(RESULTS_DIR, "assignment1_experiment_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    print("\n".join(log_lines))


if __name__ == "__main__":
    main()
