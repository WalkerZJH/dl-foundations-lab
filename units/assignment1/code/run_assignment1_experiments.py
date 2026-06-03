import argparse
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
RESULTS_DIR = os.path.join(ROOT, "results", "baseline")
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


def run_softmax(flat, num_iters=1500):
    model = Softmax()
    loss_history = model.train(
        flat["X_train"],
        flat["y_train"],
        learning_rate=1e-7,
        reg=2.5e4,
        num_iters=num_iters,
        batch_size=200,
        verbose=False,
    )
    train_acc = accuracy(model.predict(flat["X_train"]), flat["y_train"])
    val_acc = accuracy(model.predict(flat["X_val"]), flat["y_val"])
    test_acc = accuracy(model.predict(flat["X_test"]), flat["y_test"])
    return {
        "model": "Softmax",
        "setting": f"lr=1e-7, reg=2.5e4, iters={num_iters}",
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "loss_history": loss_history,
    }


def run_solver_model(
    model,
    data,
    name,
    update_rule="sgd_momentum",
    learning_rate=1e-3,
    epochs=10,
    batch_size=200,
):
    solver = Solver(
        model,
        data,
        update_rule=update_rule,
        optim_config={"learning_rate": learning_rate},
        lr_decay=0.95,
        num_epochs=epochs,
        batch_size=batch_size,
        num_train_samples=None,
        num_val_samples=None,
        print_every=100,
        verbose=False,
    )
    solver.train()
    test_acc = solver.check_accuracy(data["X_test"], data["y_test"], num_samples=None)
    return {
        "model": name,
        "setting": f"{update_rule}, lr={learning_rate}, epochs={epochs}",
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


def save_summary_md(results, elapsed):
    path = os.path.join(RESULTS_DIR, "assignment1_experiment_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Assignment 1 Baseline 实验结果摘要\n\n")
        f.write(
            "实验数据：CIFAR-10，训练集 {train_n} 张、验证集 {val_n} 张、测试集 {test_n} 张。\n\n".format(
                train_n=results[0].get("train_n", "full"),
                val_n=results[0].get("val_n", "full"),
                test_n=results[0].get("test_n", "full"),
            )
        )
        f.write("实验耗时：%.2f 秒。\n\n" % elapsed)
        f.write("| 模型 | 设置 | 训练准确率 | 验证准确率 | 测试准确率 |\n")
        f.write("| --- | --- | ---: | ---: | ---: |\n")
        for item in results:
            train_acc = item["train_acc"] if item["train_acc"] != "" else "-"
            train_text = train_acc if isinstance(train_acc, str) else "%.4f" % train_acc
            f.write(
                "| {model} | {setting} | {train_acc} | {val_acc:.4f} | {test_acc:.4f} |\n".format(
                    model=item["model"],
                    setting=item["setting"],
                    train_acc=train_text,
                    val_acc=item["val_acc"],
                    test_acc=item["test_acc"],
                )
            )
        f.write("\n## 图片输出\n\n")
        f.write("* `figures/assignment1_loss_curves.png`：训练损失曲线。\n")
        f.write("* `figures/assignment1_accuracy_comparison_full_data.png`：模型准确率对比图，供 LaTeX 报告引用。\n")
        f.write("* `figures/assignment1_accuracy_comparison_full_data.svg`：模型准确率对比图。\n")


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
    plt.savefig(os.path.join(FIGURES_DIR, "assignment1_accuracy_comparison_full_data.png"), dpi=160)
    plt.savefig(os.path.join(FIGURES_DIR, "assignment1_accuracy_comparison_full_data.svg"))
    # Backward-compatible filenames for older README links.
    plt.savefig(os.path.join(FIGURES_DIR, "assignment1_accuracy_comparison_full_knn.png"), dpi=160)
    plt.savefig(os.path.join(FIGURES_DIR, "assignment1_accuracy_comparison_full_knn.svg"))
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Run Assignment 1 baseline experiments.")
    parser.add_argument("--train-size", type=int, default=49000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--skip-knn", action="store_true")
    parser.add_argument("--softmax-iters", type=int, default=1500)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs()
    start = time.time()
    np.random.seed(42)

    data = get_CIFAR10_data(
        num_training=args.train_size,
        num_validation=args.val_size,
        num_test=args.test_size,
    )
    flat = flatten_data(data)

    log_lines = [
        "Assignment 1 experiment log",
        f"X_train={data['X_train'].shape}, X_val={data['X_val'].shape}, X_test={data['X_test'].shape}",
        "configuration: train=%d, val=%d, test=%d, epochs=%d, batch_size=%d"
        % (args.train_size, args.val_size, args.test_size, args.epochs, args.batch_size),
    ]

    knn_best, knn_rows = None, []
    if not args.skip_knn:
        knn_best, knn_rows = run_knn(flat)
    softmax = run_softmax(flat, num_iters=args.softmax_iters)
    two_layer = run_solver_model(
        TwoLayerNet(weight_scale=1e-2, reg=0.1),
        data,
        "TwoLayerNet",
        update_rule="sgd_momentum",
        learning_rate=1e-3,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    full_net = run_solver_model(
        FullyConnectedNet([100, 100], weight_scale=5e-2, reg=0.1),
        data,
        "FullyConnectedNet",
        update_rule="adam",
        learning_rate=1e-3,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    results = []
    if knn_best is not None:
        results.append({
            "model": "kNN",
            "setting": (
                f"k={knn_best['k']}, train={knn_best['train_n']}, "
                f"val={knn_best['val_n']}, test={knn_best['test_n']}"
            ),
            "train_acc": "",
            "val_acc": knn_best["val_acc"],
            "test_acc": knn_best["test_acc"],
            "train_n": args.train_size,
            "val_n": args.val_size,
            "test_n": args.test_size,
        })
    results.extend([softmax, two_layer, full_net])
    for item in results:
        item.setdefault("train_n", args.train_size)
        item.setdefault("val_n", args.val_size)
        item.setdefault("test_n", args.test_size)

    rows = []
    rows.extend(knn_rows)
    for item in [item for item in results if item["model"] != "kNN"]:
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
    save_summary_md(results, elapsed)

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
