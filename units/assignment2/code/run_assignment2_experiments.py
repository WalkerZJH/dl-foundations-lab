import argparse
import csv
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cs231n.classifiers.cnn import ThreeLayerConvNet
from cs231n.classifiers.fc_net import FullyConnectedNet
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient_array
from cs231n.layers import (
    batchnorm_backward,
    batchnorm_backward_alt,
    batchnorm_forward,
    conv_backward_naive,
    conv_forward_naive,
    dropout_backward,
    dropout_forward,
    layernorm_backward,
    layernorm_forward,
    max_pool_backward_naive,
    max_pool_forward_naive,
    spatial_batchnorm_backward,
    spatial_batchnorm_forward,
    spatial_groupnorm_backward,
    spatial_groupnorm_forward,
)
from cs231n.solver import Solver


ASSIGNMENT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ASSIGNMENT_ROOT, "results", "baseline")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
SMOKE_ERROR_THRESHOLD = 1e-6


def ensure_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def append_trace(message):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "run_trace.txt")
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write("[%s] %s\n" % (timestamp, message))


def rel_error(x, y):
    return float(np.max(np.abs(x - y) / np.maximum(1e-8, np.abs(x) + np.abs(y))))


def run_numpy_layer_smoke():
    np.random.seed(231)
    errors = {}

    x = np.random.randn(4, 5)
    gamma = np.random.randn(5)
    beta = np.random.randn(5)
    dout = np.random.randn(4, 5)
    out, cache = batchnorm_forward(x, gamma, beta, {"mode": "train"})
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    dx_alt, dgamma_alt, dbeta_alt = batchnorm_backward_alt(dout, cache)
    fx = lambda xx: batchnorm_forward(xx, gamma, beta, {"mode": "train"})[0]
    fg = lambda gg: batchnorm_forward(x, gg, beta, {"mode": "train"})[0]
    fb = lambda bb: batchnorm_forward(x, gamma, bb, {"mode": "train"})[0]
    errors["batchnorm_dx_error"] = rel_error(
        eval_numerical_gradient_array(fx, x.copy(), dout), dx
    )
    errors["batchnorm_dgamma_error"] = rel_error(
        eval_numerical_gradient_array(fg, gamma.copy(), dout), dgamma
    )
    errors["batchnorm_dbeta_error"] = rel_error(
        eval_numerical_gradient_array(fb, beta.copy(), dout), dbeta
    )
    errors["batchnorm_alt_dx_error"] = rel_error(dx, dx_alt)
    errors["batchnorm_alt_dgamma_error"] = rel_error(dgamma, dgamma_alt)
    errors["batchnorm_alt_dbeta_error"] = rel_error(dbeta, dbeta_alt)

    x = np.random.randn(3, 6)
    gamma = np.random.randn(6)
    beta = np.random.randn(6)
    dout = np.random.randn(3, 6)
    out, cache = layernorm_forward(x, gamma, beta, {})
    dx, dgamma, dbeta = layernorm_backward(dout, cache)
    fx = lambda xx: layernorm_forward(xx, gamma, beta, {})[0]
    fg = lambda gg: layernorm_forward(x, gg, beta, {})[0]
    fb = lambda bb: layernorm_forward(x, gamma, bb, {})[0]
    errors["layernorm_dx_error"] = rel_error(
        eval_numerical_gradient_array(fx, x.copy(), dout), dx
    )
    errors["layernorm_dgamma_error"] = rel_error(
        eval_numerical_gradient_array(fg, gamma.copy(), dout), dgamma
    )
    errors["layernorm_dbeta_error"] = rel_error(
        eval_numerical_gradient_array(fb, beta.copy(), dout), dbeta
    )

    x = np.random.randn(4, 5)
    dout = np.random.randn(4, 5)
    out, cache = dropout_forward(x, {"mode": "train", "p": 0.7, "seed": 123})
    dx = dropout_backward(dout, cache)
    if out.shape != x.shape or dx.shape != x.shape:
        raise AssertionError("dropout smoke test failed: shape mismatch")

    x = np.random.randn(2, 3, 5, 5)
    w = np.random.randn(2, 3, 3, 3)
    b = np.random.randn(2)
    conv_param = {"stride": 1, "pad": 1}
    out, cache = conv_forward_naive(x, w, b, conv_param)
    dout = np.random.randn(*out.shape)
    dx, dw, db = conv_backward_naive(dout, cache)
    fx = lambda xx: conv_forward_naive(xx, w, b, conv_param)[0]
    fw = lambda ww: conv_forward_naive(x, ww, b, conv_param)[0]
    fb = lambda bb: conv_forward_naive(x, w, bb, conv_param)[0]
    errors["conv_dx_error"] = rel_error(
        eval_numerical_gradient_array(fx, x.copy(), dout), dx
    )
    errors["conv_dw_error"] = rel_error(
        eval_numerical_gradient_array(fw, w.copy(), dout), dw
    )
    errors["conv_db_error"] = rel_error(
        eval_numerical_gradient_array(fb, b.copy(), dout), db
    )

    x = np.random.randn(2, 3, 4, 4)
    pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}
    out, cache = max_pool_forward_naive(x, pool_param)
    dout = np.random.randn(*out.shape)
    dx = max_pool_backward_naive(dout, cache)
    fx = lambda xx: max_pool_forward_naive(xx, pool_param)[0]
    errors["max_pool_dx_error"] = rel_error(
        eval_numerical_gradient_array(fx, x.copy(), dout), dx
    )

    x = np.random.randn(2, 3, 4, 5)
    gamma = np.ones(3)
    beta = np.zeros(3)
    bn_param = {"mode": "train"}
    out, cache = spatial_batchnorm_forward(x, gamma, beta, bn_param)
    dout = np.random.randn(*out.shape)
    dx, dgamma, dbeta = spatial_batchnorm_backward(dout, cache)
    if dx.shape != x.shape or dgamma.shape != gamma.shape or dbeta.shape != beta.shape:
        raise AssertionError("spatial batchnorm smoke test failed: shape mismatch")

    x = np.random.randn(2, 6, 4, 4)
    gamma = np.ones((1, 6, 1, 1))
    beta = np.zeros((1, 6, 1, 1))
    out, cache = spatial_groupnorm_forward(x, gamma, beta, G=3, gn_param={})
    dout = np.random.randn(*out.shape)
    dx, dgamma, dbeta = spatial_groupnorm_backward(dout, cache)
    if dx.shape != x.shape or dgamma.shape != gamma.shape or dbeta.shape != beta.shape:
        raise AssertionError("spatial groupnorm smoke test failed: shape mismatch")

    failed = {name: value for name, value in errors.items() if value > SMOKE_ERROR_THRESHOLD}
    if failed:
        raise AssertionError(f"NumPy layer smoke test failed: {failed}")
    return {"name": "NumPy 层梯度烟测", "status": "passed"}


def run_pytorch_rnn_smoke():
    import torch
    from cs231n.classifiers.rnn_pytorch import CaptioningRNN

    torch.manual_seed(231)
    word_to_idx = {"<NULL>": 0, "<START>": 1, "<END>": 2, "cat": 3, "sat": 4}
    model = CaptioningRNN(
        word_to_idx,
        input_dim=6,
        wordvec_dim=4,
        hidden_dim=5,
        cell_type="rnn",
        dtype=torch.float64,
    )
    features = torch.randn(3, 6, dtype=torch.float64)
    captions = torch.tensor(
        [[1, 3, 4, 2, 0], [1, 4, 3, 2, 0], [1, 3, 2, 0, 0]], dtype=torch.long
    )
    loss = model.loss(features, captions)
    sampled = model.sample(features, max_length=4)
    if not torch.isfinite(loss):
        raise AssertionError("PyTorch RNN smoke test failed: non-finite loss")
    if tuple(sampled.shape) != (3, 4):
        raise AssertionError("PyTorch RNN smoke test failed: sample shape mismatch")
    return {"name": "PyTorch RNN Captioning 烟测", "status": "passed"}


def format_smoke_tests_for_log(smoke_tests):
    log_names = {
        "NumPy 层梯度烟测": "numpy_layer_gradient_smoke",
        "PyTorch RNN Captioning 烟测": "pytorch_rnn_captioning_smoke",
    }
    return "; ".join(
        f"{log_names.get(item['name'], item['name'])}: {item['status']}"
        for item in smoke_tests
    )


def solver_train_sample_count(value, train_size):
    if value is None:
        return None
    return min(value, train_size)


def run_solver_model(
    model,
    data,
    name,
    setting,
    update_rule,
    learning_rate,
    epochs,
    batch_size,
    num_train_samples,
):
    solver = Solver(
        model,
        data,
        update_rule=update_rule,
        optim_config={"learning_rate": learning_rate},
        lr_decay=0.95,
        num_epochs=epochs,
        batch_size=batch_size,
        num_train_samples=solver_train_sample_count(
            num_train_samples, data["X_train"].shape[0]
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
    return {
        "model": name,
        "setting": setting,
        "train_acc": float(full_train_acc),
        "history_train_acc": float(solver.train_acc_history[-1]),
        "val_acc": float(solver.best_val_acc),
        "test_acc": float(test_acc),
        "final_loss": float(solver.loss_history[-1]),
        "loss_history": [float(x) for x in solver.loss_history],
        "train_acc_history": [float(x) for x in solver.train_acc_history],
        "val_acc_history": [float(x) for x in solver.val_acc_history],
    }


def run_cifar_experiments(args):
    data = get_CIFAR10_data(
        num_training=args.train_size,
        num_validation=args.val_size,
        num_test=args.test_size,
    )
    append_trace(
        "loaded data train=%d val=%d test=%d"
        % (args.train_size, args.val_size, args.test_size)
    )

    fc_model = FullyConnectedNet(
        [100, 100],
        input_dim=3 * 32 * 32,
        num_classes=10,
        normalization="batchnorm",
        dropout_keep_ratio=0.8,
        reg=0.1,
        weight_scale=5e-2,
    )
    append_trace("start model FullyConnectedNet + BN + Dropout")
    fc_result = run_solver_model(
        fc_model,
        data,
        "FullyConnectedNet + BN + Dropout",
        f"adam, lr=1e-3, epochs={args.epochs}, hidden=[100,100], keep=0.8",
        "adam",
        1e-3,
        args.epochs,
        args.batch_size,
        args.num_train_samples,
    )
    append_trace(
        "finish model FullyConnectedNet + BN + Dropout val_acc=%.4f test_acc=%.4f"
        % (fc_result["val_acc"], fc_result["test_acc"])
    )

    conv_model = ThreeLayerConvNet(
        num_filters=args.conv_filters,
        filter_size=3,
        hidden_dim=100,
        weight_scale=1e-2,
        reg=1e-3,
    )
    append_trace("start model ThreeLayerConvNet")
    conv_result = run_solver_model(
        conv_model,
        data,
        "ThreeLayerConvNet",
        (
            f"adam, lr=1e-3, epochs={args.epochs}, "
            f"filters={args.conv_filters}, filter_size=3"
        ),
        "adam",
        1e-3,
        args.epochs,
        args.batch_size,
        args.num_train_samples,
    )
    append_trace(
        "finish model ThreeLayerConvNet val_acc=%.4f test_acc=%.4f"
        % (conv_result["val_acc"], conv_result["test_acc"])
    )

    return data, [fc_result, conv_result]


def save_metrics(results):
    csv_path = os.path.join(RESULTS_DIR, "assignment2_experiment_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "setting",
                "train_acc",
                "history_train_acc",
                "val_acc",
                "test_acc",
                "final_loss",
            ],
        )
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "model": item["model"],
                    "setting": item["setting"],
                    "train_acc": item["train_acc"],
                    "history_train_acc": item["history_train_acc"],
                    "val_acc": item["val_acc"],
                    "test_acc": item["test_acc"],
                    "final_loss": item["final_loss"],
                }
            )


def save_summary(summary):
    json_path = os.path.join(RESULTS_DIR, "assignment2_experiment_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log_path = os.path.join(RESULTS_DIR, "assignment2_experiment_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary["log_lines"]) + "\n")

    md_path = os.path.join(RESULTS_DIR, "assignment2_experiment_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Assignment 2 实验摘要\n\n")
        f.write("## 数据与配置\n\n")
        cfg = summary["config"]
        f.write(
            f"* CIFAR-10 完整训练设置：train={cfg['train_size']}，"
            f"val={cfg['val_size']}，test={cfg['test_size']}。\n"
        )
        f.write(
            f"* 训练配置：epochs={cfg['epochs']}，batch_size={cfg['batch_size']}，"
            f"conv_filters={cfg['conv_filters']}，"
            f"num_train_samples={cfg['num_train_samples']}。\n"
        )
        f.write("\n## 烟测结果\n\n")
        f.write("| 检查项 | 结果 |\n")
        f.write("| --- | --- |\n")
        for item in summary["smoke_tests"]:
            f.write(f"| {item['name']} | {item['status']} |\n")
        f.write("\n烟测只用于确认实现路径可运行，不参与正式模型性能比较。\n\n")
        f.write("## 正式模型结果\n\n")
        f.write("| 模型 | 设置 | 完整训练准确率 | 验证准确率 | 测试准确率 | 最终 loss |\n")
        f.write("| --- | --- | ---: | ---: | ---: | ---: |\n")
        for item in summary["results"]:
            f.write(
                "| {model} | {setting} | {train_acc:.4f} | {val_acc:.4f} | "
                "{test_acc:.4f} | {final_loss:.4f} |\n".format(**item)
            )
        f.write("\n## 说明\n\n")
        f.write(
            "正式结果仅来自 CIFAR-10 训练实验；烟测结果只记录通过或失败。"
            f"当前正式训练轮数为 {cfg['epochs']} epochs。\n"
        )


def save_plots(results):
    plt.figure(figsize=(8, 5))
    for item in results:
        plt.plot(item["loss_history"], label=item["model"])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Assignment 2 Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "assignment2_loss_curves.png"), dpi=160)
    plt.close()

    names = [item["model"] for item in results]
    val_acc = [item["val_acc"] for item in results]
    test_acc = [item["test_acc"] for item in results]
    x = np.arange(len(names))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, val_acc, width, label="Validation")
    plt.bar(x + width / 2, test_acc, width, label="Test")
    plt.xticks(x, names, rotation=10)
    plt.ylim(0, max(max(val_acc), max(test_acc), 0.1) + 0.05)
    plt.ylabel("Accuracy")
    plt.title("Assignment 2 Accuracy Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "assignment2_accuracy_comparison.png"), dpi=160)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Run Assignment 2 archive experiments.")
    parser.add_argument("--train-size", type=int, default=49000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--conv-filters", type=int, default=8)
    parser.add_argument("--num-train-samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs()
    np.random.seed(42)
    start = time.time()
    append_trace("baseline started")

    numpy_smoke = run_numpy_layer_smoke()
    pytorch_rnn_smoke = run_pytorch_rnn_smoke()
    smoke_tests = [numpy_smoke, pytorch_rnn_smoke]
    append_trace("smoke tests completed")
    data, results = run_cifar_experiments(args)
    elapsed = time.time() - start

    config = {
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "conv_filters": args.conv_filters,
        "num_train_samples": args.num_train_samples,
    }
    log_lines = [
        "Assignment 2 experiment log",
        f"X_train={data['X_train'].shape}, X_val={data['X_val'].shape}, X_test={data['X_test'].shape}",
        f"config={json.dumps(config, ensure_ascii=False)}",
        "smoke_tests=" + format_smoke_tests_for_log(smoke_tests),
    ]
    for item in results:
        log_lines.append(
            "{model}: setting={setting}, train_acc={train_acc:.4f}, "
            "val_acc={val_acc:.4f}, test_acc={test_acc:.4f}, final_loss={final_loss:.4f}".format(
                **item
            )
        )
    log_lines.append(f"elapsed_seconds={elapsed:.2f}")

    summary = {
        "config": config,
        "smoke_tests": smoke_tests,
        "results": results,
        "elapsed_seconds": elapsed,
        "log_lines": log_lines,
    }
    save_metrics(results)
    save_plots(results)
    save_summary(summary)
    append_trace("baseline completed elapsed_seconds=%.2f" % elapsed)

    print("\n".join(log_lines))


if __name__ == "__main__":
    main()
