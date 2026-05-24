import numpy as np

"""Optimization update rules.

This file contains implementations of optimization update rules used for
training models: SGD, SGD with momentum, RMSProp, and Adam.

中文说明：
实现常用的参数更新规则，用于训练模型：SGD、带动量的 SGD、RMSProp 和 Adam。
"""


def sgd(w, dw, config=None):
    """Stochastic gradient descent update.

    中文说明：
    使用简单的 SGD 更新参数：w <- w - learning_rate * dw。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    next_w = w - config["learning_rate"] * dw
    return next_w, config


def sgd_momentum(w, dw, config=None):
    """SGD with momentum.

    中文说明：
    在 SGD 的基础上加入动量项以加速收敛并减少震荡。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    v = config["momentum"] * v - config["learning_rate"] * dw
    next_w = w + v
    config["velocity"] = v
    return next_w, config


def rmsprop(w, dw, config=None):
    """RMSProp update rule.

    中文说明：
    使用 RMSProp 自适应学习率方法，维护每个参数的平方梯度累积（cache）。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    config["cache"] = (
        config["decay_rate"] * config["cache"] + (1 - config["decay_rate"]) * dw * dw
    )
    next_w = w - config["learning_rate"] * dw / (
        np.sqrt(config["cache"]) + config["epsilon"]
    )
    return next_w, config


def adam(w, dw, config=None):
    """Adam update rule.

    中文说明：
    Adam 结合了动量和 RMSProp 的优点，使用一阶动量 `m` 和二阶动量 `v`，
    并进行偏差校正。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    config["t"] += 1
    config["m"] = config["beta1"] * config["m"] + (1 - config["beta1"]) * dw
    config["v"] = config["beta2"] * config["v"] + (1 - config["beta2"]) * (dw * dw)
    m_unbias = config["m"] / (1 - config["beta1"] ** config["t"])
    v_unbias = config["v"] / (1 - config["beta2"] ** config["t"])
    next_w = w - config["learning_rate"] * m_unbias / (
        np.sqrt(v_unbias) + config["epsilon"]
    )
    return next_w, config
