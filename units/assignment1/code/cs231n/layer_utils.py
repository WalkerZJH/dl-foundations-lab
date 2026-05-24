from .layers import *
"""Layer convenience functions.

This file provides convenience wrappers that combine common sequences of
layers into single forward/backward functions (e.g., affine -> ReLU).

中文说明：
提供层组合的便捷函数，用于将常见的层序列（例如 仿射 -> ReLU）合并为一个前向/后向函数。
"""


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    中文说明：
    将仿射变换（affine）和 ReLU 激活合并为一个便利层。

    输入：
    - x：仿射层的输入
    - w, b：仿射层的权重和偏置

    返回：
    - out：ReLU 的输出
    - cache：用于反向传播的缓存对象
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    """
    Backward pass for the affine-relu convenience layer

    中文说明：
    仿射-ReLU 组合层的反向传播。
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

