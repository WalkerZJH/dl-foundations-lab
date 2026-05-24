from builtins import range
import numpy as np

"""cs231n layers

This module implements forward and backward passes for common neural network
layers (affine, ReLU, batchnorm, dropout, convolution, pooling, etc.).

中文说明：
该模块实现了常用神经网络层的前向与反向传播函数（仿射、ReLU、批量归一化、
dropout、卷积、池化等），用于构建和训练神经网络。
"""


def affine_forward(x, w, b):
    out = x.reshape(x.shape[0], -1).dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    x, w, b = cache
    x_flat = x.reshape(x.shape[0], -1)
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x_flat.T.dot(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    x = cache
    dx = dout * (x > 0)
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    if mode == "train":
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        x_centered = x - sample_mean
        std = np.sqrt(sample_var + eps)
        x_norm = x_centered / std
        out = gamma * x_norm + beta
        cache = (x_norm, gamma, x_centered, std, sample_var, eps)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
    elif mode == "test":
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta
        cache = None
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var
    return out, cache


def batchnorm_backward(dout, cache):
    x_norm, gamma, x_centered, std, sample_var, eps = cache
    N = dout.shape[0]

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)
    dx_norm = dout * gamma
    dvar = np.sum(dx_norm * x_centered * -0.5 * (sample_var + eps) ** -1.5, axis=0)
    dmean = np.sum(dx_norm * -1.0 / std, axis=0)
    dmean += dvar * np.mean(-2.0 * x_centered, axis=0)
    dx = dx_norm / std + dvar * 2.0 * x_centered / N + dmean / N
    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    x_norm, gamma, x_centered, std, sample_var, eps = cache
    N = dout.shape[0]

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)
    dx_norm = dout * gamma
    dx = (1.0 / N) / std * (
        N * dx_norm - np.sum(dx_norm, axis=0) - x_norm * np.sum(dx_norm * x_norm, axis=0)
    )
    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    eps = ln_param.get("eps", 1e-5)
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    out = gamma * x_norm + beta
    cache = (x_norm, gamma, np.sqrt(var + eps))
    return out, cache


def layernorm_backward(dout, cache):
    x_norm, gamma, std = cache
    D = dout.shape[1]

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)
    dx_norm = dout * gamma
    dx = (1.0 / D) / std * (
        D * dx_norm
        - np.sum(dx_norm, axis=1, keepdims=True)
        - x_norm * np.sum(dx_norm * x_norm, axis=1, keepdims=True)
    )
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    if mode == "train":
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
    elif mode == "test":
        out = x
    else:
        raise ValueError('Invalid dropout mode "%s"' % mode)

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)
    return out, cache


def dropout_backward(dout, cache):
    dropout_param, mask = cache
    mode = dropout_param["mode"]
    if mode == "train":
        dx = dout * mask
    elif mode == "test":
        dx = dout
    else:
        raise ValueError('Invalid dropout mode "%s"' % mode)
    return dx


def conv_forward_naive(x, w, b, conv_param):
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param["stride"], conv_param["pad"]
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    out = np.zeros((N, F, H_out, W_out), dtype=x.dtype)
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                h0 = i * stride
                for j in range(W_out):
                    w0 = j * stride
                    window = x_pad[n, :, h0:h0 + HH, w0:w0 + WW]
                    out[n, f, i, j] = np.sum(window * w[f]) + b[f]
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param["stride"], conv_param["pad"]
    _, _, H_out, W_out = dout.shape

    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.sum(dout, axis=(0, 2, 3))

    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                h0 = i * stride
                for j in range(W_out):
                    w0 = j * stride
                    window = x_pad[n, :, h0:h0 + HH, w0:w0 + WW]
                    dw[f] += window * dout[n, f, i, j]
                    dx_pad[n, :, h0:h0 + HH, w0:w0 + WW] += w[f] * dout[n, f, i, j]

    dx = dx_pad[:, :, pad:pad + H, pad:pad + W] if pad > 0 else dx_pad
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    N, C, H, W = x.shape
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride

    out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                h0 = i * stride
                for j in range(W_out):
                    w0 = j * stride
                    window = x[n, c, h0:h0 + pool_height, w0:w0 + pool_width]
                    out[n, c, i, j] = np.max(window)
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]
    _, _, H_out, W_out = dout.shape
    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                h0 = i * stride
                for j in range(W_out):
                    w0 = j * stride
                    window = x[n, c, h0:h0 + pool_height, w0:w0 + pool_width]
                    mask = window == np.max(window)
                    dx[n, c, h0:h0 + pool_height, w0:w0 + pool_width] += mask * dout[n, c, i, j]
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    N, C, H, W = x.shape
    x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)
    out_flat, cache = batchnorm_forward(x_flat, gamma, beta, bn_param)
    out = out_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return out, cache


def spatial_batchnorm_backward(dout, cache):
    N, C, H, W = dout.shape
    dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    dx_flat, dgamma, dbeta = batchnorm_backward_alt(dout_flat, cache)
    dx = dx_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    eps = gn_param.get("eps", 1e-5)
    N, C, H, W = x.shape
    x_group = x.reshape(N, G, C // G, H, W)
    mean = np.mean(x_group, axis=(2, 3, 4), keepdims=True)
    var = np.var(x_group, axis=(2, 3, 4), keepdims=True)
    x_norm_group = (x_group - mean) / np.sqrt(var + eps)
    x_norm = x_norm_group.reshape(N, C, H, W)
    out = gamma * x_norm + beta
    cache = (G, x_norm_group, gamma, var, eps)
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    G, x_norm_group, gamma, var, eps = cache
    N, C, H, W = dout.shape
    group_size = C // G * H * W
    x_norm = x_norm_group.reshape(N, C, H, W)

    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    dgamma = np.sum(dout * x_norm, axis=(0, 2, 3), keepdims=True)
    dx_norm = (dout * gamma).reshape(N, G, C // G, H, W)
    dx_group = (1.0 / group_size) / np.sqrt(var + eps) * (
        group_size * dx_norm
        - np.sum(dx_norm, axis=(2, 3, 4), keepdims=True)
        - x_norm_group * np.sum(dx_norm * x_norm_group, axis=(2, 3, 4), keepdims=True)
    )
    dx = dx_group.reshape(N, C, H, W)
    return dx, dgamma, dbeta


def svm_loss(x, y):
    N = x.shape[0]
    correct_scores = x[np.arange(N), y][:, np.newaxis]
    margins = np.maximum(0.0, x - correct_scores + 1.0)
    margins[np.arange(N), y] = 0.0
    loss = np.sum(margins) / N

    dx = (margins > 0).astype(float)
    dx[np.arange(N), y] = -np.sum(dx, axis=1)
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(np.maximum(probs[np.arange(N), y], 1e-12))) / N

    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
