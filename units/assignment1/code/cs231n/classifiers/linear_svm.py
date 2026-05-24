from builtins import range
import numpy as np

"""Structured SVM loss implementations.

This module provides naive and vectorized implementations of the structured
SVM loss and its gradient.

中文说明：
提供结构化 SVM 损失及其梯度的朴素实现和向量化实现。
"""


def svm_loss_naive(W, X, y, reg):
    """Structured SVM loss function, naive implementation.

    中文说明：
    朴素实现（含循环）的多类 SVM 损失与梯度计算。
    """
    dW = np.zeros(W.shape)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1.0
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """Structured SVM loss function, vectorized implementation.

    中文说明：
    向量化实现的多类 SVM 损失与梯度计算，效率更高。
    """
    num_train = X.shape[0]
    scores = X.dot(W)
    correct_scores = scores[np.arange(num_train), y][:, np.newaxis]
    margins = np.maximum(0.0, scores - correct_scores + 1.0)
    margins[np.arange(num_train), y] = 0.0

    loss = np.sum(margins) / num_train
    loss += reg * np.sum(W * W)

    binary = (margins > 0).astype(float)
    row_sum = np.sum(binary, axis=1)
    binary[np.arange(num_train), y] = -row_sum
    dW = X.T.dot(binary) / num_train
    dW += 2 * reg * W
    return loss, dW
