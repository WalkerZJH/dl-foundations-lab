from __future__ import annotations

import numpy as np


class SoftmaxCrossEntropy:
    def __init__(self) -> None:
        self._probabilities: np.ndarray | None = None
        self._targets: np.ndarray | None = None

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        self._probabilities = probabilities
        self._targets = targets
        correct = probabilities[np.arange(targets.size), targets]
        return float(-np.mean(np.log(np.clip(correct, 1e-12, None))))

    def backward(self) -> np.ndarray:
        if self._probabilities is None or self._targets is None:
            raise RuntimeError("SoftmaxCrossEntropy.backward requires a preceding forward call")
        grad = self._probabilities.copy()
        grad[np.arange(self._targets.size), self._targets] -= 1.0
        return grad / self._targets.size


def softmax_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_partition = np.log(np.sum(np.exp(shifted), axis=1))
    correct = shifted[np.arange(targets.size), targets]
    return float(np.mean(log_partition - correct))
