from __future__ import annotations

import numpy as np

from .module import Module, Parameter


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rng: np.random.Generator,
        weight_scale: float | None = None,
        dtype: np.dtype = np.dtype(np.float64),
    ) -> None:
        super().__init__()
        scale = weight_scale if weight_scale is not None else np.sqrt(2.0 / in_features)
        self.weight = Parameter.from_array(
            rng.normal(0.0, scale, (in_features, out_features)), dtype=dtype
        )
        self.bias = Parameter.from_array(np.zeros(out_features, dtype=dtype))
        self._inputs: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._inputs = inputs
        return inputs @ self.weight.data + self.bias.data

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._inputs is None:
            raise RuntimeError("Linear.backward requires a preceding forward call")
        self.weight.grad[...] = self._inputs.T @ grad_output
        self.bias.grad[...] = grad_output.sum(axis=0)
        return grad_output @ self.weight.data.T


class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()
        self._positive: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._positive = inputs > 0.0
        return np.maximum(inputs, 0.0)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._positive is None:
            raise RuntimeError("ReLU.backward requires a preceding forward call")
        return grad_output * self._positive


class Flatten(Module):
    def __init__(self) -> None:
        super().__init__()
        self._input_shape: tuple[int, ...] | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._input_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._input_shape is None:
            raise RuntimeError("Flatten.backward requires a preceding forward call")
        return grad_output.reshape(self._input_shape)


class BatchNorm1D(Module):
    def __init__(
        self,
        features: int,
        momentum: float = 0.9,
        eps: float = 1e-5,
        dtype: np.dtype = np.dtype(np.float64),
    ) -> None:
        super().__init__()
        self.gamma = Parameter.from_array(np.ones(features, dtype=dtype))
        self.beta = Parameter.from_array(np.zeros(features, dtype=dtype))
        self.running_mean = np.zeros(features, dtype=dtype)
        self.running_var = np.ones(features, dtype=dtype)
        self.momentum = momentum
        self.eps = eps
        self._cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def local_buffers(self) -> dict[str, np.ndarray]:
        return {"running_mean": self.running_mean, "running_var": self.running_var}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if self.training:
            mean = inputs.mean(axis=0)
            var = inputs.var(axis=0)
            inv_std = 1.0 / np.sqrt(var + self.eps)
            normalized = (inputs - mean) * inv_std
            self.running_mean[...] = (
                self.momentum * self.running_mean + (1.0 - self.momentum) * mean
            )
            self.running_var[...] = (
                self.momentum * self.running_var + (1.0 - self.momentum) * var
            )
            self._cache = (inputs - mean, normalized, inv_std)
        else:
            normalized = (inputs - self.running_mean) / np.sqrt(
                self.running_var + self.eps
            )
            self._cache = None
        return self.gamma.data * normalized + self.beta.data

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("BatchNorm1D.backward requires a training-mode forward call")
        centered, normalized, inv_std = self._cache
        batch_size = grad_output.shape[0]
        self.gamma.grad[...] = np.sum(grad_output * normalized, axis=0)
        self.beta.grad[...] = np.sum(grad_output, axis=0)
        grad_normalized = grad_output * self.gamma.data
        return (
            inv_std
            / batch_size
            * (
                batch_size * grad_normalized
                - np.sum(grad_normalized, axis=0)
                - centered
                * inv_std**2
                * np.sum(grad_normalized * centered, axis=0)
            )
        )


class Dropout(Module):
    def __init__(self, keep_ratio: float, seed: int | None = None) -> None:
        super().__init__()
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")
        self.keep_ratio = keep_ratio
        self.rng = np.random.default_rng(seed)
        self._mask: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if not self.training or self.keep_ratio == 1.0:
            self._mask = None
            return inputs
        self._mask = (self.rng.random(inputs.shape) < self.keep_ratio) / self.keep_ratio
        return inputs * self._mask

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if not self.training or self.keep_ratio == 1.0:
            return grad_output
        if self._mask is None:
            raise RuntimeError("Dropout.backward requires a training-mode forward call")
        return grad_output * self._mask
