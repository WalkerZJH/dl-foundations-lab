from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Parameter:
    data: np.ndarray
    grad: np.ndarray

    @classmethod
    def from_array(
        cls, data: np.ndarray, dtype: np.dtype[Any] | type[np.generic] | None = None
    ) -> "Parameter":
        array = np.asarray(data, dtype=dtype)
        return cls(array, np.zeros_like(array))


class Module:
    def __init__(self) -> None:
        self.training = True

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return self.forward(inputs)

    def _children(self) -> Iterator[tuple[str, "Module"]]:
        for name, value in self.__dict__.items():
            if isinstance(value, Module):
                yield name, value
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    if isinstance(item, Module):
                        yield f"{name}.{index}", item

    def named_parameters(self, prefix: str = "") -> dict[str, Parameter]:
        parameters: dict[str, Parameter] = {}
        for name, value in self.__dict__.items():
            if isinstance(value, Parameter):
                key = f"{prefix}{name}" if prefix else name
                parameters[key] = value
        for child_name, child in self._children():
            child_prefix = f"{prefix}{child_name}." if prefix else f"{child_name}."
            parameters.update(child.named_parameters(child_prefix))
        return parameters

    def local_buffers(self) -> dict[str, np.ndarray]:
        return {}

    def named_buffers(self, prefix: str = "") -> dict[str, np.ndarray]:
        buffers = {
            f"{prefix}{name}" if prefix else name: value
            for name, value in self.local_buffers().items()
        }
        for child_name, child in self._children():
            child_prefix = f"{prefix}{child_name}." if prefix else f"{child_name}."
            buffers.update(child.named_buffers(child_prefix))
        return buffers

    def state_dict(self) -> dict[str, np.ndarray]:
        state = {
            f"param::{name}": parameter.data.copy()
            for name, parameter in self.named_parameters().items()
        }
        state.update(
            {f"buffer::{name}": value.copy() for name, value in self.named_buffers().items()}
        )
        return state

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        parameters = self.named_parameters()
        buffers = self.named_buffers()
        expected = {f"param::{name}" for name in parameters} | {
            f"buffer::{name}" for name in buffers
        }
        if set(state) != expected:
            missing = sorted(expected - set(state))
            extra = sorted(set(state) - expected)
            raise ValueError(f"State mismatch: missing={missing}, extra={extra}")
        for name, parameter in parameters.items():
            parameter.data[...] = state[f"param::{name}"]
        for name, buffer in buffers.items():
            buffer[...] = state[f"buffer::{name}"]

    def train(self) -> "Module":
        self.training = True
        for _, child in self._children():
            child.train()
        return self

    def eval(self) -> "Module":
        self.training = False
        for _, child in self._children():
            child.eval()
        return self

    def zero_grad(self) -> None:
        for parameter in self.named_parameters().values():
            parameter.grad.fill(0.0)

    def parameter_count(self) -> int:
        return int(sum(parameter.data.size for parameter in self.named_parameters().values()))


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self.layers = list(layers)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        output = inputs
        for layer in self.layers:
            output = layer(output)
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad


def copy_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: value.copy() for name, value in state.items()}


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value
