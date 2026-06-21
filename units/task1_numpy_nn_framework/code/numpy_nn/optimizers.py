from __future__ import annotations

from typing import Any

import numpy as np

from .module import Parameter


class Optimizer:
    def __init__(self, parameters: dict[str, Parameter], learning_rate: float) -> None:
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self) -> None:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        return {"learning_rate": self.learning_rate}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.learning_rate = float(state["learning_rate"])


class SGD(Optimizer):
    def __init__(
        self,
        parameters: dict[str, Parameter],
        learning_rate: float,
        momentum: float = 0.0,
    ) -> None:
        super().__init__(parameters, learning_rate)
        self.momentum = momentum
        self.velocity = {
            name: np.zeros_like(parameter.data) for name, parameter in parameters.items()
        }

    def step(self) -> None:
        for name, parameter in self.parameters.items():
            self.velocity[name] *= self.momentum
            self.velocity[name] -= self.learning_rate * parameter.grad
            parameter.data += self.velocity[name]

    def state_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "velocity": {name: value.copy() for name, value in self.velocity.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.momentum = float(state["momentum"])
        for name, value in state["velocity"].items():
            self.velocity[name][...] = value


class Adam(Optimizer):
    def __init__(
        self,
        parameters: dict[str, Parameter],
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(parameters, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step_count = 0
        self.first_moment = {
            name: np.zeros_like(parameter.data) for name, parameter in parameters.items()
        }
        self.second_moment = {
            name: np.zeros_like(parameter.data) for name, parameter in parameters.items()
        }

    def step(self) -> None:
        self.step_count += 1
        for name, parameter in self.parameters.items():
            grad = parameter.grad
            self.first_moment[name] = self.beta1 * self.first_moment[name] + (1.0 - self.beta1) * grad
            self.second_moment[name] = (
                self.beta2 * self.second_moment[name] + (1.0 - self.beta2) * grad**2
            )
            corrected_first = self.first_moment[name] / (1.0 - self.beta1**self.step_count)
            corrected_second = self.second_moment[name] / (1.0 - self.beta2**self.step_count)
            parameter.data -= self.learning_rate * corrected_first / (
                np.sqrt(corrected_second) + self.eps
            )

    def state_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "step_count": self.step_count,
            "first_moment": {
                name: value.copy() for name, value in self.first_moment.items()
            },
            "second_moment": {
                name: value.copy() for name, value in self.second_moment.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.beta1 = float(state["beta1"])
        self.beta2 = float(state["beta2"])
        self.eps = float(state["eps"])
        self.step_count = int(state["step_count"])
        for name, value in state["first_moment"].items():
            self.first_moment[name][...] = value
        for name, value in state["second_moment"].items():
            self.second_moment[name][...] = value


def build_optimizer(
    name: str,
    parameters: dict[str, Parameter],
    learning_rate: float,
    momentum: float = 0.9,
) -> Optimizer:
    if name == "sgd":
        return SGD(parameters, learning_rate)
    if name == "momentum":
        return SGD(parameters, learning_rate, momentum=momentum)
    if name == "adam":
        return Adam(parameters, learning_rate)
    raise ValueError(f"Unsupported optimizer: {name}")
