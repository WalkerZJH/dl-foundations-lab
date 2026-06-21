from __future__ import annotations

import numpy as np

from .convolution import Conv2DVectorized, MaxPool2D
from .layers import BatchNorm1D, Dropout, Flatten, Linear, ReLU
from .module import Sequential


def build_model(config: dict[str, object], seed: int) -> Sequential:
    rng = np.random.default_rng(seed)
    name = str(config["name"])
    input_shape = tuple(int(value) for value in config.get("input_shape", [1, 8, 8]))
    num_classes = int(config.get("num_classes", 10))
    dtype = np.dtype(str(config.get("dtype", "float64")))
    hidden_dim = int(config.get("hidden_dim", 64))
    dropout_keep_ratio = float(config.get("dropout_keep_ratio", 1.0))
    batchnorm = bool(config.get("batchnorm", False))

    if name == "mlp":
        input_features = int(np.prod(input_shape))
        layers = [Flatten(), Linear(input_features, hidden_dim, rng, dtype=dtype)]
        if batchnorm:
            layers.append(
                BatchNorm1D(
                    hidden_dim,
                    momentum=float(config.get("batchnorm_momentum", 0.9)),
                    dtype=dtype,
                )
            )
        layers.append(ReLU())
        if dropout_keep_ratio < 1.0:
            layers.append(Dropout(dropout_keep_ratio, seed=seed + 10_000))
        layers.append(
            Linear(
                hidden_dim,
                num_classes,
                rng,
                weight_scale=np.sqrt(1.0 / hidden_dim),
                dtype=dtype,
            )
        )
        return Sequential(*layers)

    if name == "cnn":
        filters = int(config.get("num_filters", 8))
        channels, height, width = input_shape
        pooled_height = (height - 2) // 2 + 1
        pooled_width = (width - 2) // 2 + 1
        layers = [
            Conv2DVectorized(
                channels, filters, 3, rng, stride=1, padding=1, dtype=dtype
            ),
            ReLU(),
            MaxPool2D(2, 2),
            Flatten(),
            Linear(filters * pooled_height * pooled_width, hidden_dim, rng, dtype=dtype),
        ]
        if batchnorm:
            layers.append(
                BatchNorm1D(
                    hidden_dim,
                    momentum=float(config.get("batchnorm_momentum", 0.9)),
                    dtype=dtype,
                )
            )
        layers.append(ReLU())
        if dropout_keep_ratio < 1.0:
            layers.append(Dropout(dropout_keep_ratio, seed=seed + 10_000))
        layers.append(
            Linear(
                hidden_dim,
                num_classes,
                rng,
                weight_scale=np.sqrt(1.0 / hidden_dim),
                dtype=dtype,
            )
        )
        return Sequential(*layers)

    raise ValueError(f"Unsupported model: {name}")
