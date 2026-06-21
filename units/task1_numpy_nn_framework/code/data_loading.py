from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np


@dataclass(frozen=True)
class DatasetSplit:
    images: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class ClassificationData:
    train: DatasetSplit
    validation: DatasetSplit
    test: DatasetSplit
    source: str


def load_digits_splits(
    seed: int = 42,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> ClassificationData:
    try:
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise RuntimeError(
            "Task1 data loading requires scikit-learn in the selected environment."
        ) from exc

    if train_ratio <= 0.0 or validation_ratio <= 0.0:
        raise ValueError("train_ratio and validation_ratio must be positive")
    test_ratio = 1.0 - train_ratio - validation_ratio
    if test_ratio <= 0.0:
        raise ValueError("train_ratio + validation_ratio must be less than 1")

    digits = load_digits()
    images = np.asarray(digits.images, dtype=np.float64)[:, None, :, :] / 16.0
    labels = np.asarray(digits.target, dtype=np.int64)

    train_images, remainder_images, train_labels, remainder_labels = train_test_split(
        images,
        labels,
        train_size=train_ratio,
        random_state=seed,
        stratify=labels,
    )
    validation_fraction = validation_ratio / (validation_ratio + test_ratio)
    validation_images, test_images, validation_labels, test_labels = train_test_split(
        remainder_images,
        remainder_labels,
        train_size=validation_fraction,
        random_state=seed + 1,
        stratify=remainder_labels,
    )
    return ClassificationData(
        train=DatasetSplit(train_images, train_labels),
        validation=DatasetSplit(validation_images, validation_labels),
        test=DatasetSplit(test_images, test_labels),
        source="scikit-learn load_digits (1797 samples, 8x8 grayscale)",
    )


def load_cifar10_splits(
    dataset_dir: Path,
    seed: int = 42,
    validation_ratio: float = 0.1,
) -> ClassificationData:
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise RuntimeError(
            "Task1 CIFAR-10 splitting requires scikit-learn in the selected environment."
        ) from exc

    def load_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
        with path.open("rb") as handle:
            payload = pickle.load(handle, encoding="latin1")
        images = np.asarray(payload["data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
        labels = np.asarray(payload["labels"], dtype=np.int64)
        return images, labels

    required = [dataset_dir / f"data_batch_{index}" for index in range(1, 6)]
    required.append(dataset_dir / "test_batch")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing CIFAR-10 batch files: {missing}")

    train_batches = [load_batch(path) for path in required[:5]]
    raw_train_images = np.concatenate([batch[0] for batch in train_batches], axis=0)
    raw_train_labels = np.concatenate([batch[1] for batch in train_batches], axis=0)
    raw_test_images, test_labels = load_batch(required[5])
    indices = np.arange(raw_train_labels.size)
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_ratio,
        random_state=seed,
        stratify=raw_train_labels,
    )

    train_images = raw_train_images[train_indices].astype(np.float32)
    validation_images = raw_train_images[validation_indices].astype(np.float32)
    test_images = raw_test_images.astype(np.float32)
    train_images /= 255.0
    validation_images /= 255.0
    test_images /= 255.0
    return ClassificationData(
        train=DatasetSplit(train_images, raw_train_labels[train_indices]),
        validation=DatasetSplit(validation_images, raw_train_labels[validation_indices]),
        test=DatasetSplit(test_images, test_labels),
        source="CIFAR-10 python batches (50000 train/validation, 10000 test)",
    )


def describe_splits(data: ClassificationData) -> dict[str, object]:
    result: dict[str, object] = {"source": data.source, "total_samples": 0}
    total = 0
    for name, split in (
        ("train", data.train),
        ("validation", data.validation),
        ("test", data.test),
    ):
        counts = np.bincount(split.labels, minlength=10)
        result[name] = {
            "samples": int(split.labels.size),
            "class_counts": counts.tolist(),
            "image_shape": list(split.images.shape[1:]),
            "value_min": float(split.images.min()),
            "value_max": float(split.images.max()),
        }
        total += int(split.labels.size)
    result["total_samples"] = total
    return result
