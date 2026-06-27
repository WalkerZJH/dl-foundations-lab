"""Download and export AG-News for Task2.

The script keeps the local dataset under units/data/ag_news and the Hugging Face
cache under units/data/.hf_cache. Exported labels follow Hugging Face AG-News:
0=World, 1=Sports, 2=Business, 3=Sci/Tech.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


LABEL_MAPPING = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Hugging Face AG-News to CSV.")
    parser.add_argument("--out-dir", default="units/data/ag_news")
    parser.add_argument("--cache-dir", default="units/data/.hf_cache")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def to_frame(split) -> pd.DataFrame:
    frame = pd.DataFrame({"label": split["label"], "text": split["text"]})
    frame["label"] = frame["label"].astype(int)
    frame["text"] = frame["text"].astype(str)
    return frame[["label", "text"]]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("ag_news", cache_dir=str(cache_dir))
    train_full = to_frame(dataset["train"])
    test = to_frame(dataset["test"])

    train, val = train_test_split(
        train_full,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=train_full["label"],
    )
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    metadata = {
        "dataset": "AG-News",
        "source": "huggingface/datasets: load_dataset('ag_news')",
        "label_mapping": {str(k): v for k, v in LABEL_MAPPING.items()},
        "label_index_base": 0,
        "official_splits": {"train": int(len(train_full)), "test": int(len(test))},
        "exported_splits": {
            "train": int(len(train)),
            "val": int(len(val)),
            "test": int(len(test)),
        },
        "validation_split": {
            "created_from": "official train split",
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "stratified_by": "label",
        },
        "columns": ["label", "text"],
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata["exported_splits"], ensure_ascii=False))


if __name__ == "__main__":
    main()
