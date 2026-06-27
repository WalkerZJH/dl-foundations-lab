from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\sA-Za-z0-9]")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def tokenize(text: str, lowercase: bool = True) -> list[str]:
    if lowercase:
        text = text.lower()
    return TOKEN_RE.findall(text)


@dataclass
class Vocab:
    token_to_id: dict[str, int]

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    def encode(self, tokens: list[str], max_len: int) -> tuple[list[int], int]:
        ids = [self.token_to_id.get(tok, self.token_to_id[UNK_TOKEN]) for tok in tokens[:max_len]]
        length = max(1, len(ids))
        if len(ids) < max_len:
            ids.extend([self.token_to_id[PAD_TOKEN]] * (max_len - len(ids)))
        return ids, length

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.token_to_id, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "Vocab":
        return cls(json.loads(path.read_text(encoding="utf-8")))


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{"label": int(row["label"]), "text": row["text"]} for row in csv.DictReader(handle)]


def build_vocab(rows: list[dict], vocab_size: int, min_freq: int, lowercase: bool = True) -> Vocab:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(tokenize(row["text"], lowercase=lowercase))
    token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if token in token_to_id:
            continue
        token_to_id[token] = len(token_to_id)
        if len(token_to_id) >= vocab_size:
            break
    return Vocab(token_to_id)


class NewsDataset(Dataset):
    def __init__(self, rows: list[dict], vocab: Vocab, max_seq_len: int, lowercase: bool = True) -> None:
        self.rows = rows
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.lowercase = lowercase

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        ids, length = self.vocab.encode(tokenize(row["text"], self.lowercase), self.max_seq_len)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "lengths": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(row["label"], dtype=torch.long),
        }


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "lengths": torch.stack([item["lengths"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
    }


def make_loaders(config: dict, data_dir: Path, vocab_path: Path | None = None) -> tuple[Vocab, dict[str, DataLoader]]:
    train_rows = read_csv(data_dir / "train.csv")
    val_rows = read_csv(data_dir / "val.csv")
    test_rows = read_csv(data_dir / "test.csv")
    lowercase = config.get("lowercase", True)
    if vocab_path and vocab_path.exists():
        vocab = Vocab.from_json(vocab_path)
    else:
        vocab = build_vocab(train_rows, config["vocab_size"], config["min_freq"], lowercase)
        if vocab_path:
            vocab_path.parent.mkdir(parents=True, exist_ok=True)
            vocab.to_json(vocab_path)
    loaders = {}
    for split, rows in {"train": train_rows, "val": val_rows, "test": test_rows}.items():
        loaders[split] = DataLoader(
            NewsDataset(rows, vocab, config["max_seq_len"], lowercase),
            batch_size=config["batch_size"],
            shuffle=(split == "train"),
            num_workers=config.get("num_workers", 0),
            collate_fn=collate_batch,
        )
    return vocab, loaders
