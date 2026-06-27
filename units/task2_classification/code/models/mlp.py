from __future__ import annotations

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """Embedding + masked mean pooling + MLP baseline."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        mask = (input_ids != 0).unsqueeze(-1)
        summed = (embedded * mask).sum(dim=1)
        denom = lengths.clamp_min(1).unsqueeze(1).to(embedded.dtype)
        pooled = summed / denom
        return self.classifier(pooled)
