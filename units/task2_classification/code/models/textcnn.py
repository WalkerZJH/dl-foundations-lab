from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TextCNNClassifier(nn.Module):
    """TextCNN with multiple temporal convolution widths."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int,
        num_filters: int,
        kernel_sizes: list[int],
        dropout: float,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList(
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k) for k in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids).transpose(1, 2)
        features = []
        for conv in self.convs:
            activated = F.relu(conv(embedded))
            pooled = F.max_pool1d(activated, kernel_size=activated.size(2)).squeeze(2)
            features.append(pooled)
        return self.fc(self.dropout(torch.cat(features, dim=1)))
