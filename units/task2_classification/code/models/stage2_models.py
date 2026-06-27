from __future__ import annotations

import math

import torch
from torch import nn


class FastTextClassifier(nn.Module):
    """Embedding bag style classifier with masked mean pooling."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int,
        dropout: float,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        mask = (input_ids != 0).unsqueeze(-1)
        pooled = (embedded * mask).sum(dim=1) / lengths.clamp_min(1).unsqueeze(1).to(embedded.dtype)
        return self.fc(self.dropout(pooled))


class BiLSTMAttentionClassifier(nn.Module):
    """BiLSTM with additive attention pooling over valid tokens."""

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
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=input_ids.size(1)
        )
        scores = self.attn(torch.tanh(output)).squeeze(-1)
        scores = scores.masked_fill(input_ids == 0, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (output * weights).sum(dim=1)
        return self.fc(self.dropout(pooled))


class RCNNClassifier(nn.Module):
    """Recurrent convolutional classifier: BiLSTM context + max pooling."""

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
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(embedding_dim + hidden_dim * 2, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=input_ids.size(1)
        )
        features = torch.tanh(self.proj(torch.cat([embedded, output], dim=-1)))
        features = features.masked_fill((input_ids == 0).unsqueeze(-1), -1e9)
        pooled = features.max(dim=1).values
        return self.fc(self.dropout(pooled))


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerEncoderClassifier(nn.Module):
    """Small Transformer encoder trained from scratch."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int,
        max_seq_len: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.positional = PositionalEncoding(embedding_dim, max_seq_len)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = input_ids == 0
        embedded = self.positional(self.embedding(input_ids))
        encoded = self.encoder(embedded, src_key_padding_mask=mask)
        valid = (~mask).unsqueeze(-1)
        pooled = (encoded * valid).sum(dim=1) / lengths.clamp_min(1).unsqueeze(1).to(encoded.dtype)
        return self.fc(self.dropout(pooled))
