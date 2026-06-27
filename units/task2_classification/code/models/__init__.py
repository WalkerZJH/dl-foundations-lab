"""Model registry for Task2 AG-News experiments."""

from .lstm import LSTMClassifier
from .mlp import MLPClassifier
from .textcnn import TextCNNClassifier
from .stage2_models import (
    BiLSTMAttentionClassifier,
    FastTextClassifier,
    RCNNClassifier,
    TransformerEncoderClassifier,
)


def build_model(config: dict, vocab_size: int, num_classes: int = 4):
    model_name = config["model"]
    if model_name == "mlp":
        return MLPClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            dropout=config["dropout"],
            padding_idx=config.get("padding_idx", 0),
        )
    if model_name == "textcnn":
        return TextCNNClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embedding_dim=config["embedding_dim"],
            num_filters=config["num_filters"],
            kernel_sizes=config["kernel_sizes"],
            dropout=config["dropout"],
            padding_idx=config.get("padding_idx", 0),
        )
    if model_name in {"lstm", "bilstm"}:
        return LSTMClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config.get("num_layers", 1),
            bidirectional=model_name == "bilstm" or config.get("bidirectional", False),
            dropout=config["dropout"],
            padding_idx=config.get("padding_idx", 0),
        )
    if model_name == "fasttext":
        return FastTextClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embedding_dim=config["embedding_dim"],
            dropout=config["dropout"],
            padding_idx=config.get("padding_idx", 0),
        )
    if model_name == "bilstm_attention":
        return BiLSTMAttentionClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            dropout=config["dropout"],
            padding_idx=config.get("padding_idx", 0),
        )
    if model_name == "rcnn":
        return RCNNClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            dropout=config["dropout"],
            padding_idx=config.get("padding_idx", 0),
        )
    if model_name == "transformer_encoder":
        return TransformerEncoderClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embedding_dim=config["embedding_dim"],
            max_seq_len=config["max_seq_len"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            ffn_dim=config["ffn_dim"],
            dropout=config["dropout"],
            padding_idx=config.get("padding_idx", 0),
        )
    raise ValueError(f"Unknown model: {model_name}")
