"""
GRU Classifier for Emotion Classification.
"""
import torch
import torch.nn as nn

class GRUClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 28,
        num_layers: int = 1,
        dropout: float = 0.3,
        pad_idx: int = 0
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        embedded = self.dropout(self.embedding(input_ids))
        out, h_n = self.gru(embedded)
        feat = h_n[-1]
        feat = self.dropout(feat)
        logits = self.fc(feat)
        return logits
