"""
Simple RNN Classifier for Emotion Classification.
"""
import torch
import torch.nn as nn

class SimpleRNNClassifier(nn.Module):
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
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            nonlinearity='relu'
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        # input_ids: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(input_ids)) # [batch_size, seq_len, embedding_dim]
        out, h_n = self.rnn(embedded) # out: [batch_size, seq_len, hidden_dim], h_n: [num_layers, batch_size, hidden_dim]
        
        # Take the last layer's final hidden state
        feat = h_n[-1] # [batch_size, hidden_dim]
        feat = self.dropout(feat)
        logits = self.fc(feat) # [batch_size, num_classes]
        return logits
