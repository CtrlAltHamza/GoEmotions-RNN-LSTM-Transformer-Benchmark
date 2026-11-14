"""
LSTM Classifier for Emotion Classification.
Supports:
- Forward-only vs Bidirectional
- Readout modes: 'last', 'avg' (mean pooling), 'max' (max pooling)
"""
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 28,
        num_layers: int = 1,
        bidirectional: bool = False,
        readout: str = 'last', # 'last', 'avg', 'max'
        dropout: float = 0.3,
        pad_idx: int = 0
    ):
        super().__init__()
        self.readout = readout.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        
        feature_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        embedded = self.dropout(self.embedding(input_ids)) # [batch, seq_len, emb_dim]
        out, (h_n, c_n) = self.lstm(embedded) # out: [batch, seq_len, num_dirs * hidden_dim]
        
        if self.readout == 'last':
            if not self.bidirectional:
                # out: [batch, seq_len, hidden_dim]
                # If lengths are provided, index the exact last valid token, else take final hidden state
                if lengths is not None:
                    batch_size = input_ids.size(0)
                    idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, out.size(2))
                    feat = out.gather(1, idx).squeeze(1)
                else:
                    feat = h_n[-1]
            else:
                # Concatenate the forward last state and backward last state
                # h_n shape: [num_layers * 2, batch, hidden_dim]
                feat = torch.cat([h_n[-2], h_n[-1]], dim=-1) # [batch, 2 * hidden_dim]
        elif self.readout in ['avg', 'mean']:
            # Mask out padding tokens from mean
            mask = (input_ids != self.pad_idx).unsqueeze(-1).float() # [batch, seq_len, 1]
            masked_out = out * mask
            sum_out = masked_out.sum(dim=1)
            lengths_tensor = mask.sum(dim=1).clamp(min=1.0)
            feat = sum_out / lengths_tensor # [batch, feature_dim]
        elif self.readout == 'max':
            # Set padding positions to -infinity before max pooling
            mask = (input_ids == self.pad_idx).unsqueeze(-1) # [batch, seq_len, 1]
            masked_out = out.masked_fill(mask, -1e9)
            feat = masked_out.max(dim=1)[0] # [batch, feature_dim]
        else:
            raise ValueError(f"Unknown readout strategy: {self.readout}")

        feat = self.dropout(feat)
        logits = self.fc(feat)
        return logits
