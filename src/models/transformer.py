"""
Pure Scratch Implementation of Transformer Encoder Classifier.
NO nn.MultiheadAttention, NO nn.TransformerEncoder, NO nn.TransformerEncoderLayer, NO scaled_dot_product_attention.
Strictly built using primitive PyTorch layers: nn.Linear, nn.Embedding, nn.LayerNorm, nn.Dropout, and torch operations.
"""
import math
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model) # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)

class MultiHeadAttentionFromScratch(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Query, Key, Value and Output
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):
        batch_size = query.size(0)

        # 1) Project and reshape into heads: [batch_size, num_heads, seq_len, d_k]
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2) Scaled dot-product attention scores: Q @ K^T / sqrt(d_k)
        # Q: [B, H, L_q, d_k], K^T: [B, H, d_k, L_k] -> scores: [B, H, L_q, L_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 3) Apply padding mask if provided
        if mask is not None:
            # mask expected shape: [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)

        # 4) Softmax to obtain attention distribution
        attn_weights = torch.softmax(scores, dim=-1) # [B, H, L_q, L_k]
        attn_weights_dropped = self.dropout(attn_weights)

        # 5) Multiply weights by V: [B, H, L_q, d_k]
        context = torch.matmul(attn_weights_dropped, V)

        # 6) Concatenate heads and project through output linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(context)

        return output, attn_weights

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class TransformerEncoderBlockFromScratch(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttentionFromScratch(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # Multi-Head Attention Sub-layer with residual connection & LayerNorm (Pre-LN)
        norm_x = self.norm1(x)
        attn_out, attn_weights = self.attention(norm_x, norm_x, norm_x, mask=mask)
        x = x + self.dropout1(attn_out)

        # Feed-Forward Sub-layer with residual connection & LayerNorm
        norm_x2 = self.norm2(x)
        ff_out = self.feed_forward(norm_x2)
        x = x + self.dropout2(ff_out)

        return x, attn_weights

class TransformerClassifierFromScratch(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128, # d_model
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        num_classes: int = 28,
        dropout: float = 0.2,
        max_len: int = 50,
        pad_idx: int = 0
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.pos_encoder = SinusoidalPositionalEncoding(embedding_dim, max_len=max_len, dropout=dropout)
        
        self.layers = nn.ModuleList([
            TransformerEncoderBlockFromScratch(
                d_model=embedding_dim,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)

        self.last_attention_weights = None

    def create_padding_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 1 for valid tokens, 0 for pad tokens
        # Shape: [batch_size, 1, 1, seq_len]
        return (input_ids != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor = None, return_attention: bool = False):
        batch_size, seq_len = input_ids.shape
        mask = self.create_padding_mask(input_ids) # [B, 1, 1, L]

        # Embeddings + Positional Encodings (scaled by sqrt(d_model) as in Vaswani et al.)
        x = self.embedding(input_ids) * math.sqrt(self.embedding_dim)
        x = self.pos_encoder(x)

        all_attentions = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask=mask)
            all_attentions.append(attn_weights)

        x = self.final_norm(x) # [batch_size, seq_len, d_model]
        self.last_attention_weights = all_attentions[-1] # [B, num_heads, seq_len, seq_len]

        # Readout: Mean pooling over non-padded tokens
        token_mask = (input_ids != self.pad_idx).unsqueeze(-1).float() # [B, seq_len, 1]
        sum_embeddings = (x * token_mask).sum(dim=1)
        lengths_clamped = token_mask.sum(dim=1).clamp(min=1.0)
        pooled = sum_embeddings / lengths_clamped

        logits = self.fc(self.dropout(pooled))

        if return_attention:
            return logits, self.last_attention_weights
        return logits
