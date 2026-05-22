"""
Transformer & Multi-Head Self-Attention (PyTorch)
===================================================
The architecture behind GPT, BERT, and modern LLMs — built from the
attention mechanism up.

Concepts:
  - Scaled dot-product attention: the core operation
  - Multi-head attention: parallel attention "perspectives"
  - Positional encoding: injecting sequence order
  - Transformer encoder block: attention + FFN + residual + norm
  - Causal masking: for autoregressive generation
  - Why attention beats RNNs: parallelism + long-range dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# ── Scaled dot-product attention ──────────────────────────────────────────────
def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    mask: Optional[torch.Tensor] = None, dropout: Optional[nn.Module] = None,
) -> tuple:
    """
    The fundamental attention operation.

    Attention(Q,K,V) = softmax(QKᵀ / √d_k) V

    Intuition: for each query, compute similarity to all keys, normalise
    to weights, then take a weighted sum of values. The √d_k scaling
    prevents the dot products from growing too large (which would push
    softmax into regions with vanishing gradients).

    Args:
        Q: (batch, heads, seq_q, d_k) — queries
        K: (batch, heads, seq_k, d_k) — keys
        V: (batch, heads, seq_k, d_v) — values
        mask: optional mask (1 = attend, 0 = ignore)

    Returns:
        (output, attention_weights)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-1e9"))

    attn_weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn_weights = dropout(attn_weights)

    output = torch.matmul(attn_weights, V)
    return output, attn_weights


# ── Multi-head attention ──────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention: run attention h times in parallel with different
    learned projections, then concatenate. Each head can focus on different
    relationships (e.g., syntax vs semantics in language).
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V (combined for efficiency)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)

        # Project and reshape into heads: (batch, heads, seq, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # Broadcast across heads

        # Apply attention per head
        attn_output, self.attn_weights = scaled_dot_product_attention(
            Q, K, V, mask, self.dropout)

        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        return self.W_o(attn_output)


# ── Positional encoding ───────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding. Transformers have no inherent notion
    of sequence order, so we inject it. Uses sin/cos of different frequencies
    so the model can learn relative positions.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # Not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ── Transformer encoder block ─────────────────────────────────────────────────
class TransformerEncoderBlock(nn.Module):
    """
    One transformer encoder layer:
      x → MHA → add&norm → FFN → add&norm

    Uses pre-norm (norm before sublayer) which trains more stably than
    the original post-norm formulation.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm + residual
        x = x + self.dropout(self.attention(self.norm1(x), self.norm1(x),
                                            self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ── Full transformer classifier ───────────────────────────────────────────────
class TransformerClassifier(nn.Module):
    """
    Transformer encoder for sequence classification (BERT-style).
    embed → positional encoding → N encoder blocks → pool → classify
    """

    def __init__(self, vocab_size: int, d_model: int = 256, num_heads: int = 8,
                 num_layers: int = 4, d_ff: int = 1024, num_classes: int = 2,
                 max_len: int = 512, dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x != 0).unsqueeze(1).unsqueeze(2)  # Padding mask
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        # Mean pooling over non-pad tokens
        pad_mask = (x.abs().sum(-1) != 0).unsqueeze(-1).float()
        pooled = (x * pad_mask).sum(1) / pad_mask.sum(1).clamp(min=1)
        return self.classifier(pooled)


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Causal (look-ahead) mask for autoregressive generation (GPT-style).
    Position i can only attend to positions <= i. Lower-triangular = 1.
    """
    return torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("1. Scaled dot-product attention:")
    Q = torch.randn(2, 8, 10, 32)  # batch, heads, seq, d_k
    K = torch.randn(2, 8, 10, 32)
    V = torch.randn(2, 8, 10, 32)
    out, weights = scaled_dot_product_attention(Q, K, V)
    print(f"   Output: {out.shape}, Attention weights: {weights.shape}")
    assert torch.allclose(weights.sum(-1), torch.ones(2, 8, 10), atol=1e-5), "Weights must sum to 1"
    print("   ✓ Attention weights sum to 1 (valid softmax)")

    print("\n2. Transformer classifier:")
    model = TransformerClassifier(vocab_size=30000, d_model=256, num_heads=8,
                                  num_layers=4, num_classes=3).to(device)
    x = torch.randint(1, 30000, (8, 64)).to(device)
    out = model(x)
    print(f"   Input: {x.shape} → Output: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n3. Causal mask (for GPT-style generation):")
    mask = create_causal_mask(5)
    print(f"   Shape: {mask.shape}")
    print(f"   Lower triangular (position i attends to <= i):\n{mask[0,0].int()}")
