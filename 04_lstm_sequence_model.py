"""
LSTM Sequence Models (PyTorch)
===============================
Recurrent networks for sequential data — time series and text.

Concepts:
  - Why RNNs for sequences (parameter sharing across time steps)
  - LSTM gates (forget, input, output) solve vanishing gradients
  - Bidirectional LSTMs (context from both directions)
  - Attention mechanism over LSTM outputs
  - Packing variable-length sequences (efficiency)
  - Sequence-to-one (classification) and sequence-to-sequence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Optional, Tuple


# ── LSTM with attention for sequence classification ───────────────────────────
class AttentionLSTM(nn.Module):
    """
    Bidirectional LSTM with self-attention pooling for sequence classification.

    Use cases: sentiment analysis, time-series classification, intent detection.

    Pipeline:
      embed → BiLSTM → attention-weighted pooling → classifier
    """

    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden_dim: int = 256, num_layers: int = 2,
                 num_classes: int = 2, dropout: float = 0.3,
                 bidirectional: bool = True, pad_idx: int = 0):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        lstm_output_dim = hidden_dim * self.num_directions

        # Attention: learn which time steps matter most
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_dim // 2, 1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def attention_pool(self, lstm_out: torch.Tensor,
                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute attention weights over time steps and produce a weighted sum.

        Args:
            lstm_out: (batch, seq_len, hidden*directions)
            mask:     (batch, seq_len) — 1 for real tokens, 0 for padding

        Returns:
            (batch, hidden*directions) — context vector
        """
        scores = self.attention(lstm_out).squeeze(-1)   # (batch, seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
        context = torch.sum(lstm_out * weights, dim=1)    # (batch, hidden*dir)
        return context

    def forward(self, x: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        mask = (x != 0).long() if lengths is not None else None
        embedded = self.embedding(x)   # (batch, seq_len, embed_dim)

        if lengths is not None:
            # Pack to skip padding computation (efficiency)
            packed = pack_padded_sequence(embedded, lengths.cpu(),
                                          batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(embedded)

        context = self.attention_pool(lstm_out, mask)
        return self.classifier(context)


# ── LSTM for time-series forecasting (seq-to-one) ─────────────────────────────
class TimeSeriesLSTM(nn.Module):
    """
    LSTM for multivariate time-series forecasting.
    Takes a window of past observations, predicts the next value(s).
    """

    def __init__(self, n_features: int, hidden_dim: int = 64,
                 num_layers: int = 2, horizon: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use the last time step's hidden state for prediction
        last_hidden = lstm_out[:, -1, :]   # (batch, hidden_dim)
        return self.head(last_hidden)       # (batch, horizon)


# ── Manual LSTM cell (to show the gates) ──────────────────────────────────────
class LSTMCellScratch(nn.Module):
    """
    Single LSTM cell implemented manually to expose the gate mechanics.

    Gates (all use sigmoid except cell candidate which uses tanh):
      f_t = σ(W_f · [h_{t-1}, x_t] + b_f)   forget gate
      i_t = σ(W_i · [h_{t-1}, x_t] + b_i)   input gate
      g_t = tanh(W_g · [h_{t-1}, x_t] + b_g) cell candidate
      o_t = σ(W_o · [h_{t-1}, x_t] + b_o)   output gate
      c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t       new cell state
      h_t = o_t ⊙ tanh(c_t)                  new hidden state
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # Combined weight matrix for all 4 gates (efficiency)
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x: torch.Tensor,
                state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.W(combined)
        f, i, g, o = gates.chunk(4, dim=1)

        f = torch.sigmoid(f)        # Forget gate
        i = torch.sigmoid(i)        # Input gate
        g = torch.tanh(g)           # Cell candidate
        o = torch.sigmoid(o)        # Output gate

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("1. Attention LSTM (text classification):")
    model = AttentionLSTM(vocab_size=10000, embed_dim=128, hidden_dim=256,
                          num_classes=3).to(device)
    x = torch.randint(1, 10000, (16, 50)).to(device)  # batch=16, seq=50
    lengths = torch.randint(20, 50, (16,))
    out = model(x, lengths)
    print(f"   Input: {x.shape} → Output: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n2. Time-series LSTM (forecasting):")
    ts_model = TimeSeriesLSTM(n_features=8, hidden_dim=64, horizon=7).to(device)
    ts_x = torch.randn(32, 30, 8).to(device)  # 30 days lookback, 8 features
    ts_out = ts_model(ts_x)
    print(f"   Input: {ts_x.shape} → Forecast: {ts_out.shape} (7-day horizon)")

    print("\n3. Manual LSTM cell (gate mechanics):")
    cell = LSTMCellScratch(input_size=10, hidden_size=20).to(device)
    h, c = torch.zeros(4, 20).to(device), torch.zeros(4, 20).to(device)
    x_t = torch.randn(4, 10).to(device)
    h, c = cell(x_t, (h, c))
    print(f"   Hidden state: {h.shape}, Cell state: {c.shape}")
