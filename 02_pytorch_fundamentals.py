"""
PyTorch Fundamentals — Tensors, Autograd, Custom Layers
=========================================================
The patterns you need to know before building anything serious in PyTorch.

Covers:
  1. Tensor operations and GPU/CPU transfers
  2. Autograd — how gradients flow
  3. Custom nn.Module with forward + extra methods
  4. Custom loss functions
  5. Custom datasets and DataLoaders
  6. Training loop with best practices
  7. Model serialisation (state_dict vs full pickle)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import numpy as np
from typing import Optional, Tuple, Callable
import time


# ── 1. Tensor essentials ──────────────────────────────────────────────────────

def tensor_basics():
    """Core tensor operations every PyTorch user must know."""
    # Creation
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    zeros = torch.zeros(3, 4)
    rand  = torch.randn(2, 3, 4)   # From N(0,1)

    # Device management (always check, don't hardcode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_gpu = x.to(device)

    # Einsum (cleaner than matmul chains for complex ops)
    A = torch.randn(32, 64)  # batch × features
    B = torch.randn(64, 128)
    C = torch.einsum("bf,fd->bd", A, B)  # Equivalent to A @ B

    # View vs reshape vs contiguous
    t = torch.randn(3, 4, 5)
    t_view = t.view(12, 5)          # Requires contiguous memory
    t_reshape = t.reshape(12, 5)    # Always works (copies if needed)
    t_flat = t.flatten()            # All elements in 1D

    # Aggregations with dim argument
    x2 = torch.randn(4, 8, 16)
    mean_per_sample = x2.mean(dim=[1, 2])     # (4,) — mean over H and W
    max_vals, max_idx = x2.max(dim=-1)         # (4, 8) — max along last dim

    # In-place operations (use carefully — breaks autograd)
    x_copy = x.clone()
    x_copy.add_(1.0)    # In-place: modifies x_copy

    print(f"Tensor basics: device={device}, C.shape={C.shape}")
    return device


# ── 2. Autograd mechanics ──────────────────────────────────────────────────────

def autograd_deep_dive():
    """Understand exactly how PyTorch builds and traverses the computation graph."""
    # Simple example: y = (x + 2)² × 3
    x = torch.tensor(2.0, requires_grad=True)
    y = 3 * (x + 2) ** 2
    y.backward()
    # dy/dx = 6*(x+2) = 6*4 = 24
    assert abs(x.grad.item() - 24.0) < 1e-5, f"Expected 24, got {x.grad.item()}"

    # Gradient accumulation — always zero_grad() before backward
    x = torch.tensor(1.0, requires_grad=True)
    for _ in range(3):
        y = x ** 2
        y.backward()
    print(f"Accumulated gradient (wrong): {x.grad.item()}")  # 6.0 (summed 3×)

    x.grad.zero_()  # Reset
    y = x ** 2
    y.backward()
    print(f"Correct gradient:             {x.grad.item()}")  # 2.0

    # Detach from graph (useful for target networks, EMA, inference)
    pred = torch.randn(4, 10, requires_grad=True)
    target = pred.detach()  # Same values, no gradient
    assert not target.requires_grad

    # torch.no_grad() — most efficient for inference
    with torch.no_grad():
        expensive_op = torch.randn(1000, 1000) @ torch.randn(1000, 1000)
    # No computation graph built — 2-3× faster for inference

    # Custom gradient function
    class ClippedReLU(torch.autograd.Function):
        """ReLU clipped at 1.0 — straight-through estimator for backward."""
        @staticmethod
        def forward(ctx, x: torch.Tensor) -> torch.Tensor:
            ctx.save_for_backward(x)
            return x.clamp(0, 1)

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
            (x,) = ctx.saved_tensors
            # Straight-through: pass gradient through where x is in [0,1]
            return grad_output * ((x >= 0) & (x <= 1)).float()

    x = torch.randn(5, requires_grad=True)
    y = ClippedReLU.apply(x)
    y.sum().backward()
    print(f"Custom autograd: grad shape = {x.grad.shape}")


# ── 3. Custom nn.Module ────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Residual (skip connection) block.
    
    Output = F(x) + x  (or F(x) + projection(x) if dimensions differ)
    
    Why residuals work: they let gradients flow directly through
    skip connections, solving the vanishing gradient problem for deep nets.
    They also initialise as identity functions (F(x) → 0 at start),
    making optimisation easier.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[activation]

        self.block = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            act_fn(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, in_features),
            nn.LayerNorm(in_features),
        )

        # Projection for skip connection if dimensions differ
        self.skip = nn.Linear(in_features, in_features, bias=False)                     if in_features != in_features else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """Initialise the last layer with near-zero weights (residual initialisation)."""
        last_linear = [m for m in self.block.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.xavier_normal_(last_linear.weight, gain=0.01)
        if last_linear.bias is not None:
            nn.init.zeros_(last_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.block(x) + self.skip(x))


class TabularNet(nn.Module):
    """
    Tabular data neural network with:
    - Embedding layers for categorical features
    - Batch norm + dropout for numerical features
    - Residual blocks for the main trunk
    - Task head (regression or classification)
    
    Outperforms gradient boosting on many tabular benchmarks when
    trained with proper regularisation.
    """

    def __init__(
        self,
        n_numerical: int,
        categorical_dims: Optional[list] = None,  # [(vocab_size, embed_dim), ...]
        hidden_dim: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.2,
        n_outputs: int = 1,
        task: str = "regression",
    ):
        super().__init__()
        self.task = task

        # Categorical embeddings
        self.embeddings = nn.ModuleList()
        embed_total = 0
        if categorical_dims:
            for vocab_size, embed_dim in categorical_dims:
                self.embeddings.append(nn.Embedding(vocab_size, embed_dim))
                embed_total += embed_dim

        # Numerical input processing
        total_input = n_numerical + embed_total
        self.input_norm = nn.BatchNorm1d(n_numerical) if n_numerical > 0 else None
        self.input_projection = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Residual trunk
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(n_blocks)
        ])

        # Task head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_outputs),
        )

    def forward(
        self,
        x_num: Optional[torch.Tensor] = None,
        x_cat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = []
        if x_num is not None:
            if self.input_norm:
                x_num = self.input_norm(x_num)
            parts.append(x_num)
        if x_cat is not None and self.embeddings:
            embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            parts.extend(embeds)

        x = torch.cat(parts, dim=-1)
        x = self.input_projection(x)
        x = self.blocks(x)
        logits = self.head(x)

        if self.task == "binary_classification":
            return torch.sigmoid(logits)
        elif self.task == "multiclass_classification":
            return F.softmax(logits, dim=-1)
        return logits


# ── 4. Production training loop ───────────────────────────────────────────────

class Trainer:
    """Clean, production-grade training loop with best practices."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[object] = None,
        grad_clip: float = 1.0,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.history = {"train_loss": [], "val_loss": [], "lr": []}

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss, n_batches = 0.0, 0

        for batch in loader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[:-1], batch[-1]
                inputs = [x.to(self.device) if isinstance(x, torch.Tensor) else x
                          for x in inputs]
                targets = targets.to(self.device)
            else:
                inputs, targets = batch.to(self.device), batch.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()

            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                if isinstance(inputs, list):
                    outputs = self.model(*inputs)
                else:
                    outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            loss.backward()

            # Gradient clipping: prevents exploding gradients
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            if self.scheduler and hasattr(self.scheduler, "step"):
                self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[:-1], batch[-1]
                inputs = [x.to(self.device) for x in inputs if isinstance(x, torch.Tensor)]
                targets = targets.to(self.device)
            else:
                inputs, targets = batch.to(self.device), batch.to(self.device)
            outputs = self.model(*inputs) if isinstance(inputs, list) else self.model(inputs)
            loss = self.criterion(outputs, targets)
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 50,
        patience: int = 10,
        save_path: Optional[str] = None,
    ) -> dict:
        best_val, patience_count = float("inf"), 0

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss   = self.evaluate(val_loader)
            epoch_time = time.time() - t0

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)

            print(f"Epoch {epoch:3d}/{n_epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"LR: {current_lr:.2e} | {epoch_time:.1f}s")

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                patience_count = 0
                if save_path:
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                    }, save_path)
            else:
                patience_count += 1
                if patience_count >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        return self.history


if __name__ == "__main__":
    device = tensor_basics()
    autograd_deep_dive()

    print("
TabularNet forward pass test:")
    model = TabularNet(n_numerical=10, hidden_dim=128, n_blocks=3,
                        n_outputs=1, task="regression")
    x_num = torch.randn(32, 10)
    out = model(x_num=x_num)
    print(f"  Input: {x_num.shape} → Output: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
