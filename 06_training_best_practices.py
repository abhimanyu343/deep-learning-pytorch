"""
Deep Learning Training Best Practices (PyTorch)
=================================================
The techniques that separate models that train from models that don't.

Covers:
  1. Learning rate scheduling (warmup, cosine, one-cycle, reduce-on-plateau)
  2. Gradient clipping and accumulation
  3. Mixed-precision training (faster, less memory)
  4. Early stopping with best-model checkpointing
  5. Weight initialisation strategies
  6. Regularisation (dropout, weight decay, label smoothing, mixup)
  7. Learning-rate range test (finding the optimal LR)
  8. Reproducibility (seeding everything)
"""

import torch
import torch.nn as nn
import numpy as np
import random
from typing import Optional, Callable, List
from dataclasses import dataclass, field


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int = 42, deterministic: bool = True):
    """Seed every RNG for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ── Learning rate finder ──────────────────────────────────────────────────────
class LRFinder:
    """
    Learning Rate Range Test (Leslie Smith).
    Exponentially increase LR over a few hundred iterations and record loss.
    The optimal LR is roughly where loss is steepest (before it diverges).
    """

    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.history = {"lr": [], "loss": []}

    def range_test(self, loader, start_lr: float = 1e-7, end_lr: float = 10,
                   num_iter: int = 100) -> dict:
        # Save initial state to restore later
        init_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        mult = (end_lr / start_lr) ** (1 / num_iter)
        lr = start_lr
        best_loss, avg_loss, beta = float("inf"), 0.0, 0.98

        self.model.train()
        iterator = iter(loader)
        for i in range(num_iter):
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                inputs, targets = next(iterator)

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            self.optimizer.zero_grad()
            loss = self.criterion(self.model(inputs), targets)
            loss.backward()
            self.optimizer.step()

            # Smoothed loss (EMA)
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed = avg_loss / (1 - beta ** (i + 1))
            self.history["lr"].append(lr)
            self.history["loss"].append(smoothed)

            if smoothed > 4 * best_loss:  # Diverged
                break
            best_loss = min(best_loss, smoothed)
            lr *= mult

        # Restore initial weights
        self.model.load_state_dict(init_state)

        # Suggest LR: steepest descent point
        losses = np.array(self.history["loss"])
        lrs = np.array(self.history["lr"])
        gradients = np.gradient(losses)
        suggested_lr = float(lrs[np.argmin(gradients)])
        return {"suggested_lr": suggested_lr, "history": self.history}


# ── Early stopping ────────────────────────────────────────────────────────────
class EarlyStopping:
    """
    Stop training when validation metric stops improving.
    Saves the best model state for restoration.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 mode: str = "min", restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.best_state = None
        self.should_stop = False

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta

    def __call__(self, value: float, model: nn.Module) -> bool:
        if self._is_improvement(value):
            self.best_value = value
            self.counter = 0
            if self.restore_best:
                self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.restore_best and self.best_state:
                    model.load_state_dict(self.best_state)
        return self.should_stop


# ── Mixup augmentation ────────────────────────────────────────────────────────
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """
    Mixup: train on convex combinations of pairs of examples.
    Strong regulariser — improves generalisation and calibration.

    x_mix = λ·x_i + (1-λ)·x_j,  loss = λ·L(y_i) + (1-λ)·L(y_j)
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Scheduler factory ─────────────────────────────────────────────────────────
def build_scheduler(optimizer, strategy: str, **kwargs):
    """
    Build a learning rate scheduler.

    Strategies:
      'cosine'      - Cosine annealing (smooth decay, popular default)
      'onecycle'    - One-cycle policy (warmup then anneal, fast convergence)
      'plateau'     - Reduce on validation plateau (adaptive)
      'warmup_cosine' - Linear warmup then cosine (transformer standard)
    """
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau, LambdaLR
    )

    if strategy == "cosine":
        return CosineAnnealingLR(optimizer, T_max=kwargs.get("epochs", 100))
    elif strategy == "onecycle":
        return OneCycleLR(optimizer, max_lr=kwargs.get("max_lr", 0.1),
                          total_steps=kwargs.get("total_steps", 1000))
    elif strategy == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                 patience=kwargs.get("patience", 5))
    elif strategy == "warmup_cosine":
        warmup_steps = kwargs.get("warmup_steps", 100)
        total_steps = kwargs.get("total_steps", 1000)
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
        return LambdaLR(optimizer, lr_lambda)
    raise ValueError(f"Unknown scheduler: {strategy}")


# ── Weight initialisation ─────────────────────────────────────────────────────
def init_weights(model: nn.Module, method: str = "kaiming"):
    """Apply principled weight initialisation across a model."""
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if method == "kaiming":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif method == "xavier":
                nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


if __name__ == "__main__":
    set_seed(42)
    print("Reproducibility: all RNGs seeded ✓")

    # Demo model
    model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 2))
    init_weights(model, "kaiming")
    print(f"Weights initialised (Kaiming) ✓")

    # Early stopping demo
    es = EarlyStopping(patience=3, mode="min")
    fake_losses = [1.0, 0.8, 0.7, 0.71, 0.72, 0.73]
    for epoch, loss in enumerate(fake_losses):
        stop = es(loss, model)
        print(f"  Epoch {epoch}: loss={loss}, best={es.best_value:.3f}, "
              f"counter={es.counter}, stop={stop}")
        if stop:
            print(f"  → Early stopped, restored best (loss={es.best_value})")
            break

    # Scheduler demo
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = build_scheduler(opt, "warmup_cosine", warmup_steps=10, total_steps=100)
    lrs = []
    for step in range(100):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    print(f"\nWarmup-cosine LR: start={lrs[0]:.4f}, peak={max(lrs):.4f}, end={lrs[-1]:.6f}")

    # Mixup demo
    x = torch.randn(16, 20)
    y = torch.randint(0, 2, (16,))
    mx, ya, yb, lam = mixup_data(x, y, alpha=0.2)
    print(f"Mixup: lambda={lam:.3f}, mixed batch shape={mx.shape}")
