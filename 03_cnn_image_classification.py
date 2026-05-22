"""
CNN for Image Classification (PyTorch)
=======================================
A ResNet-style convolutional network with residual blocks, batch norm,
and a full training pipeline on CIFAR-10 style data.

Concepts demonstrated:
  - Convolutional feature extraction (why conv beats dense for images)
  - Residual connections (skip connections → trainable deep networks)
  - Batch normalisation (faster, more stable training)
  - Global average pooling (parameter-efficient alternative to flatten+dense)
  - Data augmentation (random crop, flip → better generalisation)
  - Learning rate scheduling (warmup + cosine annealing)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ── Building block: Residual block ────────────────────────────────────────────
class ConvResidualBlock(nn.Module):
    """
    Residual block: two 3x3 convs with a skip connection.
    The skip connection lets gradients flow directly, enabling very deep nets.

    out = ReLU( BN(conv2(ReLU(BN(conv1(x))))) + shortcut(x) )
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Shortcut: 1x1 conv to match dimensions when stride != 1 or channels change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)   # Residual connection
        return F.relu(out)


# ── Full ResNet ───────────────────────────────────────────────────────────────
class ResNet(nn.Module):
    """
    ResNet for image classification (CIFAR-10 sized: 32x32x3).

    Architecture:
      stem conv → 3 residual stages (increasing channels, decreasing spatial)
      → global average pool → linear classifier
    """

    def __init__(self, num_classes: int = 10, channels: List[int] = None,
                 blocks_per_stage: int = 2):
        super().__init__()
        channels = channels or [64, 128, 256]
        self.in_channels = channels[0]

        # Stem: initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # Residual stages
        self.stage1 = self._make_stage(channels[0], blocks_per_stage, stride=1)
        self.stage2 = self._make_stage(channels[1], blocks_per_stage, stride=2)
        self.stage3 = self._make_stage(channels[2], blocks_per_stage, stride=2)

        # Global average pool + classifier
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels[2], num_classes)

        self._initialise_weights()

    def _make_stage(self, out_channels: int, n_blocks: int, stride: int) -> nn.Sequential:
        layers = [ConvResidualBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(n_blocks - 1):
            layers.append(ConvResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _initialise_weights(self):
        """Kaiming init for conv layers (matches ReLU activations)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Data augmentation pipeline ────────────────────────────────────────────────
def get_transforms(train: bool = True):
    """
    Image transforms. Augmentation only on training data.
    Augmentation acts as regularisation — the model sees varied versions
    of each image, reducing overfitting.
    """
    from torchvision import transforms
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),     # Spatial invariance
            transforms.RandomHorizontalFlip(),         # Mirror invariance
            transforms.ColorJitter(0.2, 0.2, 0.2),     # Lighting robustness
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),  # CIFAR-10 stats
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])


# ── Training step ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, 100.0 * correct / total


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ResNet(num_classes=10, blocks_per_stage=2).to(device)
    print(f"ResNet parameters: {model.count_parameters():,}")

    # Sanity check forward pass with random CIFAR-sized batch
    dummy = torch.randn(8, 3, 32, 32).to(device)
    out = model(dummy)
    print(f"Forward pass: {dummy.shape} → {out.shape}")
    assert out.shape == (8, 10), "Output shape mismatch"

    # Training setup (would run on real CIFAR-10 data)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing regularises
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    print("\nModel architecture verified. Ready to train on CIFAR-10.")
    print("To train: load torchvision.datasets.CIFAR10 with get_transforms()")
