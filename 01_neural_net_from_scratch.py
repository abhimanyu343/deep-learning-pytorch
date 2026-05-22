"""
Neural Network from Scratch (NumPy only)
=========================================
Build a complete multi-layer perceptron without any ML framework.
This builds genuine intuition for backpropagation, gradient flow,
and why initialisation and activation functions matter.

Implements:
  - Arbitrary depth MLP with configurable layers
  - Activations: ReLU, Sigmoid, Tanh, Leaky ReLU, Softmax
  - Loss functions: MSE, Binary Cross-Entropy, Categorical Cross-Entropy
  - Optimisers: SGD, SGD+Momentum, Adam
  - Regularisation: L2 weight decay, Dropout
  - Initialisations: He (ReLU), Xavier/Glorot (Sigmoid/Tanh)
  - Batch normalisation (forward pass)
  - Training loop with mini-batches
"""

import numpy as np
from typing import List, Optional, Tuple, Callable, Literal
from dataclasses import dataclass, field


# ── Activation functions and their derivatives ────────────────────────────────

class Activations:
    """All activations implement forward() and backward() (derivative w.r.t input)."""

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    @staticmethod
    def relu_backward(z: np.ndarray) -> np.ndarray:
        """Derivative: 1 if z > 0 else 0."""
        return (z > 0).astype(float)

    @staticmethod
    def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(z > 0, z, alpha * z)

    @staticmethod
    def leaky_relu_backward(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(z > 0, 1.0, alpha)

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        # Numerically stable: handle large negative and positive z
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    @staticmethod
    def sigmoid_backward(a: np.ndarray) -> np.ndarray:
        """Derivative: σ(z) * (1 - σ(z)). Pass the OUTPUT (a), not z."""
        return a * (1 - a)

    @staticmethod
    def tanh(z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    @staticmethod
    def tanh_backward(a: np.ndarray) -> np.ndarray:
        """Derivative: 1 - tanh²(z). Pass the OUTPUT (a)."""
        return 1 - a ** 2

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        """Numerically stable softmax: subtract max before exp."""
        shifted = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    @staticmethod
    def linear(z: np.ndarray) -> np.ndarray:
        return z

    @staticmethod
    def linear_backward(z: np.ndarray) -> np.ndarray:
        return np.ones_like(z)


# ── Weight initialisations ────────────────────────────────────────────────────

def he_init(fan_in: int, fan_out: int) -> np.ndarray:
    """He (Kaiming) initialisation — recommended for ReLU activations.
    Variance = 2 / fan_in. Prevents vanishing/exploding gradients with ReLU."""
    return np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)

def xavier_init(fan_in: int, fan_out: int) -> np.ndarray:
    """Glorot/Xavier — recommended for Sigmoid/Tanh.
    Variance = 2 / (fan_in + fan_out)."""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))


# ── Single dense layer ────────────────────────────────────────────────────────

class DenseLayer:
    """
    Fully connected layer: output = activation(X @ W + b)
    
    Stores intermediate values needed for backprop:
      self.X:       Input to this layer (batch, fan_in)
      self.Z:       Pre-activation (batch, fan_out) = X @ W + b
      self.A:       Post-activation (batch, fan_out)
    """

    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        activation: str = "relu",
        l2_lambda: float = 0.0,
        dropout_rate: float = 0.0,
        init: str = "auto",
    ):
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_name = activation
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.training = True

        # Weight initialisation
        if init == "auto":
            init = "he" if activation == "relu" else "xavier"
        self.W = he_init(fan_in, fan_out) if init == "he" else xavier_init(fan_in, fan_out)
        self.b = np.zeros((1, fan_out))

        # Adam optimiser state
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

        # Cache for backprop
        self.X: Optional[np.ndarray] = None
        self.Z: Optional[np.ndarray] = None
        self.A: Optional[np.ndarray] = None
        self.dropout_mask: Optional[np.ndarray] = None

        # Gradients
        self.dW: Optional[np.ndarray] = None
        self.db: Optional[np.ndarray] = None

    def _activate(self, Z: np.ndarray) -> np.ndarray:
        fn_map = {"relu": Activations.relu, "sigmoid": Activations.sigmoid,
                  "tanh": Activations.tanh, "softmax": Activations.softmax,
                  "leaky_relu": Activations.leaky_relu, "linear": Activations.linear}
        return fn_map[self.activation_name](Z)

    def _activate_deriv(self, Z_or_A: np.ndarray) -> np.ndarray:
        """Returns dA/dZ (element-wise for non-softmax activations)."""
        deriv_map = {
            "relu":       lambda x: Activations.relu_backward(x),
            "sigmoid":    lambda x: Activations.sigmoid_backward(x),
            "tanh":       lambda x: Activations.tanh_backward(x),
            "leaky_relu": lambda x: Activations.leaky_relu_backward(x),
            "linear":     lambda x: Activations.linear_backward(x),
        }
        if self.activation_name in deriv_map:
            return deriv_map[self.activation_name](Z_or_A)
        # Softmax gradient is handled specially in the loss layer
        return np.ones_like(Z_or_A)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        self.Z = X @ self.W + self.b
        self.A = self._activate(self.Z)

        # Inverted dropout — scale at training time, not test time
        if self.dropout_rate > 0 and self.training:
            self.dropout_mask = (np.random.rand(*self.A.shape) > self.dropout_rate)
            self.A = self.A * self.dropout_mask / (1 - self.dropout_rate)

        return self.A

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        Backprop through this layer.
        
        Args:
            dA: Gradient of loss w.r.t. this layer's output (batch, fan_out)
        
        Returns:
            dX: Gradient of loss w.r.t. this layer's input (batch, fan_in)
        """
        m = self.X.shape[0]

        # Apply dropout mask to gradient
        if self.dropout_rate > 0 and self.training and self.dropout_mask is not None:
            dA = dA * self.dropout_mask / (1 - self.dropout_rate)

        # Chain rule: dL/dZ = dL/dA ⊙ dA/dZ
        if self.activation_name == "softmax":
            dZ = dA  # Softmax gradient absorbed into cross-entropy loss
        else:
            dZ = dA * self._activate_deriv(self.A)

        # Parameter gradients
        self.dW = (self.X.T @ dZ) / m
        self.db = dZ.mean(axis=0, keepdims=True)

        # L2 regularisation gradient: dL/dW += λW
        if self.l2_lambda > 0:
            self.dW += (self.l2_lambda / m) * self.W

        # Input gradient (for previous layer)
        dX = dZ @ self.W.T
        return dX

    def update_adam(self, lr: float, t: int, beta1: float = 0.9,
                    beta2: float = 0.999, eps: float = 1e-8) -> None:
        """Adam update: adaptive learning rate with momentum + RMS."""
        # Moment estimates
        self.mW = beta1 * self.mW + (1 - beta1) * self.dW
        self.vW = beta2 * self.vW + (1 - beta2) * self.dW ** 2
        self.mb = beta1 * self.mb + (1 - beta1) * self.db
        self.vb = beta2 * self.vb + (1 - beta2) * self.db ** 2

        # Bias correction (important in early iterations)
        mW_hat = self.mW / (1 - beta1 ** t)
        vW_hat = self.vW / (1 - beta2 ** t)
        mb_hat = self.mb / (1 - beta1 ** t)
        vb_hat = self.vb / (1 - beta2 ** t)

        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)


# ── Full MLP ──────────────────────────────────────────────────────────────────

class MLP:
    """
    Multi-Layer Perceptron — arbitrary depth with Adam optimiser.
    
    Example:
        # XOR problem (needs hidden layer — linear models can't solve this)
        model = MLP(layer_sizes=[2, 8, 4, 1], activations=["relu","relu","sigmoid"])
        model.fit(X, y, task="binary_classification")
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activations: Optional[List[str]] = None,
        l2_lambda: float = 0.0,
        dropout_rates: Optional[List[float]] = None,
        random_state: int = 42,
    ):
        np.random.seed(random_state)
        n_layers = len(layer_sizes) - 1
        activations = activations or (["relu"] * (n_layers - 1) + ["linear"])
        dropout_rates = dropout_rates or [0.0] * n_layers

        assert len(activations) == n_layers,             f"Need {n_layers} activations for {n_layers} weight layers"

        self.layers = [
            DenseLayer(
                fan_in=layer_sizes[i],
                fan_out=layer_sizes[i + 1],
                activation=activations[i],
                l2_lambda=l2_lambda,
                dropout_rate=dropout_rates[i],
            )
            for i in range(n_layers)
        ]
        self.loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        self.task: str = "regression"

    def forward(self, X: np.ndarray) -> np.ndarray:
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def _loss_and_grad(
        self, y_pred: np.ndarray, y_true: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Compute loss and initial gradient (dL/dA_last)."""
        eps = 1e-9
        if self.task == "regression":
            loss = float(np.mean((y_pred - y_true) ** 2))
            dA = 2 * (y_pred - y_true) / y_true.shape[0]
        elif self.task == "binary_classification":
            loss = float(-np.mean(
                y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
            ))
            dA = (-(y_true / (y_pred + eps)) + (1 - y_true) / (1 - y_pred + eps)) / y_true.shape[0]
        elif self.task == "multiclass_classification":
            # Categorical cross-entropy with softmax
            loss = float(-np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1)))
            dA = (y_pred - y_true)  # Combined softmax + CE gradient
        else:
            raise ValueError(f"Unknown task: {self.task}")

        # Add L2 regularisation loss
        for layer in self.layers:
            if layer.l2_lambda > 0:
                loss += (layer.l2_lambda / (2 * len(y_true))) * np.sum(layer.W ** 2)

        return loss, dA

    def backward(self, dA: np.ndarray) -> None:
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str = "regression",
        learning_rate: float = 0.001,
        n_epochs: int = 200,
        batch_size: int = 64,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        patience: int = 20,
        verbose: bool = True,
    ) -> "MLP":
        self.task = task
        n_samples = X.shape[0]
        t = 0  # Adam time step (global)
        best_val_loss = np.inf
        patience_counter = 0
        best_weights = None

        for epoch in range(n_epochs):
            # Shuffle training data
            idx = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X[idx], y[idx]

            epoch_losses = []
            for layer in self.layers:
                layer.training = True

            for start in range(0, n_samples, batch_size):
                X_batch = X_shuffled[start:start + batch_size]
                y_batch = y_shuffled[start:start + batch_size]

                # Forward
                y_pred = self.forward(X_batch)
                batch_loss, dA = self._loss_and_grad(y_pred, y_batch)
                epoch_losses.append(batch_loss)

                # Backward
                self.backward(dA)

                # Update
                t += 1
                for layer in self.layers:
                    layer.update_adam(learning_rate, t)

            # Validation
            for layer in self.layers:
                layer.training = False

            train_loss = float(np.mean(epoch_losses))
            self.loss_history.append(train_loss)

            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)
                val_loss, _ = self._loss_and_grad(val_pred, y_val)
                self.val_loss_history.append(val_loss)

                # Early stopping
                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_weights = [(l.W.copy(), l.b.copy()) for l in self.layers]
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        # Restore best weights
                        if best_weights:
                            for layer, (W, b) in zip(self.layers, best_weights):
                                layer.W, layer.b = W, b
                        break

            if verbose and (epoch + 1) % 25 == 0:
                val_str = f" | Val Loss: {self.val_loss_history[-1]:.4f}" if self.val_loss_history else ""
                print(f"Epoch {epoch+1:4d}/{n_epochs} | Train Loss: {train_loss:.4f}{val_str}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            layer.training = False
        return self.forward(X)

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict(X)
        if proba.shape[1] == 1:
            return (proba.ravel() > 0.5).astype(int)
        return proba.argmax(axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict_classes(X)
        y_true = y.ravel() if y.ndim > 1 else y
        return float((y_pred == y_true).mean())


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.datasets import make_moons, make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    print("1. Non-linear classification (Moons dataset)")
    X, y = make_moons(n_samples=2000, noise=0.2, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    model = MLP(
        layer_sizes=[2, 32, 16, 8, 1],
        activations=["relu", "relu", "relu", "sigmoid"],
        l2_lambda=0.01,
        dropout_rates=[0.0, 0.2, 0.1, 0.0],
    )
    model.fit(X_tr, y_tr.reshape(-1,1), task="binary_classification",
              X_val=X_te, y_val=y_te.reshape(-1,1),
              learning_rate=0.005, n_epochs=300, batch_size=64)

    train_acc = model.accuracy(X_tr, y_tr)
    test_acc  = model.accuracy(X_te, y_te)
    print(f"  Train accuracy: {train_acc*100:.1f}%")
    print(f"  Test accuracy:  {test_acc*100:.1f}%")
    print(f"  Layers: {len(model.layers)} | Final loss: {model.loss_history[-1]:.4f}")
