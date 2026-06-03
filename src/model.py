import numpy as np


class Activate:
    def reLU(self, Z: np.ndarray) -> np.ndarray:
        return np.maximum(0, Z)

    def reLU_grad(self, Z: np.ndarray) -> np.ndarray:
        """Derivative of ReLU w.r.t. the pre-activation Z (not the output A)."""
        return (Z > 0).astype(float)

    def softmax(self, Z: np.ndarray) -> np.ndarray:
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)  # Numerical stability
        exp_vals  = np.exp(Z_shifted)
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)


class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, lr: float) -> None:
        self.lr = lr
        self.activate = Activate()

        # He initialization: better gradient flow with ReLU
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.Z1 = X @ self.W1 + self.b1        # Pre-activation (hidden)
        self.A1 = self.activate.reLU(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2  # Pre-activation (output)
        self.A2 = self.activate.softmax(self.Z2)
        return self.A2

    def backward(self, X: np.ndarray, y: np.ndarray) -> None:
        m = X.shape[0]

        # One-hot encode labels
        y_onehot = np.zeros_like(self.A2)
        y_onehot[np.arange(m), y] = 1

        # Output layer gradient
        dZ2 = self.A2 - y_onehot
        dW2 = (self.A1.T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Hidden layer gradient
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.activate.reLU_grad(self.Z1)
        dW1 = (X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Gradient descent update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.forward(X)
        return np.argmax(probs, axis=1)