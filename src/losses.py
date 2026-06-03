import numpy as np


def cross_entropy_loss(y_pred: np.ndarray, y: np.ndarray) -> float:
    """Mean cross-entropy loss over a batch."""
    m = y.shape[0]
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)  # Prevent log(0)
    return -np.mean(np.log(y_pred[np.arange(m), y]))