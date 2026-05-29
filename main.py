import os
import numpy as np
from PIL import Image

# --- CONFIG ---
PATH = os.getcwd() + '/data'
IMG_SIZE = (64, 64)
SPEED = 43        # Random seed for reproducibility
EPOCHS = 40      # How many times we pass through the dataset
LR = 0.001       # Learning rate (lowered: 0.01 overshoots on full-batch)
BATCH_SIZE = 64  # Mini-batch size for SGD
VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
# --------------


# --- LOAD IMAGES ---
def load_images(path: str, img_size: tuple) -> tuple:
    images, labels = [], []
    dogs_path, cats_path = path + '/dogs', path + '/cats'

    for file in os.listdir(dogs_path):
        if os.path.splitext(file)[1].lower() not in VALID_EXT:
            continue
        img = Image.open(os.path.join(dogs_path, file)).convert('RGB')
        img = img.resize(img_size)
        images.append(np.array(img, dtype=np.float32))
        labels.append(0)  # 0 = dog

    for file in os.listdir(cats_path):
        if os.path.splitext(file)[1].lower() not in VALID_EXT:
            continue
        img = Image.open(os.path.join(cats_path, file)).convert('RGB')
        img = img.resize(img_size)
        images.append(np.array(img, dtype=np.float32))
        labels.append(1)  # 1 = cat

    return np.array(images), np.array(labels)
# -------------------


# --- NORMALIZE ---
def normalize_data(X: np.ndarray, img_size: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize images channel-wise (zero mean, unit variance).
    Returns flattened normalized array + the mean/std used (needed for test set).
    """
    N = X.shape[0]
    H, W = img_size
    X_img = X.reshape(N, H, W, 3)

    mean_ch = X_img.mean(axis=(0, 1, 2))  # (3,)
    std_ch  = X_img.std(axis=(0, 1, 2))   # (3,)
    std_ch[std_ch == 0] = 1               # Avoid division by zero

    X_norm = (X_img - mean_ch) / std_ch   # Broadcast over (N, H, W, 3)
    return X_norm.reshape(N, H * W * 3), mean_ch, std_ch


def apply_normalization(X: np.ndarray, img_size: tuple,
                        mean_ch: np.ndarray, std_ch: np.ndarray) -> np.ndarray:
    """Apply pre-computed mean/std to a new set of images (e.g. test set)."""
    N = X.shape[0]
    H, W = img_size
    X_img = X.reshape(N, H, W, 3)
    X_norm = (X_img - mean_ch) / std_ch
    return X_norm.reshape(N, H * W * 3)
# -----------------


# --- ACTIVATION ---
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
# ------------------


# --- LOSS FUNCTION ---
def cross_entropy_loss(y_pred: np.ndarray, y: np.ndarray) -> float:
    """Mean cross-entropy loss over a batch."""
    m = y.shape[0]
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)  # Prevent log(0)
    return -np.mean(np.log(y_pred[np.arange(m), y]))
# ---------------------


# --- NN MODEL ---
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
        self.Z1 = X @ self.W1 + self.b1   # Pre-activation (hidden)
        self.A1 = self.activate.reLU(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2  # Pre-activation (output)
        self.A2 = self.activate.softmax(self.Z2)
        return self.A2

    def backward(self, X: np.ndarray, y: np.ndarray) -> None:
        m = X.shape[0]

        # One-hot encode labels
        y_onehot = np.zeros_like(self.A2)
        y_onehot[np.arange(m), y] = 1

        # Output layer gradient (softmax + cross-entropy combined derivative)
        dZ2 = self.A2 - y_onehot
        dW2 = (self.A1.T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Hidden layer gradient — use Z1 (pre-activation) for ReLU derivative
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
# ----------------


# --- MAIN ---
if __name__ == '__main__':
    np.random.seed(SPEED)  # Apply seed for reproducibility

    X_train, y_train = load_images(PATH + '/training_set', IMG_SIZE)
    X_train, mean_ch, std_ch = normalize_data(X_train, IMG_SIZE)

    X_test, y_test = load_images(PATH + '/test_set', IMG_SIZE)
    X_test = apply_normalization(X_test, IMG_SIZE, mean_ch, std_ch)

    model = NeuralNetwork(12288, 256, 2, LR)

    N = X_train.shape[0]

    for epoch in range(EPOCHS):
        # Shuffle dataset at the start of each epoch
        indices = np.random.permutation(N)

        epoch_loss, epoch_acc = 0.0, 0.0
        n_batches = 0

        for i in range(0, N, BATCH_SIZE):
            batch_idx = indices[i : i + BATCH_SIZE]
            X_batch   = X_train[batch_idx]
            y_batch   = y_train[batch_idx]

            y_pred = model.forward(X_batch)
            loss   = cross_entropy_loss(y_pred, y_batch)
            model.backward(X_batch, y_batch)

            epoch_loss += loss
            epoch_acc  += np.mean(np.argmax(y_pred, axis=1) == y_batch)
            n_batches  += 1

        # Average loss/acc across all batches in this epoch
        epoch_loss /= n_batches
        epoch_acc  /= n_batches

        print(f'Epoch {epoch + 1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

    test_pred = model.predict(X_test)
    test_acc   = np.mean(test_pred == y_test)
    print(f'\nTest Accuracy: {test_acc:.4f}')
# ------------