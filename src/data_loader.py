import os
import numpy as np
from PIL import Image
VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


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
    std_ch[std_ch == 0] = 1

    X_norm = (X_img - mean_ch) / std_ch   # (N, H, W, 3)
    return X_norm.reshape(N, H * W * 3), mean_ch, std_ch


def apply_normalization(X: np.ndarray, img_size: tuple,
                        mean_ch: np.ndarray, std_ch: np.ndarray) -> np.ndarray:
    """Apply pre-computed mean/std to a new set of images (e.g. test set)."""
    N = X.shape[0]
    H, W = img_size
    X_img = X.reshape(N, H, W, 3)
    X_norm = (X_img - mean_ch) / std_ch
    return X_norm.reshape(N, H * W * 3)