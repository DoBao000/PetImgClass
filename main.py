import os
import numpy as np
from src.data_loader import load_images, normalize_data, apply_normalization
from src.model import NeuralNetwork
from src.train import train, evaluate

# --- CONFIG ---
PATH       = os.getcwd() + '/data'
IMG_SIZE   = (64, 64)
SPEED      = 43    # Random seed for reproducibility
EPOCHS     = 40
LR         = 0.001
BATCH_SIZE = 64
# --------------

if __name__ == '__main__':
    np.random.seed(SPEED)

    # Load & normalize data
    X_train, y_train = load_images(PATH + '/training_set', IMG_SIZE)
    X_train, mean_ch, std_ch = normalize_data(X_train, IMG_SIZE)

    X_test, y_test = load_images(PATH + '/test_set', IMG_SIZE)
    X_test = apply_normalization(X_test, IMG_SIZE, mean_ch, std_ch)

    # Build & train model
    model = NeuralNetwork(input_size=12288, hidden_size=256, output_size=2, lr=LR)
    train(model, X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Evaluate
    evaluate(model, X_test, y_test)