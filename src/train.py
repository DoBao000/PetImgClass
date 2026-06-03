import numpy as np
import csv
import os
from src.model import NeuralNetwork
from src.losses import cross_entropy_loss


def save_history(history: dict):
    os.makedirs('logs', exist_ok=True)
    file_path = os.path.join('logs', 'metrics.csv')

    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(['loss', 'accuracy'])
        writer.writerow([
            history['Loss'],
            history['Accuracy']
        ])

    print(f'History saved to {file_path}')


def train(model: NeuralNetwork, X_train: np.ndarray, y_train: np.ndarray,
          epochs: int, batch_size: int) -> None:
    """Train the model using mini-batch gradient descent."""
    N = X_train.shape[0]
    history = {'Loss': 100, 'Accuracy': -1}

    for epoch in range(epochs):
        # Shuffle dataset at the start of each epoch
        indices = np.random.permutation(N)

        epoch_loss, epoch_acc = 0.0, 0.0
        n_batches = 0

        for i in range(0, N, batch_size):
            batch_idx = indices[i : i + batch_size]
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

        print(f'Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')
        if epoch_acc > history['Accuracy']:
            history['Accuracy'] = epoch_acc
            history['Loss'] = epoch_loss
    save_history(history)


def evaluate(model: NeuralNetwork, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Evaluate model accuracy on the test set."""
    test_pred = model.predict(X_test)
    test_acc  = np.mean(test_pred == y_test)
    print(f'\nTest Accuracy: {test_acc:.4f}')
    return test_acc