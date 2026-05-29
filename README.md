# Pet Image Classification

A simple deep learning project that classifies images of cats and dogs using a neural network built from scratch with NumPy.

## The purpose of this project is to understand:

* Image preprocessing
* Neural network architecture
* Forward propagation
* Backpropagation
* Gradient descent
* Activation functions
* Loss functions
* Model evaluation

---

## Features

* Load image datasets using Python and PIL
* Resize and normalize image data
* Manual implementation of:

  * ReLU activation
  * Softmax activation
  * Cross entropy loss
  * Backpropagation
  * Gradient descent
* Cat vs Dog image classification
* Accuracy evaluation on test data

---

## Technologies Used

* Python
* NumPy
* Pillow (PIL)

---

## Dataset Structure

```text
data/
│
├── training_set/
│   ├── dogs/
│   └── cats/
│
└── test_set/
    ├── dogs/
    └── cats/
```

---

## Model Architecture

```text
Input Image
    ↓
Flatten Layer
    ↓
Dense Hidden Layer (ReLU)
    ↓
Output Layer (Softmax)
    ↓
Prediction (Cat or Dog)
```

---

## Future Improvements

This project is still being improved. Planned future upgrades.

---

## Learning Goals

This project was created to gain practical experience in deep learning and understand how neural networks work internally rather than only using prebuilt libraries.

---

## Author

Bảo Đỗ Hoàng
