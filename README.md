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

## Installation

Make sure you have Python 3.8+ installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Use

### 1. Prepare the Dataset

Download or place your dataset following the structure shown in the **Dataset Structure** section above. The `data/` folder should sit in the root of the project directory.

### 2. Train and Evaluate the Model

Run the main script to train the neural network and evaluate it on the test set:

```bash
python main.py
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
