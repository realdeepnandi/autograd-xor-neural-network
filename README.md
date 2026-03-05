# Neural Network From Scratch (Autograd + XOR)

This project implements a tiny neural network framework from scratch in Python, including an automatic differentiation engine (autograd) and a multi-layer perceptron (MLP) trained using backpropagation and gradient descent.

The model is trained to solve the classic XOR problem, demonstrating how neural networks learn nonlinear relationships.

---

## Features

- Automatic differentiation (Autograd engine)
- Backpropagation using the chain rule
- Multi-layer perceptron (MLP)
- Gradient descent optimization
- XOR classification example
- Fully implemented from scratch (no ML libraries)

---

## Project Structure

```
.
├── value.py     # Autograd engine (Value class)
├── nn.py        # Neuron, Layer, and MLP implementation
├── xor.py       # Training script for XOR problem
└── README.md
```

---

## Core Idea

Each neuron computes a linear transformation followed by a nonlinear activation:

```
z = w · x + b
a = tanh(z)
```

Where:
- `w` = weights
- `x` = inputs
- `b` = bias
- `tanh` = nonlinear activation function

Training is performed using gradient descent:

```
w = w - η * gradient
```

Where `η` is the learning rate. Gradients are computed automatically using the autograd engine.

---

## XOR Problem

The XOR dataset:

| x1 | x2 | Output |
|----|----|--------|
| 1  | 0  | 1      |
| 1  | 1  | 0      |
| 0  | 1  | 1      |
| 0  | 0  | 0      |

A single linear neuron cannot solve XOR, so a multi-layer neural network is required.

---

## Example Training Output

```
Iteration 299

Predictions:
[0.943, 0.056, 0.946, 0.092]

Loss:
0.0177
```

The network successfully learns the XOR mapping.

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/<your-username>/<repo-name>
cd <repo-name>
```

Run the XOR training script:

```bash
python xor.py
```

---

## Learning Goals

This project was built to understand the fundamentals of deep learning, including:

- Automatic differentiation
- Computational graphs
- Backpropagation
- Neural network training

Implementing these concepts from scratch provides deeper insight into how modern frameworks work.

---

## Inspiration

Inspired by the excellent educational work of [Andrej Karpathy](https://karpathy.ai/) and his neural network lectures.
