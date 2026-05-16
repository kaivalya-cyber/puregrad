<p align="center">
  <h1 align="center">⚛️ PureGrad</h1>
  <p align="center">
    <strong>A Deep Learning Framework Built From Scratch</strong>
  </p>
  <p align="center">
    Zero external ML dependencies · Dynamic computation graphs · Reverse-mode automatic differentiation
  </p>
  <p align="center">
    <a href="#quickstart">Quickstart</a> •
    <a href="#features">Features</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#api-reference">API</a> •
    <a href="#tests">Tests</a> •
    <a href="#license">License</a>
  </p>
</p>

---

## What is Antigravity?

**PureGrad** is a lightweight, educational deep learning framework implemented entirely from scratch in pure Python + NumPy. No PyTorch. No TensorFlow. No Autograd libraries. Every piece of the engine — from the computation graph to backpropagation to gradient descent — is built from first principles.

It can train real neural networks on non-linear datasets and achieve convergence.

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        PUREGRAD — Custom Deep Learning Framework        ┃
┃           Zero Dependencies · Pure Autograd             ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

▸ Training MLP on Moons dataset...
  Epoch  100 │ Loss: 0.043053
  Epoch  200 │ Loss: 0.032260
  Epoch  300 │ Loss: 0.026878
  Epoch  400 │ Loss: 0.022956
  Epoch  500 │ Loss: 0.019878

  ✅ CONVERGED — Final Loss: 0.0199
  📊 Training Accuracy: 99.7%
```

---

## Quickstart

### Prerequisites

- Python 3.10+
- NumPy (the only dependency)

### Installation

```bash
git clone https://github.com/kaivalya-cyber/puregrad.git
cd puregrad

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install numpy

# Run the training demo
python train.py
```

### Minimal Example

```python
from puregrad import Tensor, MLP, MSELoss, SGD
from puregrad.data import make_moons

# Generate dataset
X_np, y_np = make_moons(n_samples=300, noise=0.1)
X_train, y_train = Tensor(X_np), Tensor(y_np)

# Build model
model = MLP(input_dim=2, hidden_dims=[16, 16], output_dim=1, activation="tanh")
optimizer = SGD(model.parameters(), lr=0.05)

# Train
for epoch in range(500):
    model.zero_grad()
    predictions = model(X_train)
    loss = MSELoss(predictions, y_train)
    loss.backward()
    optimizer.step()

print(f"Final loss: {loss.data.flat[0]:.4f}")
```

---

## Features

### 🔧 Core Engine
- **Dynamic Computation Graph** — Automatically constructs a DAG of operations during the forward pass
- **Reverse-Mode Autodiff** — Single `.backward()` call propagates gradients through the entire graph via the chain rule
- **Topological Sort** — DFS-based ordering guarantees correct gradient evaluation sequence
- **Gradient Accumulation** — Gradients accumulate (`+=`), never overwrite — supports safe variable reuse across multiple paths

### 🧮 Tensor Operations
| Category | Operations |
|:---|:---|
| Arithmetic | `+`, `-`, `*`, `/`, `**` (all with broadcasting) |
| Matrix | `@` / `matmul`, `transpose`, `reshape` |
| Reductions | `sum(axis)`, `mean(axis)` |
| Activations | `relu`, `tanh`, `sigmoid` |
| Math | `log`, `exp` |

### 🏗️ Neural Network Modules
- **`Linear(in, out)`** — Fully-connected layer with He initialization
- **`ReLU`**, **`Tanh`**, **`Sigmoid`** — Activation layers
- **`MLP(input_dim, hidden_dims, output_dim)`** — Multi-layer perceptron with configurable architecture

### 📉 Training Pipeline
- **Loss Functions** — `MSELoss`, `CrossEntropyLoss` (binary, with built-in sigmoid)
- **Optimizer** — `SGD` with optional momentum
- **Datasets** — Built-in `make_moons` and `make_circles` generators

### 🔍 Diagnostics
- **`draw_graph(tensor)`** — Print the full computation graph topology
- **`gradient_check(f, inputs)`** — Numerical gradient verification via centered finite differences

---

## Architecture

```
puregrad/
├── tensor.py       Core Tensor class + autograd engine
├── layers.py       Module │ Linear │ ReLU │ Tanh │ Sigmoid
├── nn.py           MLP composite model
├── loss.py         MSELoss │ CrossEntropyLoss
├── optim.py        SGD optimizer
├── data.py         Synthetic dataset generators
├── utils.py        Diagnostics (graph viz, gradient check)
└── tests/
    ├── test_tensor.py      9 tests — autograd verification
    ├── test_layers.py      7 tests — layer/MLP verification
    └── test_training.py    3 tests — end-to-end convergence
```

### How Autograd Works

Every operation on a `Tensor` creates a new node in a dynamically constructed DAG:

```
Input(X) ──┐
            ├── [@] matmul ──> [+] add bias ──> [tanh] ──> [@] ──> [+] ──> [sum] = Loss
Weights(W) ─┘                  Bias(B)
```

1. **Forward Pass** — Operations register new Tensor nodes with parent pointers (`_prev`) and bind a backward closure (`_backward`) computing the local vector-Jacobian product
2. **Backward Pass** — From the loss node, DFS topological sort yields the evaluation order. Walking in reverse, each node's `_backward()` pushes gradients to its parents via `+=`

### Matrix Derivative Formulas (from TRD)

For a linear layer `Y = X @ W + B`:
```
dL/dX = dL/dY @ Wᵀ
dL/dW = Xᵀ @ dL/dY  
dL/dB = Σ_rows(dL/dY)
```

---

## API Reference

### Tensor

```python
t = Tensor(data)              # Create from array/scalar
t.backward()                  # Backpropagate from this node
t.grad                        # Access accumulated gradient
t.shape                       # Shape of underlying data
t.T                           # Transpose
t.relu() / t.tanh()           # Activations
t.sum(axis) / t.mean(axis)    # Reductions
t @ other                     # Matrix multiply
```

### Layers & Models

```python
linear = Linear(in_features=4, out_features=8)
model  = MLP(input_dim=2, hidden_dims=[32, 32], output_dim=1, activation="tanh")

model(x)               # Forward pass (calls model.forward(x))
model.parameters()     # List of all trainable Tensors
model.zero_grad()      # Reset all gradients to zero
```

### Training

```python
loss = MSELoss(predictions, targets)           # Mean Squared Error
loss = CrossEntropyLoss(logits, targets)        # Binary Cross-Entropy

optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer.step()       # Apply gradient updates
optimizer.zero_grad()  # Reset gradients
```

### Diagnostics

```python
from puregrad.utils import draw_graph, gradient_check

draw_graph(loss)                               # Print computation graph
gradient_check(f, [a, b], eps=1e-5, tol=1e-4)  # Numerical verification
```

---

## Tests

Run the full test suite:

```bash
# Phase 1 & 2: Tensor engine (scalar autograd, broadcasting, matmul, activations)
python -m puregrad.tests.test_tensor

# Phase 3: Layers & MLP (shapes, gradient flow, zero_grad)
python -m puregrad.tests.test_layers

# Phase 4: Training pipeline (MSE loss, SGD step, convergence)
python -m puregrad.tests.test_training
```

### Test Results

| Suite | Tests | Status |
|:---|:---:|:---:|
| Tensor Engine | 9/9 | ✅ All passed |
| Layers & MLP | 7/7 | ✅ All passed |
| Training | 3/3 | ✅ All passed |
| **Total** | **19/19** | ✅ |

Key verifications:
- Scalar autograd: `f(a,b) = a*b + b²` matches hand-calculated derivatives
- Matrix gradients match TRD analytical formulas
- Numerical gradient check via finite differences: relative error < 1e-10
- MLP converges on Moons dataset: **loss 0.0199 < 0.05 target, 99.7% accuracy**

---

## Design Philosophy

> **"If you wish to make an apple pie from scratch, you must first invent the universe."** — Carl Sagan

PureGrad exists to make the invisible machinery of deep learning visible. Every gradient, every graph edge, every backward closure is hand-written and inspectable. No black boxes.

**Constraints by design:**
- 🚫 No PyTorch, TensorFlow, JAX, or autograd libraries
- ✅ NumPy permitted only for raw array storage and element-wise math
- ✅ All calculus, graph logic, and backpropagation is proprietary

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  Built with math and determination 🧠
</p>
