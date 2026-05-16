"""
Structural layer modules for building neural networks.

Class hierarchy:
    Module          — base class with zero_grad() and parameters()
    ├── Linear      — fully-connected layer: Y = X @ W + B
    ├── ReLU        — element-wise max(0, x)
    └── Tanh        — element-wise tanh(x)
"""

import numpy as np
from puregrad.tensor import Tensor


class Module:
    """
    Base class for all neural network modules.

    Subclasses must override ``forward()`` and ``parameters()``.
    """

    def forward(self, x):
        raise NotImplementedError

    def parameters(self):
        """Return a list of all trainable Tensor parameters."""
        return []

    def zero_grad(self):
        """Reset gradients of all parameters to zero."""
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def __call__(self, x):
        return self.forward(x)


class Linear(Module):
    """
    Fully-connected linear layer.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool
        If True, adds a learnable bias to the output.

    Forward:
        Y = X @ W + B

    Weight initialization uses He (Kaiming) normal scaling:
        W ~ N(0, sqrt(2 / in_features))
    """

    def __init__(self, in_features, out_features, bias=True):
        # He initialization for weights
        scale = np.sqrt(2.0 / in_features)
        self.weight = Tensor(
            np.random.randn(in_features, out_features).astype(np.float64) * scale
        )
        self.bias = (
            Tensor(np.zeros((1, out_features), dtype=np.float64)) if bias else None
        )
        self._use_bias = bias

    def forward(self, x):
        """
        x : Tensor of shape (batch, in_features)
        returns : Tensor of shape (batch, out_features)
        """
        out = x.matmul(self.weight)
        if self._use_bias:
            out = out + self.bias
        return out

    def parameters(self):
        params = [self.weight]
        if self._use_bias:
            params.append(self.bias)
        return params


class ReLU(Module):
    """Element-wise ReLU activation: max(0, x)."""

    def forward(self, x):
        return x.relu()


class Tanh(Module):
    """Element-wise Tanh activation."""

    def forward(self, x):
        return x.tanh()


class Sigmoid(Module):
    """Element-wise Sigmoid activation."""

    def forward(self, x):
        return x.sigmoid()
