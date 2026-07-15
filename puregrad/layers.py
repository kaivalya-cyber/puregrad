"""
Structural layer modules for building neural networks.

Class hierarchy:
    Module          — base class with zero_grad(), parameters(), train/eval mode
    ├── Linear      — fully-connected layer: Y = X @ W + B
    ├── ReLU        — element-wise max(0, x)
    ├── Tanh        — element-wise tanh(x)
    ├── Sigmoid     — element-wise sigmoid(x)
    └── Dropout     — randomly zeroes elements during training
"""

import numpy as np
from puregrad.tensor import Tensor


class Module:
    """
    Base class for all neural network modules.

    Subclasses must override ``forward()`` and ``parameters()``.
    """

    def __init__(self):
        self._training = True

    def forward(self, x):
        raise NotImplementedError

    def parameters(self):
        """Return a list of all trainable Tensor parameters."""
        return []

    def zero_grad(self):
        """Reset gradients of all parameters to zero."""
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def train(self):
        """Set the module and all submodules to training mode."""
        self._training = True
        for attr in vars(self).values():
            if isinstance(attr, Module):
                attr.train()
        return self

    def eval(self):
        """Set the module and all submodules to evaluation mode."""
        self._training = False
        for attr in vars(self).values():
            if isinstance(attr, Module):
                attr.eval()
        return self

    def state_dict(self):
        """
        Return all parameter data as a flat dictionary.

        Returns
        -------
        dict[str, np.ndarray] — parameter name → numpy array.
        """
        state = {}
        for i, p in enumerate(self.parameters()):
            state[f"param_{i}"] = p.data.copy()
        return state

    def load_state_dict(self, state_dict):
        """
        Load parameter data from a dictionary.

        Parameters
        ----------
        state_dict : dict[str, np.ndarray]
            Dictionary mapping parameter names to numpy arrays.
        """
        for i, p in enumerate(self.parameters()):
            key = f"param_{i}"
            if key in state_dict:
                p.data = state_dict[key].copy()

    def save(self, filepath):
        """
        Save model parameters to a .npz file.

        Parameters
        ----------
        filepath : str
            Path to save the model (e.g. 'model.npz').
        """
        state = self.state_dict()
        np.savez(filepath, **state)

    def load(self, filepath):
        """
        Load model parameters from a .npz file.

        Parameters
        ----------
        filepath : str
            Path to the saved model (e.g. 'model.npz').
        """
        state = dict(np.load(filepath))
        self.load_state_dict(state)

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
        super().__init__()
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


class Dropout(Module):
    """
    Dropout regularization layer.

    During training, randomly zeroes each element with probability `p`
    and scales survivors by 1/(1-p). During evaluation, this is a no-op.

    Parameters
    ----------
    p : float
        Probability of dropping a neuron (default 0.5).

    Usage
    -----
    >>> dropout = Dropout(p=0.2)
    >>> model.eval()  # turns off dropout
    >>> model.train()  # turns on dropout
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self._mask = None

    def forward(self, x):
        if not self._training or self.p == 0.0:
            return x

        # Generate mask: Bernoulli with probability (1-p)
        mask = (np.random.rand(*x.shape) > self.p).astype(np.float64)
        scale = 1.0 / (1.0 - self.p)
        self._mask = mask

        # Create output tensor: scaled masked values
        out = Tensor(x.data * mask * scale, (x,), "dropout")

        def _backward():
            # Gradient only flows through non-dropped elements
            x.grad += mask * scale * out.grad

        out._backward = _backward
        return out


class LeakyReLU(Module):
    """
    Leaky ReLU activation: f(x) = x if x > 0, else alpha * x.

    Parameters
    ----------
    alpha : float
        Slope for negative inputs (default 0.01).
    """

    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return x.leaky_relu(self.alpha)


class GELU(Module):
    """
    Gaussian Error Linear Unit (GELU) activation.

    Uses the tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    """

    def forward(self, x):
        return x.gelu()
