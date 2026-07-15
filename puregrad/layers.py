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


class BatchNorm1d(Module):
    """
    1D Batch Normalization for 2D inputs of shape (N, D).

    Normalizes across the batch dimension (axis=0) for each feature.
    Uses running mean/variance for evaluation mode.

    Parameters
    ----------
    num_features : int
        Number of features (D) in the input.
    eps : float
        Small constant for numerical stability (default 1e-5).
    momentum : float
        Momentum for running mean/variance updates (default 0.1).

    Forward (training):
        x_hat = (x - mu) / sqrt(var + eps)
        y = gamma * x_hat + beta
        Updates running stats.

    Forward (eval):
        Uses running mean/variance instead of batch statistics.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = Tensor(np.ones(num_features, dtype=np.float64))
        self.beta = Tensor(np.zeros(num_features, dtype=np.float64))

        # Running statistics (not tensors — just numpy arrays)
        self.running_mean = np.zeros(num_features, dtype=np.float64)
        self.running_var = np.ones(num_features, dtype=np.float64)

    def forward(self, x):
        x_data = x.data
        N = x_data.shape[0]

        if self._training:
            # Batch statistics
            mu = np.mean(x_data, axis=0)
            x_mu = x_data - mu
            var = np.mean(x_mu ** 2, axis=0)

            # Update running statistics
            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * var
        else:
            mu = self.running_mean
            x_mu = x_data - mu
            var = self.running_var

        std_inv = 1.0 / np.sqrt(var + self.eps)
        x_hat = x_mu * std_inv

        # Scale and shift
        y_data = self.gamma.data * x_hat + self.beta.data

        # Wire the DAG
        out = Tensor(y_data, (x, self.gamma, self.beta), "batchnorm1d")

        # Save intermediates for backward (capture by value)
        _x_mu = x_mu
        _std_inv = std_inv
        _x_hat = x_hat
        _N = N

        def _backward():
            dy = out.grad

            # Gradients w.r.t learnable parameters
            d_beta = np.sum(dy, axis=0)
            d_gamma = np.sum(dy * _x_hat, axis=0)

            # Gradients w.r.t input
            d_x_hat = dy * self.gamma.data
            d_var = np.sum(d_x_hat * _x_mu * -0.5 * (_std_inv ** 3), axis=0)
            d_mu = np.sum(d_x_hat * -_std_inv, axis=0)
            d_x = (d_x_hat * _std_inv) + (d_var * 2.0 * _x_mu / _N) + (d_mu / _N)

            # Accumulate gradients
            self.beta.grad += d_beta
            self.gamma.grad += d_gamma
            x.grad += d_x

        out._backward = _backward
        return out

    def parameters(self):
        return [self.gamma, self.beta]
