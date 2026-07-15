"""
Optimizers for updating trainable parameters.

SGD  — Stochastic Gradient Descent with optional momentum.
Adam — Adaptive Moment Estimation (Kingma & Ba, 2015).
"""

import numpy as np


class SGD:
    """
    Stochastic Gradient Descent optimizer.

    Parameters
    ----------
    params : list[Tensor]
        Trainable parameters returned by model.parameters().
    lr : float
        Learning rate.
    momentum : float
        Momentum factor (default 0.0 = vanilla SGD).
    weight_decay : float
        L2 regularization strength (default 0.0 = no decay).

    Usage
    -----
    >>> optimizer = SGD(model.parameters(), lr=0.01)
    >>> optimizer.step()
    """

    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        # Velocity buffers for momentum
        self._velocities = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        """Apply a single gradient descent update to all parameters."""
        for i, p in enumerate(self.params):
            g = p.grad
            if self.weight_decay > 0:
                g = g + self.weight_decay * p.data
            if self.momentum > 0:
                self._velocities[i] = (
                    self.momentum * self._velocities[i] - self.lr * g
                )
                p.data += self._velocities[i]
            else:
                p.data -= self.lr * g

    def zero_grad(self):
        """Reset gradients of all parameters to zero."""
        for p in self.params:
            p.grad = np.zeros_like(p.data)


class Adam:
    """
    Adam (Adaptive Moment Estimation) optimizer.

    Combines momentum (first moment) with adaptive per-parameter
    learning rates (second moment). Includes bias correction for
    the first few steps.

    Parameters
    ----------
    params : list[Tensor]
        Trainable parameters.
    lr : float
        Learning rate (default 0.001).
    betas : tuple[float, float]
        Coefficients for running averages of gradient and its square
        (default (0.9, 0.999)).
    eps : float
        Term added for numerical stability (default 1e-8).
    weight_decay : float
        L2 regularization strength (default 0.0 = no decay).

    Usage
    -----
    >>> optimizer = Adam(model.parameters(), lr=0.001)
    >>> optimizer.step()
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0  # timestep counter for bias correction

        # First moment (momentum) and second moment (RMS) buffers
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        """Apply a single Adam update to all parameters."""
        self.t += 1
        beta1, beta2 = self.betas

        for i, p in enumerate(self.params):
            g = p.grad
            if self.weight_decay > 0:
                g = g + self.weight_decay * p.data

            # Update biased first moment estimate
            self._m[i] = beta1 * self._m[i] + (1.0 - beta1) * g
            # Update biased second raw moment estimate
            self._v[i] = beta2 * self._v[i] + (1.0 - beta2) * g * g

            # Bias-corrected moment estimates
            m_hat = self._m[i] / (1.0 - beta1 ** self.t)
            v_hat = self._v[i] / (1.0 - beta2 ** self.t)

            # Parameter update
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Reset gradients of all parameters to zero."""
        for p in self.params:
            p.grad = np.zeros_like(p.data)
