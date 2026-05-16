"""
Optimizers for updating trainable parameters.

SGD — Stochastic Gradient Descent with optional momentum.
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

    Usage
    -----
    >>> optimizer = SGD(model.parameters(), lr=0.01)
    >>> optimizer.step()
    """

    def __init__(self, params, lr=0.01, momentum=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        # Velocity buffers for momentum
        self._velocities = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        """Apply a single gradient descent update to all parameters."""
        for i, p in enumerate(self.params):
            if self.momentum > 0:
                self._velocities[i] = (
                    self.momentum * self._velocities[i] - self.lr * p.grad
                )
                p.data += self._velocities[i]
            else:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        """Reset gradients of all parameters to zero."""
        for p in self.params:
            p.grad = np.zeros_like(p.data)
