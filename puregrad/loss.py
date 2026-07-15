"""
Loss functions for training neural networks.

All loss functions accept Tensor inputs and return a scalar Tensor
suitable for calling .backward() on.
"""

import numpy as np
from puregrad.tensor import Tensor


def MSELoss(predictions, targets):
    """
    Mean Squared Error loss.

    L = (1/N) * Σ (predictions - targets)²

    Parameters
    ----------
    predictions : Tensor  — shape (N, D) or (N, 1)
    targets : Tensor       — same shape as predictions

    Returns
    -------
    Tensor — scalar loss value.
    """
    if not isinstance(targets, Tensor):
        targets = Tensor(targets)
    diff = predictions - targets
    sq = diff * diff
    return sq.mean()


def CrossEntropyLoss(logits, targets):
    """
    Binary Cross-Entropy loss with built-in sigmoid.

    L = -(1/N) * Σ [ t * log(σ(z)) + (1-t) * log(1 - σ(z)) ]

    Parameters
    ----------
    logits : Tensor  — raw un-activated outputs, shape (N, 1)
    targets : Tensor — binary labels 0/1, shape (N, 1)

    Returns
    -------
    Tensor — scalar loss value.
    """
    if not isinstance(targets, Tensor):
        targets = Tensor(targets)
    probs = logits.sigmoid()
    # Numerically stable BCE via log with epsilon protection
    loss = targets * probs.log() + (1.0 - targets) * (1.0 - probs).log()
    return (loss * (-1.0)).mean()


def SoftmaxCrossEntropyLoss(logits, targets):
    """
    Multi-class Cross-Entropy loss with built-in softmax.

    L = -(1/N) * Σ log(softmax(logits)[i, target_i])

    Parameters
    ----------
    logits : Tensor  — raw logits, shape (N, C)
    targets : Tensor or np.ndarray — integer class labels, shape (N,) or (N, 1)

    Returns
    -------
    Tensor — scalar loss value.

    Example
    -------
    >>> logits = Tensor(np.random.randn(32, 10))  # 32 samples, 10 classes
    >>> targets = np.array([3, 7, 1, ...])          # ground-truth class indices
    >>> loss = SoftmaxCrossEntropyLoss(logits, targets)
    """
    if isinstance(targets, Tensor):
        targets = targets.data
    targets = np.asarray(targets, dtype=np.int64).flatten()

    N = logits.data.shape[0]
    probs = logits.softmax(axis=1)

    # Gather probabilities for the correct classes: probs[i, targets[i]]
    # Use advanced indexing; we wrap result in a Tensor for autograd
    correct_probs = Tensor(probs.data[np.arange(N), targets].reshape(-1, 1),
                           (probs,), "nll")

    def _backward():
        # Gradient of NLL w.r.t softmax probs:
        # d(-log(probs[correct]))/dprobs[i,j] = -delta[targets[i],j] / probs[i,j]
        nll_grad = np.zeros_like(probs.data)
        nll_grad[np.arange(N), targets] = -1.0 / correct_probs.data.flatten()
        probs.grad += nll_grad * correct_probs.grad

    correct_probs._backward = _backward

    # Negative log-likelihood
    nll = -correct_probs.log()
    return nll.mean()
