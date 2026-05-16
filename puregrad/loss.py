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
