"""
Synthetic dataset generators for verification testing.

All generators return (X, y) as plain numpy arrays.
Wrap them in Tensor() before feeding to the model.
"""

import numpy as np


def make_moons(n_samples=300, noise=0.1, seed=42):
    """
    Generate a two-class 'moons' dataset (non-linear, interleaved crescents).

    Parameters
    ----------
    n_samples : int
        Total number of samples (split evenly between classes).
    noise : float
        Standard deviation of Gaussian noise added to the data.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 2)
    y : np.ndarray, shape (n_samples, 1)  — binary labels {0, 1}
    """
    rng = np.random.RandomState(seed)
    n_half = n_samples // 2

    # Upper moon
    theta_upper = np.linspace(0, np.pi, n_half)
    x_upper = np.column_stack([np.cos(theta_upper), np.sin(theta_upper)])

    # Lower moon (shifted and flipped)
    theta_lower = np.linspace(0, np.pi, n_samples - n_half)
    x_lower = np.column_stack(
        [1.0 - np.cos(theta_lower), 1.0 - np.sin(theta_lower) - 0.5]
    )

    X = np.vstack([x_upper, x_lower]).astype(np.float64)
    y = np.hstack(
        [np.zeros(n_half), np.ones(n_samples - n_half)]
    ).reshape(-1, 1).astype(np.float64)

    # Add Gaussian noise
    X += rng.randn(*X.shape) * noise

    # Shuffle
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


def make_circles(n_samples=300, noise=0.05, factor=0.5, seed=42):
    """
    Generate a two-class 'circles' dataset (concentric circles).

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    noise : float
        Standard deviation of Gaussian noise.
    factor : float
        Scale factor between inner and outer circle (0 < factor < 1).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 2)
    y : np.ndarray, shape (n_samples, 1)
    """
    rng = np.random.RandomState(seed)
    n_half = n_samples // 2

    theta_outer = np.linspace(0, 2 * np.pi, n_half, endpoint=False)
    theta_inner = np.linspace(0, 2 * np.pi, n_samples - n_half, endpoint=False)

    x_outer = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])
    x_inner = np.column_stack(
        [np.cos(theta_inner) * factor, np.sin(theta_inner) * factor]
    )

    X = np.vstack([x_outer, x_inner]).astype(np.float64)
    y = np.hstack(
        [np.zeros(n_half), np.ones(n_samples - n_half)]
    ).reshape(-1, 1).astype(np.float64)

    X += rng.randn(*X.shape) * noise

    idx = rng.permutation(n_samples)
    return X[idx], y[idx]
