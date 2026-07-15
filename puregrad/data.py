"""
Synthetic dataset generators and data utilities.

Generators return (X, y) as plain numpy arrays.
DataLoader yields mini-batches for training loops.
"""

import numpy as np


class DataLoader:
    """
    Mini-batch data loader with optional shuffling.

    Parameters
    ----------
    X : np.ndarray or Tensor
        Input features of shape (N, D).
    y : np.ndarray or Tensor
        Target labels of shape (N, 1) or (N,).
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        If True, shuffle the data at the start of each epoch.

    Usage
    -----
    >>> X_np, y_np = make_moons(n_samples=300)
    >>> loader = DataLoader(X_np, y_np, batch_size=32, shuffle=True)
    >>> for X_batch, y_batch in loader:
    ...     # X_batch, y_batch are numpy arrays
    ...     pred = model(Tensor(X_batch))
    """

    def __init__(self, X, y, batch_size=32, shuffle=True):
        # Accept both Tensor and np.ndarray inputs
        self.X = X.data if hasattr(X, 'data') else np.asarray(X)
        self.y = y.data if hasattr(y, 'data') else np.asarray(y)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = self.X.shape[0]

    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, self.n_samples, self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            yield self.X[batch_idx], self.y[batch_idx]

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size


def train_test_split(X, y, test_size=0.2, seed=42):
    """
    Split data into training and test sets.

    Parameters
    ----------
    X : np.ndarray
        Input features.
    y : np.ndarray
        Target labels.
    test_size : float
        Fraction of data to reserve for testing (default 0.2).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (X_train, X_test, y_train, y_test)
    """
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    n_test = int(n * test_size)
    indices = rng.permutation(n)
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


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
