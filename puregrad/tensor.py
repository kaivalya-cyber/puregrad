"""
Core Tensor class with automatic differentiation engine.

Every Tensor is a node in a dynamically constructed Directed Acyclic Graph (DAG).
Operations on Tensors produce new downstream Tensors that track their parent
lineage (_prev) and bind a closure (_backward) that computes the local
vector-Jacobian product for reverse-mode autodiff.

Gradient contract:
    - Gradients ACCUMULATE (+=), never overwrite.
    - Backward is initiated from a scalar loss node with grad seeded to 1.0.
    - Topological sort via DFS guarantees correct evaluation order.
"""

import numpy as np


class Tensor:
    """
    Multi-dimensional array with automatic differentiation support.

    Attributes
    ----------
    data : np.ndarray
        Raw numerical values (forward-pass result).
    grad : np.ndarray
        Accumulated partial derivatives, same shape as data.
    _prev : set[Tensor]
        Parent nodes in the computation graph.
    _op : str
        Label of the operation that produced this node.
    _backward : callable
        Closure computing the local VJP; called during backward().
    """

    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        """Transpose — returns a new Tensor node."""
        return self.transpose()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    # ------------------------------------------------------------------
    # Arithmetic operators (element-wise, with broadcasting)
    # ------------------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            # Gradient flows equally to both parents.
            # Un-broadcast: if shapes were broadcast, sum along added dims.
            self.grad += _unbroadcast(out.grad, self.shape)
            other.grad += _unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return self * (-1.0)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self * (other ** (-1.0))

    def __rtruediv__(self, other):
        return other * (self ** (-1.0))

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "Only scalar exponents supported"
        out = Tensor(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            self.grad += _unbroadcast(
                exponent * (self.data ** (exponent - 1)) * out.grad,
                self.shape,
            )

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Matrix / reduction operations
    # ------------------------------------------------------------------
    def matmul(self, other):
        """
        Matrix multiplication: Y = self @ other.
        Backward:
            dL/d(self)  = dL/dY @ other^T
            dL/d(other) = self^T @ dL/dY
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def sum(self, axis=None, keepdims=False):
        """Reduce-sum with proper gradient expansion."""
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, (self,), "sum")

        def _backward():
            # Expand grad back to the shape of self.data
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis=axis)
            self.grad += np.broadcast_to(g, self.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        """Reduce-mean — delegates to sum then divides by count."""
        s = self.sum(axis=axis, keepdims=keepdims)
        n = self.data.size if axis is None else self.data.shape[axis]
        return s * (1.0 / n)

    def transpose(self):
        """2-D transpose."""
        out = Tensor(self.data.T, (self,), "T")

        def _backward():
            self.grad += out.grad.T

        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), "reshape")

        def _backward():
            self.grad += out.grad.reshape(self.shape)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Activation-level ops (kept here for atomic graph construction)
    # ------------------------------------------------------------------
    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), "relu")

        def _backward():
            self.grad += (self.data > 0).astype(np.float64) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data + 1e-12), (self,), "log")

        def _backward():
            self.grad += out.grad / (self.data + 1e-12)

        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        """
        Numerically stable softmax along given axis.

        softmax(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))

        Gradient: d(softmax)/dx_j = y_j * (delta_ij - y_i)
        => dL/dx = y * (dL/dy - sum(dL/dy * y, axis, keepdims=True))
        """
        # Numerically stable: subtract max before exp
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        exp_vals = np.exp(shifted)
        sum_exp = exp_vals.sum(axis=axis, keepdims=True)
        y = exp_vals / sum_exp

        out = Tensor(y, (self,), "softmax")

        def _backward():
            # y * (dy - sum(dy * y, axis, keepdims))
            dy = out.grad
            dot = (dy * y).sum(axis=axis, keepdims=True)
            self.grad += y * (dy - dot)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Backward pass — reverse-mode automatic differentiation
    # ------------------------------------------------------------------
    def backward(self):
        """
        Execute reverse-mode autodiff from this (scalar) tensor.

        1. Build topological ordering of the computation graph via DFS.
        2. Seed this node's gradient to 1.0.
        3. Walk the topo-order in reverse, invoking each node's _backward().
        """
        # --- Topological sort (DFS) ---
        topo_order = []
        visited = set()

        def _dfs(node):
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    _dfs(parent)
                topo_order.append(node)

        _dfs(self)

        # --- Seed and propagate ---
        self.grad = np.ones_like(self.data)
        for node in reversed(topo_order):
            node._backward()


# ======================================================================
# Module-level helper: un-broadcast gradient to original shape
# ======================================================================
def _unbroadcast(grad, target_shape):
    """
    When two tensors of different shapes are combined via broadcasting,
    the gradient must be summed along the axes that were broadcast.
    """
    # Handle scalar target
    if target_shape == ():
        return grad.sum()

    # Pad target_shape with leading 1s to match grad ndim
    ndim_diff = grad.ndim - len(target_shape)
    padded_shape = (1,) * ndim_diff + target_shape

    # Sum along every axis where target had size 1 (or was added)
    reduce_axes = tuple(
        i for i, (g, t) in enumerate(zip(grad.shape, padded_shape)) if t == 1
    )
    result = grad.sum(axis=reduce_axes, keepdims=True) if reduce_axes else grad

    # Remove the leading dimensions we padded
    return result.reshape(target_shape)
