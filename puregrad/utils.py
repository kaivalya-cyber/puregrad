"""
Diagnostic and visualization utilities.

- draw_graph()      : Print the topology of the computation graph.
- gradient_check()  : Numerical gradient verification via finite differences.
"""

import numpy as np
from puregrad.tensor import Tensor


def draw_graph(tensor, indent=0, visited=None):
    """
    Recursively print the computation graph topology rooted at `tensor`.

    Parameters
    ----------
    tensor : Tensor
        The root node (typically the loss).
    indent : int
        Current indentation depth (used for recursion).
    visited : set or None
        Tracks visited nodes to avoid duplicates in diamond graphs.

    Example output
    --------------
    [sum] shape=(,) data=0.0234
      └─ [*] shape=(300, 1)
           └─ [+] shape=(300, 1)
                └─ [@] shape=(300, 16)
                ...
    """
    if visited is None:
        visited = set()

    node_id = id(tensor)
    prefix = "  " * indent
    connector = "└─ " if indent > 0 else ""

    shape_str = str(tensor.shape)
    data_preview = ""
    if tensor.data.size == 1:
        data_preview = f" data={tensor.data.flat[0]:.6f}"

    label = tensor._op if tensor._op else "leaf"
    print(f"{prefix}{connector}[{label}] shape={shape_str}{data_preview}")

    if node_id in visited:
        print(f"{prefix}  (seen)")
        return
    visited.add(node_id)

    for parent in tensor._prev:
        draw_graph(parent, indent + 1, visited)


def gradient_check(f, inputs, eps=1e-5, tol=1e-4):
    """
    Numerical gradient check via centered finite differences.

    For each element of each input tensor, computes:
        numerical_grad = (f(..., x_i + eps, ...) - f(..., x_i - eps, ...)) / (2 * eps)
    and compares against the analytical gradient from .backward().

    Parameters
    ----------
    f : callable
        A function that takes a list of Tensors and returns a scalar Tensor.
    inputs : list[Tensor]
        Input tensors to differentiate with respect to.
    eps : float
        Finite difference step size.
    tol : float
        Maximum allowed relative error.

    Returns
    -------
    bool — True if all gradients pass, False otherwise.
    """
    # Forward + backward for analytical gradients
    for inp in inputs:
        inp.grad = np.zeros_like(inp.data)
    loss = f(inputs)
    loss.backward()

    # Save ALL analytical gradients upfront (before any perturbation
    # loop zeros them as a side-effect).
    analytical_grads = [inp.grad.copy() for inp in inputs]

    all_pass = True

    for idx, inp in enumerate(inputs):
        analytical = analytical_grads[idx]
        numerical = np.zeros_like(inp.data)

        it = np.nditer(inp.data, flags=["multi_index"])
        while not it.finished:
            mi = it.multi_index
            original = inp.data[mi]

            # f(x + eps)
            inp.data[mi] = original + eps
            # Must zero grads and recompute each time
            for t in inputs:
                t.grad = np.zeros_like(t.data)
            loss_plus = f(inputs).data.flat[0]

            # f(x - eps)
            inp.data[mi] = original - eps
            for t in inputs:
                t.grad = np.zeros_like(t.data)
            loss_minus = f(inputs).data.flat[0]

            # Restore
            inp.data[mi] = original

            numerical[mi] = (loss_plus - loss_minus) / (2 * eps)
            it.iternext()

        # Compare
        diff = np.abs(analytical - numerical)
        denom = np.maximum(np.abs(analytical) + np.abs(numerical), 1e-8)
        rel_error = diff / denom
        max_err = rel_error.max()

        if max_err > tol:
            print(
                f"  ✗ GRADIENT CHECK FAILED for input[{idx}]: "
                f"max relative error = {max_err:.6e}"
            )
            all_pass = False
        else:
            print(
                f"  ✓ input[{idx}] passed: max relative error = {max_err:.6e}"
            )

    return all_pass
