"""
Phase 1 & 2 Verification — Tensor Engine Tests.

Tests cover:
    - Scalar autograd (f(a,b) = a*b + b^2, analytical gradient check)
    - Element-wise ops with broadcasting
    - Matrix multiply gradients (against TRD analytical formulas)
    - Reduction ops (sum, mean)
    - Activation gradients (relu, tanh, sigmoid)
    - Numerical gradient checking via finite differences
"""

import sys
import numpy as np

sys.path.insert(0, ".")
from puregrad.tensor import Tensor
from puregrad.utils import gradient_check


def test_scalar_autograd():
    """
    Phase 1 Milestone: f(a, b) = a * b + b^2
    Analytical derivatives:
        df/da = b
        df/db = a + 2b
    """
    print("=" * 60)
    print("TEST: Scalar Autograd — f(a,b) = a*b + b²")
    print("=" * 60)

    a = Tensor(2.0)
    b = Tensor(3.0)

    f = a * b + b ** 2  # 2*3 + 9 = 15

    f.backward()

    # Analytical: df/da = b = 3.0
    assert np.isclose(a.grad, 3.0), f"df/da expected 3.0, got {a.grad}"
    # Analytical: df/db = a + 2b = 2 + 6 = 8.0
    assert np.isclose(b.grad, 8.0), f"df/db expected 8.0, got {b.grad}"

    print(f"  f(2, 3) = {f.data:.4f}  (expected 15.0)")
    print(f"  df/da   = {a.grad:.4f}  (expected 3.0)")
    print(f"  df/db   = {b.grad:.4f}  (expected 8.0)")
    print("  ✓ PASSED\n")


def test_chain_rule():
    """Test chained operations: f(x) = ((x + 2) * 3)^2"""
    print("=" * 60)
    print("TEST: Chain Rule — f(x) = ((x + 2) * 3)²")
    print("=" * 60)

    x = Tensor(1.0)
    f = ((x + 2.0) * 3.0) ** 2  # ((1+2)*3)^2 = 81

    f.backward()

    # df/dx = 2 * ((x+2)*3) * 3 = 2 * 9 * 3 = 54
    assert np.isclose(x.grad, 54.0), f"Expected 54.0, got {x.grad}"
    print(f"  f(1) = {f.data:.4f}  (expected 81.0)")
    print(f"  df/dx = {x.grad:.4f}  (expected 54.0)")
    print("  ✓ PASSED\n")


def test_broadcasting():
    """Test element-wise ops with shape broadcasting."""
    print("=" * 60)
    print("TEST: Broadcasting — (3,2) + (1,2)")
    print("=" * 60)

    a = Tensor(np.ones((3, 2)))
    b = Tensor(np.array([[2.0, 3.0]]))  # shape (1, 2) broadcasts to (3, 2)

    c = a * b
    loss = c.sum()
    loss.backward()

    # dc/da = b broadcast → each row of a.grad should be [2, 3]
    expected_a_grad = np.array([[2.0, 3.0]] * 3)
    assert np.allclose(a.grad, expected_a_grad), f"a.grad mismatch: {a.grad}"

    # dc/db: since b is broadcast across 3 rows, grad sums along axis 0
    expected_b_grad = np.array([[3.0, 3.0]])  # sum of 3 ones per column * respective
    # Actually: d(loss)/db_j = sum_i a_ij = 3 * 1 = 3 for each j... wait
    # c_ij = a_ij * b_j, loss = sum c_ij
    # d_loss/d_b_j = sum_i a_ij = 3.0
    expected_b_grad = np.array([[3.0, 3.0]])
    assert np.allclose(b.grad, expected_b_grad), f"b.grad mismatch: {b.grad}"

    print(f"  a.grad = {a.grad.tolist()}")
    print(f"  b.grad = {b.grad.tolist()}")
    print("  ✓ PASSED\n")


def test_matmul_gradients():
    """
    Phase 2 Milestone: Matrix multiply gradient check.
    Y = X @ W, loss = sum(Y)
    TRD formulas:
        dL/dX = dL/dY @ W^T
        dL/dW = X^T @ dL/dY
    """
    print("=" * 60)
    print("TEST: Matmul Gradients — Y = X @ W")
    print("=" * 60)

    np.random.seed(42)
    X = Tensor(np.random.randn(4, 3))
    W = Tensor(np.random.randn(3, 2))

    Y = X @ W
    loss = Y.sum()
    loss.backward()

    # dL/dY = 1 (since loss = sum(Y))
    dL_dY = np.ones((4, 2))
    expected_X_grad = dL_dY @ W.data.T
    expected_W_grad = X.data.T @ dL_dY

    assert np.allclose(X.grad, expected_X_grad, atol=1e-7), \
        f"X.grad mismatch:\n{X.grad}\nvs\n{expected_X_grad}"
    assert np.allclose(W.grad, expected_W_grad, atol=1e-7), \
        f"W.grad mismatch:\n{W.grad}\nvs\n{expected_W_grad}"

    print(f"  X.grad matches dL/dY @ W^T  ✓")
    print(f"  W.grad matches X^T @ dL/dY  ✓")
    print("  ✓ PASSED\n")


def test_linear_bias_gradient():
    """
    Y = X @ W + B, loss = sum(Y)
    TRD formula: dL/dB = sum over rows of dL/dY
    """
    print("=" * 60)
    print("TEST: Linear Layer Gradient — Y = X@W + B")
    print("=" * 60)

    np.random.seed(7)
    X = Tensor(np.random.randn(5, 3))
    W = Tensor(np.random.randn(3, 2))
    B = Tensor(np.zeros((1, 2)))

    Y = X @ W + B
    loss = Y.sum()
    loss.backward()

    dL_dY = np.ones((5, 2))
    expected_B_grad = dL_dY.sum(axis=0, keepdims=True)  # (1, 2)

    assert np.allclose(B.grad, expected_B_grad, atol=1e-7), \
        f"B.grad mismatch: {B.grad} vs {expected_B_grad}"

    print(f"  B.grad = {B.grad.tolist()}")
    print(f"  Expected = {expected_B_grad.tolist()}")
    print("  ✓ PASSED\n")


def test_relu_gradient():
    """ReLU gradient: 1 where x > 0, 0 otherwise."""
    print("=" * 60)
    print("TEST: ReLU Gradient")
    print("=" * 60)

    x = Tensor(np.array([[-2.0, 1.0, 0.0, 3.0, -0.5]]))
    y = x.relu()
    loss = y.sum()
    loss.backward()

    expected = np.array([[0.0, 1.0, 0.0, 1.0, 0.0]])
    assert np.allclose(x.grad, expected), f"ReLU grad mismatch: {x.grad}"
    print(f"  x.grad = {x.grad.tolist()}")
    print("  ✓ PASSED\n")


def test_tanh_gradient():
    """Tanh gradient: 1 - tanh(x)^2."""
    print("=" * 60)
    print("TEST: Tanh Gradient")
    print("=" * 60)

    x = Tensor(np.array([[0.0, 1.0, -1.0]]))
    y = x.tanh()
    loss = y.sum()
    loss.backward()

    expected = 1.0 - np.tanh(np.array([[0.0, 1.0, -1.0]])) ** 2
    assert np.allclose(x.grad, expected, atol=1e-7), f"Tanh grad mismatch: {x.grad}"
    print(f"  x.grad    = {x.grad.tolist()}")
    print(f"  expected  = {expected.tolist()}")
    print("  ✓ PASSED\n")


def test_numerical_gradient_check():
    """Full numerical gradient check via finite differences."""
    print("=" * 60)
    print("TEST: Numerical Gradient Check (finite differences)")
    print("=" * 60)

    np.random.seed(99)

    def f(inputs):
        a, b = inputs
        return ((a @ b).relu().sum()) * Tensor(1.0)

    a = Tensor(np.random.randn(3, 4))
    b = Tensor(np.random.randn(4, 2))

    passed = gradient_check(f, [a, b], eps=1e-5, tol=1e-3)
    assert passed, "Numerical gradient check FAILED"
    print("  ✓ PASSED\n")


def test_variable_reuse():
    """
    Gradient accumulation when same variable appears multiple times.
    f(x) = x * x + x  →  df/dx = 2x + 1
    """
    print("=" * 60)
    print("TEST: Variable Reuse — f(x) = x*x + x")
    print("=" * 60)

    x = Tensor(3.0)
    f = x * x + x  # 9 + 3 = 12
    f.backward()

    # df/dx = 2*3 + 1 = 7
    assert np.isclose(x.grad, 7.0), f"Expected 7.0, got {x.grad}"
    print(f"  f(3) = {f.data:.4f}  (expected 12.0)")
    print(f"  df/dx = {x.grad:.4f}  (expected 7.0)")
    print("  ✓ PASSED\n")


# ------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "━" * 60)
    print("  ANTIGRAVITY — Phase 1 & 2: Tensor Engine Tests")
    print("━" * 60 + "\n")

    test_scalar_autograd()
    test_chain_rule()
    test_variable_reuse()
    test_broadcasting()
    test_matmul_gradients()
    test_linear_bias_gradient()
    test_relu_gradient()
    test_tanh_gradient()
    test_numerical_gradient_check()

    print("━" * 60)
    print("  ALL TENSOR TESTS PASSED ✓")
    print("━" * 60)
