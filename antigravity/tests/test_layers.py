"""
Phase 3 Verification — Layer & MLP Tests.

Tests cover:
    - Linear layer forward shape integrity
    - Linear layer gradient flow
    - MLP forward pass shape
    - MLP parameters collection
    - Module zero_grad
"""

import sys
import numpy as np

sys.path.insert(0, ".")
from antigravity.tensor import Tensor
from antigravity.layers import Linear, ReLU, Tanh
from antigravity.nn import MLP


def test_linear_forward_shape():
    """Linear(3, 5) with batch_size=8 should produce (8, 5)."""
    print("=" * 60)
    print("TEST: Linear Forward Shape — (8, 3) → (8, 5)")
    print("=" * 60)

    layer = Linear(3, 5)
    x = Tensor(np.random.randn(8, 3))
    y = layer(x)

    assert y.shape == (8, 5), f"Expected (8, 5), got {y.shape}"
    print(f"  Output shape: {y.shape}")
    print("  ✓ PASSED\n")


def test_linear_gradient_flow():
    """Verify gradients flow through Linear layer to weight and bias."""
    print("=" * 60)
    print("TEST: Linear Gradient Flow")
    print("=" * 60)

    np.random.seed(42)
    layer = Linear(4, 2)
    x = Tensor(np.random.randn(3, 4))

    y = layer(x)
    loss = y.sum()
    loss.backward()

    assert not np.allclose(layer.weight.grad, 0), "Weight grads should be nonzero"
    assert not np.allclose(layer.bias.grad, 0), "Bias grads should be nonzero"
    print(f"  Weight grad norm: {np.linalg.norm(layer.weight.grad):.6f}")
    print(f"  Bias grad norm:   {np.linalg.norm(layer.bias.grad):.6f}")
    print("  ✓ PASSED\n")


def test_mlp_forward_shape():
    """MLP(2, [16, 16], 1) should map (32, 2) → (32, 1)."""
    print("=" * 60)
    print("TEST: MLP Forward Shape — (32, 2) → (32, 1)")
    print("=" * 60)

    model = MLP(input_dim=2, hidden_dims=[16, 16], output_dim=1)
    x = Tensor(np.random.randn(32, 2))
    y = model(x)

    assert y.shape == (32, 1), f"Expected (32, 1), got {y.shape}"
    print(f"  Output shape: {y.shape}")
    print("  ✓ PASSED\n")


def test_mlp_parameters():
    """Verify MLP collects all trainable parameters."""
    print("=" * 60)
    print("TEST: MLP Parameters Collection")
    print("=" * 60)

    model = MLP(input_dim=2, hidden_dims=[16, 16], output_dim=1)
    params = model.parameters()

    # 3 Linear layers × (weight + bias) = 6 parameter tensors
    assert len(params) == 6, f"Expected 6 parameter tensors, got {len(params)}"

    total_params = sum(p.data.size for p in params)
    print(f"  Number of parameter tensors: {len(params)}")
    print(f"  Total trainable parameters:  {total_params}")
    # 2*16 + 16 + 16*16 + 16 + 16*1 + 1 = 32+16+256+16+16+1 = 337
    print("  ✓ PASSED\n")


def test_zero_grad():
    """Verify zero_grad resets all parameter gradients."""
    print("=" * 60)
    print("TEST: Module.zero_grad()")
    print("=" * 60)

    model = MLP(input_dim=2, hidden_dims=[8], output_dim=1)
    x = Tensor(np.random.randn(4, 2))

    # Forward + backward to populate grads
    y = model(x)
    loss = y.sum()
    loss.backward()

    # Ensure grads are nonzero
    has_nonzero = any(not np.allclose(p.grad, 0) for p in model.parameters())
    assert has_nonzero, "Should have nonzero grads before zero_grad"

    # Zero them
    model.zero_grad()

    all_zero = all(np.allclose(p.grad, 0) for p in model.parameters())
    assert all_zero, "All grads should be zero after zero_grad()"
    print("  ✓ PASSED\n")


def test_relu_activation():
    """ReLU layer wraps Tensor.relu correctly."""
    print("=" * 60)
    print("TEST: ReLU Activation Layer")
    print("=" * 60)

    relu = ReLU()
    x = Tensor(np.array([[-1.0, 0.5, 0.0, 2.0]]))
    y = relu(x)

    expected = np.array([[0.0, 0.5, 0.0, 2.0]])
    assert np.allclose(y.data, expected), f"ReLU output mismatch: {y.data}"
    print(f"  Input:  {x.data.tolist()}")
    print(f"  Output: {y.data.tolist()}")
    print("  ✓ PASSED\n")


def test_tanh_activation():
    """Tanh layer wraps Tensor.tanh correctly."""
    print("=" * 60)
    print("TEST: Tanh Activation Layer")
    print("=" * 60)

    tanh = Tanh()
    x = Tensor(np.array([[0.0, 1.0, -1.0]]))
    y = tanh(x)

    expected = np.tanh(np.array([[0.0, 1.0, -1.0]]))
    assert np.allclose(y.data, expected), f"Tanh output mismatch: {y.data}"
    print(f"  Input:  {x.data.tolist()}")
    print(f"  Output: {y.data.tolist()}")
    print("  ✓ PASSED\n")


# ------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "━" * 60)
    print("  ANTIGRAVITY — Phase 3: Layer & MLP Tests")
    print("━" * 60 + "\n")

    test_linear_forward_shape()
    test_linear_gradient_flow()
    test_mlp_forward_shape()
    test_mlp_parameters()
    test_zero_grad()
    test_relu_activation()
    test_tanh_activation()

    print("━" * 60)
    print("  ALL LAYER TESTS PASSED ✓")
    print("━" * 60)
