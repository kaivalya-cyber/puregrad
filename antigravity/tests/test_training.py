"""
Phase 4 Verification — End-to-End Training Test.
"""
import sys
import numpy as np
sys.path.insert(0, ".")
from antigravity.tensor import Tensor
from antigravity.nn import MLP
from antigravity.loss import MSELoss
from antigravity.optim import SGD
from antigravity.data import make_moons


def test_mse_loss():
    print("=" * 60)
    print("TEST: MSELoss Computation")
    print("=" * 60)
    pred = Tensor(np.array([[1.0], [2.0], [3.0]]))
    target = Tensor(np.array([[1.0], [2.0], [3.0]]))
    loss = MSELoss(pred, target)
    assert np.isclose(loss.data, 0.0), f"Expected 0, got {loss.data}"
    print(f"  Perfect prediction loss: {loss.data.flat[0]:.6f}")

    pred2 = Tensor(np.array([[0.0], [0.0], [0.0]]))
    target2 = Tensor(np.array([[1.0], [2.0], [3.0]]))
    loss2 = MSELoss(pred2, target2)
    expected = (1 + 4 + 9) / 3.0
    assert np.isclose(loss2.data, expected, atol=1e-4)
    print(f"  Non-zero loss: {loss2.data.flat[0]:.6f} (expected {expected:.4f})")
    print("  PASSED\n")


def test_sgd_step():
    print("=" * 60)
    print("TEST: SGD Optimizer Step")
    print("=" * 60)
    w = Tensor(np.array([[1.0, 2.0]]))
    w.grad = np.array([[0.1, -0.2]])
    optimizer = SGD([w], lr=1.0)
    optimizer.step()
    expected = np.array([[0.9, 2.2]])
    assert np.allclose(w.data, expected)
    print(f"  After step: {w.data.tolist()} (expected {expected.tolist()})")
    print("  PASSED\n")


def test_training_convergence():
    print("=" * 60)
    print("TEST: Training Convergence on Moons (500 steps)")
    print("=" * 60)
    np.random.seed(42)
    X_np, y_np = make_moons(n_samples=200, noise=0.1, seed=42)
    X_train = Tensor(X_np)
    y_train = Tensor(y_np)

    model = MLP(input_dim=2, hidden_dims=[16, 16], output_dim=1, activation="tanh")
    optimizer = SGD(model.parameters(), lr=0.05)

    for epoch in range(500):
        model.zero_grad()
        predictions = model(X_train)
        loss = MSELoss(predictions, y_train)
        loss.backward()
        optimizer.step()
        loss_val = float(loss.data.flat[0])
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1:>4d} | Loss: {loss_val:.6f}")

    print(f"\n  Final loss: {loss_val:.6f}  (target < 0.05)")
    assert loss_val < 0.05, f"Did not converge: {loss_val:.6f}"
    print("  CONVERGENCE ACHIEVED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  ANTIGRAVITY Phase 4: Training Tests")
    print("=" * 60 + "\n")
    test_mse_loss()
    test_sgd_step()
    test_training_convergence()
    print("=" * 60)
    print("  ALL TRAINING TESTS PASSED")
    print("=" * 60)
