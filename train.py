#!/usr/bin/env python3
"""
Antigravity Framework — Main Training Script

Trains an MLP on the synthetic Moons dataset using the Antigravity
deep learning framework (zero external ML dependencies).

Usage:
    python train.py
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from puregrad import Tensor, MLP, MSELoss, SGD
from puregrad.data import make_moons
from puregrad.utils import draw_graph


def main():
    # ── Configuration ──────────────────────────────────────────────
    SEED = 42
    N_SAMPLES = 300
    NOISE = 0.1
    HIDDEN_DIMS = [32, 32]
    ACTIVATION = "tanh"
    LR = 0.05
    EPOCHS = 500
    LOSS_TARGET = 0.05

    np.random.seed(SEED)

    # ── Banner ─────────────────────────────────────────────────────
    print()
    print("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    print("┃       ANTIGRAVITY — Custom Deep Learning Framework      ┃")
    print("┃           Zero Dependencies · Pure Autograd             ┃")
    print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
    print()

    # ── Dataset ────────────────────────────────────────────────────
    print("▸ Generating Moons dataset...")
    X_np, y_np = make_moons(n_samples=N_SAMPLES, noise=NOISE, seed=SEED)
    X_train = Tensor(X_np)
    y_train = Tensor(y_np)
    print(f"  Samples: {N_SAMPLES}  |  Features: {X_np.shape[1]}  |  Noise: {NOISE}")
    print()

    # ── Model ──────────────────────────────────────────────────────
    print("▸ Building MLP...")
    model = MLP(
        input_dim=2,
        hidden_dims=HIDDEN_DIMS,
        output_dim=1,
        activation=ACTIVATION,
    )
    params = model.parameters()
    total_params = sum(p.data.size for p in params)
    print(f"  Architecture: 2 → {' → '.join(map(str, HIDDEN_DIMS))} → 1")
    print(f"  Activation:   {ACTIVATION}")
    print(f"  Parameters:   {total_params}")
    print()

    # ── Optimizer ──────────────────────────────────────────────────
    optimizer = SGD(params, lr=LR)
    print(f"▸ Optimizer: SGD (lr={LR})")
    print()

    # ── Training Loop ──────────────────────────────────────────────
    print("▸ Training...")
    print("  ┌──────────┬────────────────┬──────────────────────────┐")
    print("  │  Epoch   │     Loss       │       Progress           │")
    print("  ├──────────┼────────────────┼──────────────────────────┤")

    losses = []
    for epoch in range(EPOCHS):
        # Zero gradients
        model.zero_grad()

        # Forward pass
        predictions = model(X_train)

        # Compute loss
        loss = MSELoss(predictions, y_train)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        loss_val = float(loss.data.flat[0])
        losses.append(loss_val)

        # Console visualization log
        if (epoch + 1) % 25 == 0 or epoch == 0:
            bar_len = 24
            progress = min(1.0, (1.0 - loss_val / losses[0]) if losses[0] > 0 else 0)
            filled = int(bar_len * max(0, progress))
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"  │  {epoch+1:>5d}   │  {loss_val:>12.6f}  │ {bar} │")

    print("  └──────────┴────────────────┴──────────────────────────┘")
    print()

    # ── Results ────────────────────────────────────────────────────
    final_loss = losses[-1]
    converged = final_loss < LOSS_TARGET

    if converged:
        print(f"  ✅ CONVERGED — Final Loss: {final_loss:.6f} < {LOSS_TARGET}")
    else:
        print(f"  ❌ NOT CONVERGED — Final Loss: {final_loss:.6f} > {LOSS_TARGET}")

    # ── Accuracy ───────────────────────────────────────────────────
    final_pred = model(X_train)
    pred_labels = (final_pred.data > 0.5).astype(np.float64)
    accuracy = np.mean(pred_labels == y_np) * 100
    print(f"  📊 Training Accuracy: {accuracy:.1f}%")
    print()

    # ── Computation Graph (first few levels) ───────────────────────
    print("▸ Computation Graph (loss node):")
    # Show just the loss topology
    draw_graph(loss)
    print()

    # ── Loss Curve (ASCII) ─────────────────────────────────────────
    print("▸ Loss Curve:")
    _plot_ascii(losses)
    print()

    print("━" * 58)
    print("  Done. Framework verified successfully.")
    print("━" * 58)
    print()


def _plot_ascii(losses, width=50, height=15):
    """Render a simple ASCII loss curve in the terminal."""
    n = len(losses)
    max_loss = max(losses)
    min_loss = min(losses)
    span = max_loss - min_loss if max_loss != min_loss else 1.0

    # Downsample to width
    step = max(1, n // width)
    sampled = [losses[i] for i in range(0, n, step)][:width]

    for row in range(height):
        threshold = max_loss - (row / (height - 1)) * span
        line = "  │"
        for val in sampled:
            if val >= threshold:
                line += "█"
            else:
                line += " "
        if row == 0:
            line += f"  {max_loss:.4f}"
        elif row == height - 1:
            line += f"  {min_loss:.4f}"
        print(line)

    print("  └" + "─" * len(sampled))
    print(f"   0{' ' * (len(sampled) - 5)}{n} epochs")


if __name__ == "__main__":
    main()
