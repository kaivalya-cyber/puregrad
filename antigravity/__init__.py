"""
Antigravity — A Custom Deep Learning Framework From Scratch

Zero external ML dependencies. Pure Python + NumPy.
Dynamic computation graph with automatic differentiation.
"""

from antigravity.tensor import Tensor
from antigravity.layers import Module, Linear, ReLU, Tanh
from antigravity.nn import MLP
from antigravity.loss import MSELoss, CrossEntropyLoss
from antigravity.optim import SGD
from antigravity.data import make_moons, make_circles
from antigravity.utils import draw_graph, gradient_check

__version__ = "1.0.0"
__all__ = [
    "Tensor",
    "Module", "Linear", "ReLU", "Tanh",
    "MLP",
    "MSELoss", "CrossEntropyLoss",
    "SGD",
    "make_moons", "make_circles",
    "draw_graph", "gradient_check",
]
