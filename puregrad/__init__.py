"""
Antigravity — A Custom Deep Learning Framework From Scratch

Zero external ML dependencies. Pure Python + NumPy.
Dynamic computation graph with automatic differentiation.
"""

from puregrad.tensor import Tensor
from puregrad.layers import Module, Linear, ReLU, Tanh
from puregrad.nn import MLP
from puregrad.loss import MSELoss, CrossEntropyLoss
from puregrad.optim import SGD
from puregrad.data import make_moons, make_circles
from puregrad.utils import draw_graph, gradient_check

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
