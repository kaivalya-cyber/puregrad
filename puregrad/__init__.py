"""
Antigravity — A Custom Deep Learning Framework From Scratch

Zero external ML dependencies. Pure Python + NumPy.
Dynamic computation graph with automatic differentiation.
"""

from puregrad.tensor import Tensor
from puregrad.layers import Module, Linear, ReLU, Tanh, Sigmoid, Dropout, LeakyReLU, GELU, BatchNorm1d
from puregrad.nn import MLP, Sequential
from puregrad.loss import MSELoss, CrossEntropyLoss, SoftmaxCrossEntropyLoss
from puregrad.optim import SGD, Adam, StepLR, ExponentialLR, CosineAnnealingLR
from puregrad.data import make_moons, make_circles, DataLoader, train_test_split
from puregrad.utils import draw_graph, gradient_check

__version__ = "1.0.0"
__all__ = [
    "Tensor",
    "Module", "Linear", "ReLU", "Tanh", "Sigmoid", "Dropout", "LeakyReLU", "GELU", "BatchNorm1d",
    "MLP", "Sequential",
    "MSELoss", "CrossEntropyLoss", "SoftmaxCrossEntropyLoss",
    "SGD", "Adam", "StepLR", "ExponentialLR", "CosineAnnealingLR",
    "make_moons", "make_circles", "DataLoader", "train_test_split",
    "draw_graph", "gradient_check",
]
