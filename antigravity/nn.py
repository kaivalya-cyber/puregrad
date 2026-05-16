"""
Composite neural network architectures.

MLP — Multi-Layer Perceptron with configurable hidden dims and activation.
"""

from antigravity.layers import Module, Linear, ReLU, Tanh


class MLP(Module):
    """
    Multi-Layer Perceptron.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input features.
    hidden_dims : list[int]
        Sizes of hidden layers.
    output_dim : int
        Dimensionality of output.
    activation : str
        Activation function between hidden layers. One of 'relu', 'tanh'.

    Example
    -------
    >>> model = MLP(input_dim=2, hidden_dims=[16, 16], output_dim=1)
    >>> predictions = model(X_train)
    """

    _ACTIVATIONS = {
        "relu": ReLU,
        "tanh": Tanh,
    }

    def __init__(self, input_dim, hidden_dims, output_dim, activation="relu"):
        if activation not in self._ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Choose from: {list(self._ACTIVATIONS.keys())}"
            )

        act_cls = self._ACTIVATIONS[activation]
        self.layers = []

        # Hidden layers
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i + 1]))
            self.layers.append(act_cls())

        # Output layer (no activation — raw logits / values)
        self.layers.append(Linear(dims[-1], output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
