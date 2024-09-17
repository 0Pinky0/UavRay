from torch import nn

_activations = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'silu': nn.SiLU,
}


def get_activation_class(name: str):
    return _activations[name]
