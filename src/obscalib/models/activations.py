"""Configurable Activations."""

from torch import nn


def make_activation(name: str) -> nn.Module:
    """Construct one of the explicitly supported activation modules."""

    activations: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    try:
        activation_type = activations[name.lower()]
    except KeyError as error:
        supported = ", ".join(sorted(activations))
        raise ValueError(f"Unsupported activation {name!r}; choose {supported}.") from error
    return activation_type()