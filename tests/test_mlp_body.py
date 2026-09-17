import pytest
import torch
from torch import nn

from obscalib.config import MLPConfig
from obscalib.models import MLPBody, make_activation


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("relu", nn.ReLU),
        ("gelu", nn.GELU),
        ("silu", nn.SiLU),
        ("tanh", nn.Tanh),
    ],
)
def test_every_supported_activation(name: str, expected_type: type[nn.Module]) -> None:
    assert isinstance(make_activation(name), expected_type)


@pytest.mark.parametrize("hidden_dims", [(12,), (12, 9, 7)])
def test_mlp_hidden_depth_output_shape_and_gradients(
    hidden_dims: tuple[int, ...],
) -> None:
    mlp = MLPBody(
        MLPConfig(
            hidden_dims=hidden_dims,
            output_dim=4,
            activation="silu",
            dropout=0.1,
        ),
        input_dim=10,
    )
    features = torch.randn(3, 10, requires_grad=True)

    output = mlp(features)
    output.sum().backward()

    assert output.shape == (3, 4)
    assert features.grad is not None
    assert all(parameter.grad is not None for parameter in mlp.parameters())


def test_unknown_activation_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported activation"):
        make_activation("mystery")
