"""Configurable shared multilayer perceptron."""

import torch
from torch import nn

from obscalib.config import MLPConfig
from obscalib.models.activations import make_activation


class MLPBody(nn.Module):
    """Map flattened Transformer summaries to shared calibration features."""

    def __init__(self, config: MLPConfig, *, input_dim: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        self.config = config
        self.input_dim = input_dim
        layers: list[nn.Module] = []
        previous_dim = input_dim

        # Hidden blocks are configurable in both count and width.
        for hidden_dim in config.hidden_dims:
            layers.extend(
                (
                    nn.Linear(previous_dim, hidden_dim),
                    make_activation(config.activation),
                    nn.Dropout(config.dropout),
                )
            )
            previous_dim = hidden_dim

        # The final shared feature layer intentionally has no activation.
        layers.append(nn.Linear(previous_dim, config.output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Transform features shaped [B, input_dim] to [B, output_dim]."""

        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(f"features must have shape [B, {self.input_dim}].")
        return self.layers(features)
