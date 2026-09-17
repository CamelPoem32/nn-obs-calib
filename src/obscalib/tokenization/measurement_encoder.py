"""Learned projection of canonical measurements."""

import torch
from torch import nn


class MeasurementEncoder(nn.Module):
    """Project canonical measurement vectors to an embedding dimension."""

    def __init__(self, input_dim: int, embedding_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or embedding_dim <= 0:
            raise ValueError("input_dim and embedding_dim must be positive.")
        self.input_dim = input_dim
        self.output_dim = embedding_dim
        self.projection = nn.Linear(input_dim, embedding_dim)

    def forward(self, measurements: torch.Tensor) -> torch.Tensor:
        """Encode measurements shaped [B, N, input_dim]."""

        if measurements.ndim != 3 or measurements.shape[-1] != self.input_dim:
            raise ValueError(
                f"measurements must have shape [B, N, {self.input_dim}]."
            )
        return self.projection(measurements)
