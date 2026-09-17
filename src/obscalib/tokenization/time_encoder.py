"""Simple explicit encoding for scalar timestamps."""

import torch
from torch import nn


class TimeEncoder(nn.Module):
    """Linearly project caller-normalized scalar timestamps.

    The caller remains responsible for choosing a window-relative time origin.
    time_scale only performs a documented fixed rescaling before projection.
    """

    def __init__(self, embedding_dim: int, time_scale: float = 1.0) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if time_scale <= 0.0:
            raise ValueError("time_scale must be positive.")
        self.output_dim = embedding_dim
        self.time_scale = time_scale
        self.projection = nn.Linear(1, embedding_dim)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Encode timestamps shaped [B, N] as [B, N, embedding_dim]."""

        if timestamps.ndim != 2:
            raise ValueError("timestamps must have shape [B, N].")
        scaled_timestamps = timestamps.unsqueeze(-1) / self.time_scale
        return self.projection(scaled_timestamps)
