"""Separate sensor-identity and sensor-type embeddings."""

import torch
from torch import nn


class MetadataEncoder(nn.Module):
    """Embed sensor IDs and sensor types, then concatenate their features."""

    def __init__(
        self,
        num_sensor_ids: int,
        num_sensor_types: int,
        sensor_embedding_dim: int,
        type_embedding_dim: int,
    ) -> None:
        super().__init__()
        for name, value in (
            ("num_sensor_ids", num_sensor_ids),
            ("num_sensor_types", num_sensor_types),
            ("sensor_embedding_dim", sensor_embedding_dim),
            ("type_embedding_dim", type_embedding_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive.")

        self.output_dim = sensor_embedding_dim + type_embedding_dim
        self.sensor_embedding = nn.Embedding(num_sensor_ids, sensor_embedding_dim)
        self.type_embedding = nn.Embedding(num_sensor_types, type_embedding_dim)

    def forward(
        self,
        sensor_ids: torch.Tensor,
        type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return concatenated metadata embeddings shaped [B, N, d_meta]."""

        if sensor_ids.ndim != 2 or type_ids.shape != sensor_ids.shape:
            raise ValueError("sensor_ids and type_ids must share shape [B, N].")
        sensor_features = self.sensor_embedding(sensor_ids)
        type_features = self.type_embedding(type_ids)
        return torch.cat((sensor_features, type_features), dim=-1)
