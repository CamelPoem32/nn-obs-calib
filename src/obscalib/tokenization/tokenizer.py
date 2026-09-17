"""Assembly of prepared measurements, metadata, time, and observability."""

import torch
from torch import nn

from obscalib.data.structures import (
    CanonicalMeasurements,
    ObservabilityResult,
    TokenBatch,
)
from obscalib.tokenization.measurement_encoder import MeasurementEncoder
from obscalib.tokenization.metadata_encoder import MetadataEncoder
from obscalib.tokenization.time_encoder import TimeEncoder


class Tokenizer(nn.Module):
    """Combine independently prepared features into Transformer input tokens."""

    def __init__(
        self,
        measurement_encoder: MeasurementEncoder,
        metadata_encoder: MetadataEncoder,
        time_encoder: TimeEncoder,
        observability_dim: int,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        if observability_dim <= 0:
            raise ValueError("observability_dim must be positive.")

        self.measurement_encoder = measurement_encoder
        self.metadata_encoder = metadata_encoder
        self.time_encoder = time_encoder
        self.observability_dim = observability_dim
        concatenated_dim = (
            measurement_encoder.output_dim
            + metadata_encoder.output_dim
            + time_encoder.output_dim
            + observability_dim
        )
        self.output_dim = concatenated_dim if output_dim is None else output_dim
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        self.output_projection: nn.Module
        if self.output_dim == concatenated_dim:
            self.output_projection = nn.Identity()
        else:
            self.output_projection = nn.Linear(concatenated_dim, self.output_dim)

    def forward(
        self,
        measurements: CanonicalMeasurements,
        observability: ObservabilityResult,
    ) -> TokenBatch:
        """Build tokens without geometry conversion, sorting, or estimation."""

        measurements.validate()
        features = observability.features
        if features is None:
            raise ValueError(
                "observability features must be produced by an ObservabilityMapper."
            )

        expected_prefix = (*measurements.values.shape[:2], self.observability_dim)
        if features.shape != expected_prefix:
            raise ValueError(
                "observability.features must have shape "
                f"{expected_prefix}, got {tuple(features.shape)}."
            )

        # Encode each concept separately so future ablations remain local.
        measurement_features = self.measurement_encoder(measurements.values)
        metadata_features = self.metadata_encoder(
            measurements.sensor_ids, measurements.type_ids
        )
        time_features = self.time_encoder(measurements.timestamps)

        # x: [B, N, d_measurement + d_metadata + d_time + d_obs]
        concatenated = torch.cat(
            (
                measurement_features,
                metadata_features,
                time_features,
                features,
            ),
            dim=-1,
        )
        x = self.output_projection(concatenated)
        return TokenBatch(
            x=x,
            valid_mask=measurements.valid_mask,
            sensor_ids=measurements.sensor_ids,
            type_ids=measurements.type_ids,
        )
