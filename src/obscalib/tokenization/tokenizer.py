"""Assembly and chronological sorting of Transformer measurement tokens."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from obscalib.data.sorting import sort_measurement_sequence
from obscalib.data.structures import CanonicalSensorStreamBatch, MeasurementSequenceBatch, SensorMetadata, TokenBatch
from obscalib.observability.structures import ObservabilityResult


class Tokenizer(nn.Module):
    """
    Assemble canonical sensor streams into final Transformer tokens.

    Processing order:

        canonical measurement vector
            -> zero-pad to measurement_dim
            -> concatenate corrected relative timestamp
            -> concatenate raw MeasurementType integer
            -> concatenate optional window-level observability
            -> concatenate all sensor streams
            -> chronological sort.
    """

    def __init__(self, measurement_dim: int = 6) -> None:
        super().__init__()

        if measurement_dim <= 0:
            raise ValueError("measurement_dim must be positive.")

        self.measurement_dim = measurement_dim

    def forward(self, streams: Mapping[str, CanonicalSensorStreamBatch], observability: ObservabilityResult | None = None, *, metadata: Mapping[str, SensorMetadata] | None = None) -> TokenBatch:
        """Build and chronologically sort the complete multi-sensor token sequence."""

        if not streams:
            raise ValueError("At least one canonical sensor stream is required.")

        if metadata is not None and set(streams) != set(metadata):
            raise ValueError("streams and metadata must contain identical stream keys.")

        stream_list = list(streams.values())

        for stream in stream_list:
            stream.validate()

        batch_size = stream_list[0].values.shape[0]
        dtype = stream_list[0].values.dtype
        device = stream_list[0].values.device

        calibration_observability_features = None

        if observability is not None:
            if observability.calibration_features is None:
                raise ValueError("Tokenizer requires observability.calibration_features when observability is enabled.")

            if metadata is None:
                raise ValueError("Tokenizer requires sensor metadata when calibration-local observability is enabled.")

            required_calibration_keys = {stream_metadata.calibration_key for stream_metadata in metadata.values()}
            missing_calibration_keys = sorted(required_calibration_keys - set(observability.calibration_features))

            if missing_calibration_keys:
                raise KeyError(f"Observability features are missing calibration keys required by sensor streams: {missing_calibration_keys}.")

            calibration_observability_features = {}
            observability_dimension = None

            for calibration_key in required_calibration_keys:
                calibration_feature = observability.calibration_features[calibration_key]

                if calibration_feature.ndim != 2 or calibration_feature.shape[0] != batch_size:
                    raise ValueError(f"Observability features for calibration {calibration_key!r} must have shape [B, d_calibration_observability].")

                if not torch.isfinite(calibration_feature).all():
                    raise ValueError(f"Observability features for calibration {calibration_key!r} must be finite.")

                if observability_dimension is None:
                    observability_dimension = calibration_feature.shape[-1]
                elif calibration_feature.shape[-1] != observability_dimension:
                    raise ValueError("All calibration-local observability feature blocks must have the same dimension.")

                calibration_observability_features[calibration_key] = calibration_feature.to(device=device, dtype=dtype)

        stream_token_vectors: list[torch.Tensor] = []
        stream_timestamps: list[torch.Tensor] = []
        stream_masks: list[torch.Tensor] = []

        for stream_name, stream in streams.items():
            if stream.values.shape[0] != batch_size:
                raise ValueError(f"Stream {stream_name!r} does not share the common batch size.")

            if stream.values.dtype != dtype or stream.values.device != device:
                raise ValueError(f"Stream {stream_name!r} does not share the common dtype and device.")

            native_dim = stream.values.shape[-1]

            if native_dim > self.measurement_dim:
                raise ValueError(f"Stream {stream_name!r} has canonical dimension {native_dim}, larger than measurement_dim={self.measurement_dim}.")

            padded_measurements = F.pad(stream.values, (0, self.measurement_dim - native_dim))
            num_measurements = stream.values.shape[1]
            corrected_timestamps = stream.timestamps.to(dtype=dtype).unsqueeze(-1)
            measurement_types = torch.full((batch_size, num_measurements, 1), float(int(stream.measurement_type)), dtype=dtype, device=device)
            token_parts = [padded_measurements, corrected_timestamps, measurement_types]

            if calibration_observability_features is not None:
                calibration_key = metadata[stream_name].calibration_key
                observability_per_measurement = calibration_observability_features[calibration_key][:, None, :].expand(-1, num_measurements, -1)
                token_parts.append(observability_per_measurement)

            stream_token_vectors.append(torch.cat(token_parts, dim=-1))
            stream_timestamps.append(stream.timestamps)
            stream_masks.append(stream.sample_mask)

        unsorted_sequence = MeasurementSequenceBatch(x=torch.cat(stream_token_vectors, dim=1), timestamps=torch.cat(stream_timestamps, dim=1), token_mask=torch.cat(stream_masks, dim=1))
        sorted_sequence = sort_measurement_sequence(unsorted_sequence)

        return TokenBatch(x=sorted_sequence.x, token_mask=sorted_sequence.token_mask)