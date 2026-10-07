'''Assembly and chronological sorting of Transformer measurement tokens.'''

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from obscalib.data.sorting import sort_measurement_sequence
from obscalib.data.structures import CanonicalSensorStreamBatch, MeasurementSequenceBatch, SensorMetadata, TokenBatch
from obscalib.observability.structures import ObservabilityResult


class Tokenizer(nn.Module):
    """
    Assemble canonical sensor streams into final Transformer tokens.

    calibration_keys=None preserves the legacy token contract. Providing an
    ordered sequence additionally generates calibration_ids without inserting
    those IDs into the numerical token feature vector.
    """

    def __init__(self, measurement_dim: int = 6, calibration_keys: Sequence[str] | None = None) -> None:
        super().__init__()

        if measurement_dim <= 0:
            raise ValueError("measurement_dim must be positive.")

        if calibration_keys is not None:
            calibration_keys = tuple(calibration_keys)

            if not calibration_keys:
                raise ValueError("calibration_keys must not be empty when provided.")

            if any(not isinstance(key, str) or not key for key in calibration_keys):
                raise ValueError("calibration_keys must contain non-empty strings.")

            if len(calibration_keys) != len(set(calibration_keys)):
                raise ValueError("calibration_keys must be unique.")

        self.measurement_dim = measurement_dim
        self.calibration_keys = None if calibration_keys is None else tuple(calibration_keys)

    def forward(self, streams: Mapping[str, CanonicalSensorStreamBatch], observability: ObservabilityResult | None = None,
                calibration_keys: Sequence[str] | None = None,) -> TokenBatch:
        """Build and chronologically sort the complete multi-sensor token sequence."""

        if not streams:
            raise ValueError("At least one canonical sensor stream is required.")

        stream_list = list(streams.values())

        for stream in stream_list:
            stream.validate()

        batch_size = stream_list[0].values.shape[0]
        dtype = stream_list[0].values.dtype
        device = stream_list[0].values.device
        observability_features = None if observability is None else observability.features
        resolved_calibration_keys = tuple(calibration_keys) if calibration_keys is not None else self.calibration_keys

        if observability_features is not None:
            if observability_features.ndim != 2 or observability_features.shape[0] != batch_size:
                raise ValueError("observability.features must have shape [B, d_observability].")

            observability_features = observability_features.to(device=device, dtype=dtype)

        stream_token_vectors: list[torch.Tensor] = []
        stream_timestamps: list[torch.Tensor] = []
        stream_masks: list[torch.Tensor] = []
        stream_calibration_ids: list[torch.Tensor] = []

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

            if observability_features is not None:
                token_parts.append(observability_features[:, None, :].expand(-1, num_measurements, -1))

            stream_token_vectors.append(torch.cat(token_parts, dim=-1))
            stream_timestamps.append(stream.timestamps)
            stream_masks.append(stream.sample_mask)

            if resolved_calibration_keys is not None:
                if stream.calibration_key is None:
                    raise ValueError(f"Stream {stream_name!r} has no calibration_key but calibration-aware tokenization is enabled.")

                if stream.calibration_key not in resolved_calibration_keys:
                    raise KeyError(f"Unknown calibration key {stream.calibration_key!r} for stream {stream_name!r}. Expected one of {resolved_calibration_keys}.")

                calibration_id = resolved_calibration_keys.index(stream.calibration_key)
                stream_calibration_ids.append(torch.full((batch_size, num_measurements), calibration_id, dtype=torch.long, device=device))

        calibration_ids = None if resolved_calibration_keys is None else torch.cat(stream_calibration_ids, dim=1)
        unsorted_sequence = MeasurementSequenceBatch(
            x=torch.cat(stream_token_vectors, dim=1),
            timestamps=torch.cat(stream_timestamps, dim=1),
            token_mask=torch.cat(stream_masks, dim=1),
            calibration_ids=calibration_ids,
            calibration_keys=() if resolved_calibration_keys is None else resolved_calibration_keys,
        )
        sorted_sequence = sort_measurement_sequence(unsorted_sequence)

        return TokenBatch(
            x=sorted_sequence.x,
            token_mask=sorted_sequence.token_mask,
            calibration_ids=sorted_sequence.calibration_ids,
            calibration_keys=sorted_sequence.calibration_keys,
        )