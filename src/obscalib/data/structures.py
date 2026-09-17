"""Explicit tensor contracts exchanged by calibration pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from obscalib.calibration.state import CalibrationState


def _validate_sequence_fields(
    values: torch.Tensor,
    timestamps: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    if values.ndim != 3:
        raise ValueError("values must have shape [B, N, D].")
    if timestamps.ndim != 2 or valid_mask.ndim != 2:
        raise ValueError("timestamps and valid_mask must have shape [B, N].")
    if values.shape[:2] != timestamps.shape or timestamps.shape != valid_mask.shape:
        raise ValueError("values, timestamps, and valid_mask must share B and N.")
    if valid_mask.dtype is not torch.bool:
        raise TypeError("valid_mask must have boolean dtype.")


@dataclass
class SensorStreamBatch:
    """One raw sensor stream with shapes [B, N_s, d_s] and [B, N_s]."""

    values: torch.Tensor
    timestamps: torch.Tensor
    valid_mask: torch.Tensor

    def validate(self) -> None:
        """Validate only the shape and mask invariants shared by all sensors."""

        _validate_sequence_fields(self.values, self.timestamps, self.valid_mask)


@dataclass(frozen=True)
class SensorMetadata:
    """Minimal stable identity and geometry metadata for one sensor."""

    sensor_id: int
    sensor_type: int
    geometry_type: str


@dataclass
class CalibrationTarget:
    """Optional future supervision fields without fixed loss semantics."""

    target_transform: torch.Tensor | None = None
    target_time_offset: torch.Tensor | None = None
    change_label: torch.Tensor | None = None
    change_time: torch.Tensor | None = None


@dataclass
class WindowBatch:
    """Raw streams, carried calibration, metadata, and optional labels."""

    streams: dict[str, SensorStreamBatch]
    calibration: dict[str, CalibrationState]
    metadata: dict[str, SensorMetadata]
    labels: dict[str, CalibrationTarget] | None = None


@dataclass
class CanonicalMeasurements:
    """Merged canonical sequence using [B, N, d_g] batch-first values."""

    values: torch.Tensor
    timestamps: torch.Tensor
    sensor_ids: torch.Tensor
    type_ids: torch.Tensor
    valid_mask: torch.Tensor

    def validate(self) -> None:
        """Validate sequence fields and aligned integer metadata shapes."""

        _validate_sequence_fields(self.values, self.timestamps, self.valid_mask)
        if self.sensor_ids.shape != self.timestamps.shape:
            raise ValueError("sensor_ids must have shape [B, N].")
        if self.type_ids.shape != self.timestamps.shape:
            raise ValueError("type_ids must have shape [B, N].")


@dataclass
class ObservabilityResult:
    """Scientific observability output and optional NN-ready features.

    raw deliberately has no tensor-shape contract: matrix-, rank-, and
    CRLB-based estimators may preserve different scientific structures.
    features, when populated by an observability mapper, follows
    [B, N, d_obs] for tokenization.
    """

    raw: object | None = None
    features: torch.Tensor | None = None


@dataclass
class TokenBatch:
    """Prepared sequence tokens with x shaped [B, N, d_x]."""

    x: torch.Tensor
    valid_mask: torch.Tensor
    sensor_ids: torch.Tensor
    type_ids: torch.Tensor

    def validate(self) -> None:
        """Validate aligned token and metadata sequence dimensions."""

        if self.x.ndim != 3:
            raise ValueError("x must have shape [B, N, d_x].")
        expected = self.x.shape[:2]
        for name in ("valid_mask", "sensor_ids", "type_ids"):
            if getattr(self, name).shape != expected:
                raise ValueError(f"{name} must have shape [B, N].")
        if self.valid_mask.dtype is not torch.bool:
            raise TypeError("valid_mask must have boolean dtype.")