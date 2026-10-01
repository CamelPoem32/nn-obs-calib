"""Precomputed dimensionless soft-projector observability targets for calibration learning."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from obscalib.data.collation import collate_windows
from obscalib.observability.estimators import ObservabilityEstimator
from obscalib.observability.layout import CalibrationParameterLayout
from obscalib.observability.structures import BatchedObservabilityMatrix, WindowObservabilityMatrix
from obscalib.observability.mappings import dimensionless_soft_projector_observability


@dataclass(frozen=True)
class SoftProjectorTargetConfig:
    """Physical normalization and spectral softness used for observability targets."""

    rotation_scale_rad: float = math.radians(5.0)
    translation_scale_m: float = 0.05
    time_offset_scale_s: float = 0.015
    lambda_value_setting: float = 0.25
    target_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        for name, value in (("rotation_scale_rad", self.rotation_scale_rad), ("translation_scale_m", self.translation_scale_m), ("time_offset_scale_s", self.time_offset_scale_s), ("lambda_value_setting", self.lambda_value_setting)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")

        if not self.target_dtype.is_floating_point:
            raise TypeError("target_dtype must be floating-point.")


@dataclass(frozen=True)
class ObservabilityTargetCache:
    """
    RAM-backed observability targets.

    values has shape [N_windows, N_calibrations, 7] with calibration ordering
    defined explicitly by calibration_keys.
    """

    values: torch.Tensor
    calibration_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.values.ndim != 3 or self.values.shape[-1] != 7:
            raise ValueError("values must have shape [N_windows, N_calibrations, 7].")

        if self.values.shape[1] != len(self.calibration_keys):
            raise ValueError("values calibration dimension must match calibration_keys.")

        if len(self.calibration_keys) != len(set(self.calibration_keys)):
            raise ValueError("calibration_keys must be unique.")

        if self.values.device.type != "cpu":
            raise ValueError("ObservabilityTargetCache values must remain on CPU.")

        if not torch.isfinite(self.values).all():
            raise ValueError("ObservabilityTargetCache values must be finite.")

        if torch.any(self.values < 0.0) or torch.any(self.values > 1.0):
            raise ValueError("ObservabilityTargetCache values must lie in [0, 1].")

    def targets_for_indices(self, dataset_indices: torch.Tensor, *, device: torch.device | str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        if dataset_indices.ndim != 1:
            raise ValueError("dataset_indices must have shape [B].")

        indices = dataset_indices.detach().to(device="cpu", dtype=torch.long)

        if torch.any(indices < 0) or torch.any(indices >= self.values.shape[0]):
            raise IndexError("dataset_indices contains an index outside the observability target cache.")

        selected = self.values.index_select(0, indices).to(device=device, dtype=dtype)

        return {calibration_key: selected[:, calibration_index, :] for calibration_index, calibration_key in enumerate(self.calibration_keys)}


def _extract_single_fisher(result) -> tuple[torch.Tensor, CalibrationParameterLayout]:
    raw = result.raw

    if isinstance(raw, WindowObservabilityMatrix):
        return raw.fisher_information_matrix, raw.layout

    if isinstance(raw, BatchedObservabilityMatrix):
        if raw.fisher_information_matrix.shape[0] != 1:
            raise ValueError("Single-window target construction requires a batched observability result with B=1.")

        return raw.fisher_information_matrix[0], raw.layout

    raise TypeError(f"Unsupported observability result type: {type(raw).__name__}.")


def build_observability_target_cache(dataset, estimator: ObservabilityEstimator, *, calibration_keys: tuple[str, ...] | None = None, config: SoftProjectorTargetConfig | None = None) -> ObservabilityTargetCache:
    """Compute one seven-coordinate observability target per dataset window and calibration."""

    config = SoftProjectorTargetConfig() if config is None else config
    calibration_keys = tuple(dataset.calibration_keys if calibration_keys is None else calibration_keys)

    if not calibration_keys:
        raise ValueError("At least one calibration key is required.")

    values = torch.empty((len(dataset), len(calibration_keys), 7), dtype=config.target_dtype, device="cpu")

    for dataset_index in range(len(dataset)):
        sample = dataset[dataset_index]

        if sample.dataset_index is not None and sample.dataset_index != dataset_index:
            raise ValueError(f"Dataset index mismatch: dataset[{dataset_index}] reports dataset_index={sample.dataset_index}.")

        window = collate_windows([sample])

        try:
            result = estimator(window.streams, window.current_calibration, window.metadata)
            fisher, layout = _extract_single_fisher(result)
        except Exception as error:
            sequence_name = dataset.sequence_name_for_window(dataset_index) if hasattr(dataset, "sequence_name_for_window") else "<unknown>"
            raise RuntimeError(f"Observability target computation failed for dataset index {dataset_index}, sequence {sequence_name!r}.") from error

        if tuple(layout.calibration_keys) != calibration_keys:
            raise ValueError(f"Observability layout keys {tuple(layout.calibration_keys)} do not match target cache calibration keys {calibration_keys}.")

        coordinate_observability, _, _, _ = dimensionless_soft_projector_observability(
            fisher,
            layout,
            calibration_scale=(
                config.rotation_scale_rad,
                config.rotation_scale_rad,
                config.rotation_scale_rad,
                config.translation_scale_m,
                config.translation_scale_m,
                config.translation_scale_m,
                config.time_offset_scale_s,
            ),
            lambda_value_setting=config.lambda_value_setting,
        )

        for calibration_index, calibration_key in enumerate(calibration_keys):
            block = layout.block_for(calibration_key)
            values[dataset_index, calibration_index] = coordinate_observability[block.parameter_slice].to(dtype=config.target_dtype)

    return ObservabilityTargetCache(values=values, calibration_keys=calibration_keys)