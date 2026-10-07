"""Static ground-truth calibration augmentation for rotation and time offset."""

from __future__ import annotations

import torch

from obscalib.augmentations.config import PerturbationDistribution, TrueCalibrationAugmentationConfig
from obscalib.calibration.state import CalibrationState
from obscalib.data.structures import CalibrationTargetBatch, GeometryType, SensorStreamBatch, WindowBatch
from obscalib.geometry.lie import so3_exp


class TrueCalibrationAugmenter:
    """
    Randomize the static ground-truth calibration before prior perturbation.

    This stage is intentionally different from SensorFrameRandomizer:

    - rotation changes the true sensor orientation in the parent/body frame;
    - translation is preserved;
    - time offset changes the true static temporal calibration;
    - raw measurements/timestamps are rendered consistently with the new truth.

    The intended pipeline order is:

        frame randomization
        -> true calibration augmentation
        -> prior perturbation
        -> calibration event
        -> measurement noise

    Rotation uses

        R_parent_sensor_new = Exp(delta_phi_true) @ R_parent_sensor_old

    while preserving the calibration translation. For raw measurements, the
    equivalent old-sensor to new-sensor coordinate transform is

        A = T_old^-1 @ T_new,

    which is rotation-only when the sensor origin is unchanged.

    Time offset uses

        tau_true_new = tau_true_old + delta_tau_true

    and measured timestamps are shifted by -delta_tau_true so applying the new
    true offset recovers the same underlying physical time.
    """

    def __init__(self, config: TrueCalibrationAugmentationConfig) -> None:
        self.config = config

    def __call__(self, window: WindowBatch, generator: torch.Generator | None = None) -> tuple[WindowBatch, dict[str, CalibrationState], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        _validate_generator(generator)
        _validate_window(window)

        if not self.config.enabled:
            return _copy_window(window), dict(window.current_calibration), {}, {}

        true_calibration = dict(window.current_calibration)
        delta_phi_by_key: dict[str, torch.Tensor] = {}
        delta_tau_by_key: dict[str, torch.Tensor] = {}
        relative_frame_by_key: dict[str, torch.Tensor] = {}

        for calibration_key, state in window.current_calibration.items():
            batch_size = state.transform.shape[0]
            device = state.transform.device
            dtype = state.transform.dtype

            overall_probability = self.config.probability_by_key.get(calibration_key, self.config.probability)
            active = _sample_mask(batch_size, overall_probability, device=device, generator=generator)

            delta_phi = torch.zeros(batch_size, 3, device=device, dtype=dtype)
            delta_tau = torch.zeros(batch_size, 1, device=device, dtype=dtype)
            delta_translation = torch.zeros(batch_size, 3, device=device, dtype=dtype)

            if self.config.rotation is not None and self.config.rotation_probability > 0.0:
                _require_uniform_distribution(self.config.rotation, "rotation")
                rotation_active = active & _sample_mask(batch_size, self.config.rotation_probability, device=device, generator=generator)
                sampled_phi = _sample_rotation_vectors(batch_size, self.config.rotation.maximum, device=device, dtype=dtype, generator=generator)
                delta_phi = torch.where(rotation_active[:, None], sampled_phi, delta_phi)

            if self.config.time_offset is not None and self.config.time_offset_probability > 0.0:
                _require_uniform_distribution(self.config.time_offset, "time_offset")
                time_active = active & _sample_mask(batch_size, self.config.time_offset_probability, device=device, generator=generator)
                sampled_tau = _sample_signed_scalar(batch_size, self.config.time_offset.maximum, device=device, dtype=dtype, generator=generator)
                delta_tau = torch.where(time_active[:, None], sampled_tau, delta_tau)

            if self.config.translation is not None and self.config.translation_probability > 0.0:
                _require_uniform_distribution(self.config.translation, "translation")
                translation_active = active & _sample_mask(batch_size, self.config.translation_probability, device=device, generator=generator)
                sampled_translation = _sample_translation_vectors(batch_size, self.config.translation.maximum, device=device, dtype=dtype, generator=generator)
                delta_translation = torch.where(translation_active[:, None], sampled_translation, delta_translation)
                
            old_transform = state.transform
            new_transform = old_transform.clone()
            new_transform[..., :3, :3] = so3_exp(delta_phi) @ old_transform[..., :3, :3]
            new_transform[..., :3, 3] = old_transform[..., :3, 3] + delta_translation
            new_time_offset = state.time_offset + delta_tau

            true_calibration[calibration_key] = CalibrationState(transform=new_transform, time_offset=new_time_offset)
            delta_phi_by_key[calibration_key] = delta_phi
            delta_tau_by_key[calibration_key] = delta_tau
            relative_frame_by_key[calibration_key] = _relative_frame(old_transform, new_transform)

        augmented_streams = {
            stream_key: _render_static_truth_change(
                stream=stream,
                geometry_type=window.metadata[stream_key].geometry_type,
                relative_frame=relative_frame_by_key[window.metadata[stream_key].calibration_key],
                delta_tau=delta_tau_by_key[window.metadata[stream_key].calibration_key],
            )
            for stream_key, stream in window.streams.items()
        }

        augmented_targets = _update_static_targets(window.targets, true_calibration)

        augmented_window = WindowBatch(
            streams=augmented_streams,
            current_calibration=true_calibration,
            metadata=dict(window.metadata),
            targets=augmented_targets,
            dataset_indices=window.dataset_indices,
        )

        return augmented_window, true_calibration, delta_phi_by_key, delta_tau_by_key


def _render_static_truth_change(stream: SensorStreamBatch, geometry_type: GeometryType, relative_frame: torch.Tensor, delta_tau: torch.Tensor) -> SensorStreamBatch:
    values = _reexpress_measurements(stream.values, geometry_type, relative_frame)
    timestamps = _shift_timestamps(stream.timestamps, stream.sample_mask, delta_tau)
    interval_start_timestamps = None if stream.interval_start_timestamps is None else _shift_timestamps(stream.interval_start_timestamps, stream.sample_mask, delta_tau)
    return SensorStreamBatch(values=values, timestamps=timestamps, sample_mask=stream.sample_mask, interval_start_timestamps=interval_start_timestamps)


def _reexpress_measurements(measurements: torch.Tensor, geometry_type: GeometryType, relative_frame: torch.Tensor) -> torch.Tensor:
    R = relative_frame[..., :3, :3]
    R_inverse = R.transpose(-1, -2)

    if geometry_type == GeometryType.VECTOR:
        if measurements.ndim != 3 or measurements.shape[-1] != 3:
            raise ValueError("VECTOR measurements must have shape [B, N, 3].")
        return measurements @ R

    if geometry_type == GeometryType.SO3:
        if measurements.ndim != 4 or measurements.shape[-2:] != (3, 3):
            raise ValueError("SO3 measurements must have shape [B, N, 3, 3].")
        return R_inverse[:, None] @ measurements @ R[:, None]

    if geometry_type == GeometryType.SE3:
        if measurements.ndim != 4 or measurements.shape[-2:] != (4, 4):
            raise ValueError("SE3 measurements must have shape [B, N, 4, 4].")
        A = relative_frame[:, None]
        A_inverse = _inverse_se3(relative_frame)[:, None]
        return A_inverse @ measurements @ A

    raise ValueError(f"Unsupported geometry type: {geometry_type!r}.")


def _rotation_only_relative_frame(old_transform: torch.Tensor, new_transform: torch.Tensor) -> torch.Tensor:
    R_old = old_transform[..., :3, :3]
    R_new = new_transform[..., :3, :3]
    relative = torch.zeros_like(old_transform)
    relative[..., :3, :3] = R_old.transpose(-1, -2) @ R_new
    relative[..., 3, 3] = 1.0
    return relative


def _shift_timestamps(timestamps: torch.Tensor, sample_mask: torch.Tensor, delta_tau: torch.Tensor) -> torch.Tensor:
    shifted = timestamps - delta_tau
    return torch.where(sample_mask, shifted, timestamps)


def _update_static_targets(targets: dict[str, CalibrationTargetBatch] | None, true_calibration: dict[str, CalibrationState]) -> dict[str, CalibrationTargetBatch] | None:
    if targets is None:
        return None

    updated_targets: dict[str, CalibrationTargetBatch] = {}

    for calibration_key, target in targets.items():
        if calibration_key not in true_calibration:
            updated_targets[calibration_key] = target
            continue

        state = true_calibration[calibration_key]
        updated_targets[calibration_key] = CalibrationTargetBatch(
            next_transform=state.transform if target.next_transform is not None else None,
            next_time_offset=state.time_offset if target.next_time_offset is not None else None,
            change_label=target.change_label,
            change_time=target.change_time,
        )

    return updated_targets



def _require_uniform_distribution(config, name: str) -> None:
    if config.distribution != PerturbationDistribution.UNIFORM:
        raise NotImplementedError(f"True-calibration augmentation currently supports only UNIFORM {name} sampling.")


def _sample_rotation_vectors(batch_size: int, maximum: float, *, device: torch.device, dtype: torch.dtype, generator: torch.Generator | None) -> torch.Tensor:
    axis = torch.randn(batch_size, 3, device=device, dtype=dtype, generator=generator)
    axis_norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    fallback = torch.zeros_like(axis)
    fallback[:, 0] = 1.0
    axis = torch.where(axis_norm > 1e-12, axis / axis_norm.clamp_min(1e-12), fallback)
    angle = torch.rand(batch_size, 1, device=device, dtype=dtype, generator=generator) * maximum
    return axis * angle


def _sample_signed_scalar(batch_size: int, maximum: float, *, device: torch.device, dtype: torch.dtype, generator: torch.Generator | None) -> torch.Tensor:
    return (2.0 * torch.rand(batch_size, 1, device=device, dtype=dtype, generator=generator) - 1.0) * maximum


def _sample_mask(batch_size: int, probability: float, *, device: torch.device, generator: torch.Generator | None) -> torch.Tensor:
    if probability <= 0.0:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)
    if probability >= 1.0:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    return torch.rand(batch_size, device=device, generator=generator) < probability


def _inverse_rotation_only_se3(transform: torch.Tensor) -> torch.Tensor:
    inverse = torch.zeros_like(transform)
    inverse[..., :3, :3] = transform[..., :3, :3].transpose(-1, -2)
    inverse[..., 3, 3] = 1.0
    return inverse


def _copy_window(window: WindowBatch) -> WindowBatch:
    return WindowBatch(streams=dict(window.streams), current_calibration=dict(window.current_calibration), metadata=dict(window.metadata), targets=None if window.targets is None else dict(window.targets), dataset_indices=window.dataset_indices)


def _validate_window(window: WindowBatch) -> None:
    if not window.streams:
        raise ValueError("True-calibration augmentation requires at least one sensor stream.")
    if set(window.streams) != set(window.metadata):
        raise ValueError("Window streams and metadata must have identical keys.")
    if not window.current_calibration:
        raise ValueError("True-calibration augmentation requires at least one calibration state.")

    required_calibration_keys = {metadata.calibration_key for metadata in window.metadata.values()}
    missing_calibration_keys = required_calibration_keys - set(window.current_calibration)

    if missing_calibration_keys:
        raise KeyError(f"Missing calibration states for sensor streams: {sorted(missing_calibration_keys)}")

    for state in window.current_calibration.values():
        state.validate()

    for stream_key, stream in window.streams.items():
        stream.validate()
        calibration_key = window.metadata[stream_key].calibration_key
        if stream.values.shape[0] != window.current_calibration[calibration_key].transform.shape[0]:
            raise ValueError(f"Stream {stream_key!r} and calibration {calibration_key!r} must share batch size.")


def _validate_generator(generator: torch.Generator | None) -> None:
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or None.")

def _sample_translation_vectors(batch_size: int, maximum: float, *, device: torch.device, dtype: torch.dtype, generator: torch.Generator | None) -> torch.Tensor:
    direction = torch.randn(batch_size, 3, device=device, dtype=dtype, generator=generator)
    direction_norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)

    fallback = torch.zeros_like(direction)
    fallback[:, 0] = 1.0

    direction = torch.where(direction_norm > 1e-12, direction / direction_norm.clamp_min(1e-12), fallback)
    magnitude = torch.rand(batch_size, 1, device=device, dtype=dtype, generator=generator) * maximum

    return direction * magnitude

def _relative_frame(old_transform: torch.Tensor, new_transform: torch.Tensor) -> torch.Tensor:
    return _inverse_se3(old_transform) @ new_transform


def _inverse_se3(transform: torch.Tensor) -> torch.Tensor:
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]

    rotation_inverse = rotation.transpose(-1, -2)
    translation_inverse = -(rotation_inverse @ translation.unsqueeze(-1)).squeeze(-1)

    inverse = torch.zeros_like(transform)
    inverse[..., :3, :3] = rotation_inverse
    inverse[..., :3, 3] = translation_inverse
    inverse[..., 3, 3] = 1.0

    return inverse