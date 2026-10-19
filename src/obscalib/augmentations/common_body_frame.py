"""Haar-uniform randomization of the common calibration/body coordinate frame.

This is a *passive change of coordinates*, not a physical perturbation of the
individual sensors. For every sensor-to-body transform T_BS, apply the SAME
rotation-only homogeneous transform H_G on the left:

    T_B'S = H_G @ T_BS,           H_G = diag(G, 1),   G in SO(3).

Since the native sensor coordinates are not changed, gyro/accel vectors, LiDAR
relative poses, timestamps, and masks must remain byte-for-byte unchanged.
Only calibration transforms and transform-valued calibration targets change.
Time offsets, calibration events, and calibration metadata do not change.

Use one consistent G for an entire recurrent rollout if a calibration estimate
is carried between windows. Redrawing G in the middle of such a rollout requires
explicitly transforming the carried state as well.
"""

from __future__ import annotations

import torch

from obscalib.geometry.lie import so3_exp
from obscalib.augmentations.config import CommonBodyFrameRandomizationConfig
from obscalib.calibration.state import CalibrationState
from obscalib.data.structures import CalibrationTargetBatch, WindowBatch


class CommonBodyFrameRandomizer:
    """Apply one shared SO(3) change of common frame to all calibration keys.

    Returns (augmented_window, transformed_calibration, H_G), where H_G has
    shape [B, 4, 4] and is identical for all sensors in each batch element.
    For samples not selected by ``probability``, H_G is exactly the identity.
    With three enabled axes and no angle bound, sample Haar-uniform SO(3).
    Otherwise draw one rotation whose axis lies in the enabled axis subspace.
    With one selected axis (e.g., Z), its signed angle is uniform in
    [-maximum, +maximum] or [-pi, +pi] if the bound is omitted.
    """

    def __init__(self, config: CommonBodyFrameRandomizationConfig) -> None:
        self.config = config

    def __call__(self, window: WindowBatch, generator: torch.Generator | None = None) -> tuple[WindowBatch, dict[str, CalibrationState], torch.Tensor]:
        _validate_window(window)
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator or None.")

        reference = next(iter(window.current_calibration.values())).transform
        batch_size, device, dtype = reference.shape[0], reference.device, reference.dtype
        common_frame = torch.eye(4, device=device, dtype=dtype).expand(batch_size, 4, 4).clone()

        if self.config.enabled and self.config.probability > 0.0:
            from obscalib.augmentations.sampling import sample_uniform_so3

            if self.config.rotation is None and all(self.config.rotation_axes):
                # Preserve the original unrestricted Haar sampler exactly.
                sampled_rotation = sample_uniform_so3(batch_size=batch_size, device=device, dtype=dtype, generator=generator)
            else:
                # Restrict the axis subspace and optionally the rotation angle.
                # Z-only samples signed yaw: [-max_angle,+max_angle].
                maximum = torch.pi if self.config.rotation is None else self.config.rotation.maximum
                sampled_rotation = _sample_bounded_rotation(batch_size, maximum, self.config.rotation_axes, device=device, dtype=dtype, generator=generator)
            if self.config.probability == 1.0:
                selected = torch.ones(batch_size, device=device, dtype=torch.bool)
            else:
                selected = torch.rand(batch_size, device=device, generator=generator) < self.config.probability
            common_frame[selected, :3, :3] = sampled_rotation[selected]

        new_calibration = {
            key: CalibrationState(transform=common_frame @ state.transform, time_offset=state.time_offset)
            for key, state in window.current_calibration.items()
        }

        new_targets = None if window.targets is None else {
            key: _transform_target(target, common_frame) if key in new_calibration else target
            for key, target in window.targets.items()
        }

        augmented_window = WindowBatch(
            streams=dict(window.streams),  # Intentionally unchanged: native sensor frames are unchanged.
            current_calibration=new_calibration,
            metadata=dict(window.metadata),
            targets=new_targets,
            dataset_indices=window.dataset_indices,
        )
        return augmented_window, new_calibration, common_frame



def _sample_bounded_rotation(batch_size: int, maximum: float, rotation_axes: tuple[bool, bool, bool], *, device: torch.device, dtype: torch.dtype, generator: torch.Generator | None) -> torch.Tensor:
    """Sample one SO(3) axis in the allowed subspace and a uniform bounded magnitude."""
    axis_mask = torch.tensor(rotation_axes, device=device, dtype=dtype)
    axis = torch.randn(batch_size, 3, device=device, dtype=dtype, generator=generator) * axis_mask
    norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    fallback = torch.zeros_like(axis)
    fallback[:, rotation_axes.index(True)] = 1.0
    axis = torch.where(norm > 1e-12, axis / norm.clamp_min(1e-12), fallback)
    angle = torch.rand(batch_size, 1, device=device, dtype=dtype, generator=generator) * maximum
    return so3_exp(axis * angle)

def _transform_target(target: CalibrationTargetBatch, common_frame: torch.Tensor) -> CalibrationTargetBatch:
    """Transform body-frame target poses; retain time and event supervision."""
    return CalibrationTargetBatch(
        next_transform=None if target.next_transform is None else common_frame @ target.next_transform,
        next_time_offset=target.next_time_offset,
        change_label=target.change_label,
        change_time=target.change_time,
    )


def _validate_window(window: WindowBatch) -> None:
    if not window.current_calibration:
        raise ValueError("Common body frame randomization requires calibration states.")
    if set(window.streams) != set(window.metadata):
        raise ValueError("Window streams and metadata must have identical keys.")

    used_calibration_keys = {metadata.calibration_key for metadata in window.metadata.values()}
    missing = used_calibration_keys - set(window.current_calibration)
    if missing:
        raise KeyError(f"Missing calibration states for streams: {sorted(missing)}")

    states = list(window.current_calibration.values())
    first = states[0].transform
    if first.ndim != 3 or first.shape[-2:] != (4, 4):
        raise ValueError("Calibration transforms must have shape [B, 4, 4].")
    for state in states:
        state.validate()
        if state.transform.shape != first.shape or state.transform.device != first.device or state.transform.dtype != first.dtype:
            raise ValueError("Every calibration key must share batch size, transform dtype, and device.")
    if window.targets is not None:
        for key, target in window.targets.items():
            if key in window.current_calibration and target.next_transform is not None and target.next_transform.shape != first.shape:
                raise ValueError(f"Calibration target {key!r} next_transform must have shape {tuple(first.shape)}.")
