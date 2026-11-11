"""Euclidean conditional flow-matching paths for 6D SO(3) correction coordinates.

The state is [two SO(3) matrix columns, parent-frame delta translation, delta tau],
with width 10. Only the *final* 6D rotation coordinates are projected onto SO(3).
In particular, intermediate states are NOT guaranteed to encode valid rotations;
there is no SE(3) logarithm, exponential or branch choice in this flow path.

This module is deliberately separate from calibration_path.py's Lie-group paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from obscalib.flow_matching.calibration_path import sample_flow_times


IDENTITY_ROTATION_6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


@dataclass(frozen=True)
class Rotation6DFlowBatch:
    """Analytic flow query, all tensors shaped [...,10] except time [...,1]."""

    state: torch.Tensor
    target_state: torch.Tensor
    target_velocity: torch.Tensor
    time: torch.Tensor
    state_mode: torch.Tensor  # 0=on-path, 1=off-path, 2=identity, shape [...,1]

    @property
    def rotation_6d(self) -> torch.Tensor:
        return self.state[..., :6]

    @property
    def delta_translation(self) -> torch.Tensor:
        return self.state[..., 6:9]

    @property
    def delta_tau(self) -> torch.Tensor:
        return self.state[..., 9:10]


def identity_6d_flow_state(reference: torch.Tensor) -> torch.Tensor:
    """Build the no-correction state [...,10] using reference device/dtype."""
    if reference.ndim < 2 or not reference.is_floating_point():
        raise ValueError('reference must have floating shape [...,D], with batch dimensions')
    result = reference.new_zeros(*reference.shape[:-1], 10)
    result[..., :6] = reference.new_tensor(IDENTITY_ROTATION_6D)
    return result


def stack_rotation_6d_flow_target(rotation_6d: torch.Tensor, delta_translation: torch.Tensor, delta_tau: torch.Tensor) -> torch.Tensor:
    """Concatenate a physical target without log/exp or translation coupling."""
    if rotation_6d.ndim < 2 or rotation_6d.shape[-1] != 6 or not rotation_6d.is_floating_point():
        raise ValueError('rotation_6d must have floating shape [...,6]')
    if delta_translation.shape != rotation_6d.shape[:-1] + (3,) or delta_tau.shape != rotation_6d.shape[:-1] + (1,):
        raise ValueError('delta_translation and delta_tau must have matching [...,3] and [...,1] shapes')
    if any(v.device != rotation_6d.device or v.dtype != rotation_6d.dtype for v in (delta_translation, delta_tau)):
        raise ValueError('All targets must have the same dtype/device')
    result = torch.cat((rotation_6d, delta_translation, delta_tau), dim=-1)
    if not bool(torch.isfinite(result).all()):
        raise ValueError('Target contains nonfinite values')
    return result


@torch.no_grad()
def make_rotation_6d_flow_batch(
    target_state: torch.Tensor,
    *,
    time: torch.Tensor | None = None,
    mode: str = 'mixed',
    off_path_probability: float = 0.4,
    identity_probability: float = 0.15,
    rotation_noise_std: float = 0.10,
    translation_noise_std_m: float = 0.02,
    tau_noise_std_s: float = 0.02,
    max_time: float = 0.95,
    generator: torch.Generator | None = None,
    shared_across_calibrations: bool = False,
) -> Rotation6DFlowBatch:
    """Correction-to-go supervision for the full 10D Euclidean flow.

    On path: x=(1-t)x0+t*x*, target v=(x*-x0).
    Off path: x=x_path+(1-t)*noise, v=(x*-x)/(1-t).
    Identity: t=0, x=x0, target v=(x*-x0).

    Off-path noise is added to six unconstrained embedding coordinates, not to a
    rotation matrix. This is intentional; no intermediate Gram-Schmidt is used.
    Rotation noise is per embedding component, not an angular noise in radians.
    """
    if target_state.ndim < 2 or target_state.shape[-1] != 10 or not target_state.is_floating_point():
        raise ValueError('target_state must have floating shape [...,10]')
    if not bool(torch.isfinite(target_state).all()):
        raise ValueError('target_state must be finite')
    if mode not in ('mixed', 'on_path', 'off_path', 'identity'):
        raise ValueError(f'Unknown state mode: {mode}')
    if not 0.0 < max_time < 1.0:
        raise ValueError('max_time must be in (0,1)')
    if any(not math.isfinite(v) or v < 0 for v in (rotation_noise_std, translation_noise_std_m, tau_noise_std_s)):
        raise ValueError('Noise standard deviations must be finite and nonnegative')
    if any(not 0.0 <= v <= 1.0 for v in (off_path_probability, identity_probability)) or off_path_probability + identity_probability > 1.0:
        raise ValueError('Invalid mixture probabilities')
    if time is None:
        time = sample_flow_times(target_state, generator=generator, shared_across_calibrations=shared_across_calibrations) * max_time
    elif time.shape != target_state.shape[:-1] + (1,) or time.device != target_state.device or time.dtype != target_state.dtype:
        raise ValueError('time shape, device and dtype must match target_state')
    if not bool(torch.isfinite(time).all()) or bool(((time < 0) | (time > max_time + 1e-7)).any()):
        raise ValueError('time must lie within [0,max_time]')

    # For [B,C,10], draw one mode for all calibration keys in the same example.
    draw_shape = time.shape[:-2] + (1, 1) if time.ndim >= 3 else time.shape
    if mode == 'mixed':
        draws = torch.rand(draw_shape, dtype=target_state.dtype, device=target_state.device, generator=generator)
        identity_mask = (draws < identity_probability).expand_as(time)
        off_path_mask = ((draws >= identity_probability) & (draws < identity_probability + off_path_probability)).expand_as(time)
    else:
        identity_mask = torch.full_like(time, mode == 'identity', dtype=torch.bool)
        off_path_mask = torch.full_like(time, mode == 'off_path', dtype=torch.bool)

    actual_time = torch.where(identity_mask, torch.zeros_like(time), time)
    remaining = 1.0 - actual_time
    off_path_mask = off_path_mask & (actual_time > 0)
    identity_state = identity_6d_flow_state(target_state)
    ideal = (1.0 - actual_time) * identity_state + actual_time * target_state
    noise_scale = target_state.new_tensor((rotation_noise_std,) * 6 + (translation_noise_std_m,) * 3 + (tau_noise_std_s,))
    noise = torch.randn(target_state.shape, dtype=target_state.dtype, device=target_state.device, generator=generator) * noise_scale
    state = ideal + remaining * off_path_mask * noise
    velocity = (target_state - state) / remaining
    state_mode = torch.where(identity_mask, 2, torch.where(off_path_mask, 1, 0)).to(torch.int64)
    if not bool(torch.isfinite(velocity).all()):
        raise FloatingPointError('Nonfinite 6D FM velocity target')
    return Rotation6DFlowBatch(state, target_state, velocity, actual_time, state_mode)


__all__ = ['IDENTITY_ROTATION_6D', 'Rotation6DFlowBatch', 'identity_6d_flow_state', 'stack_rotation_6d_flow_target', 'make_rotation_6d_flow_batch']
