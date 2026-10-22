"""Analytic conditional flow-matching paths for calibration corrections.

The target correction is the existing obscalib left-applied update:
    T_true = Exp(delta_xi) @ T_current,
    delta_xi = Log(T_true @ inverse(T_current)),
    delta_tau = tau_true - tau_current.

The flow *state* is a correction, not an absolute sensor extrinsic. It starts at
(I, 0) in SE(3) x R and ends at (Exp(delta_xi), delta_tau). The six-component
algebra ordering is [rotation_vector, translation_algebra_vector].
"""

from dataclasses import dataclass

import torch

from obscalib.geometry.lie import se3_exp


@dataclass(frozen=True)
class CalibrationFlowBatch:
    """One exactly supervised flow query, with arbitrary leading batch dimensions.

    correction_transform: [..., 4, 4], Exp(t * target_delta_xi).
    correction_xi: [..., 6], t * target_delta_xi (principal log for |phi| < pi).
    correction_tau: [..., 1], t * target_delta_tau.
    time: [..., 1], interpolation time in [0, 1].
    target_velocity: [..., 7], [target_delta_xi, target_delta_tau].

    The six-dimensional target velocity is left-trivialized: a solver advances
    the correction by D_next = D @ Exp(dt * velocity_xi).
    """

    correction_transform: torch.Tensor
    correction_xi: torch.Tensor
    correction_tau: torch.Tensor
    time: torch.Tensor
    target_velocity: torch.Tensor


def sample_flow_times(reference: torch.Tensor, *, generator: torch.Generator | None = None, shared_across_calibrations: bool = False) -> torch.Tensor:
    """Sample uniform times matching a [..., D] reference tensor.

    If reference is [B, C, D], shared_across_calibrations=True draws [B, 1, 1]
    and broadcasts over C so the same physical window uses one interpolation time.
    """

    if reference.ndim < 2 or not reference.is_floating_point() or reference.shape[-1] < 1:
        raise ValueError("reference must be floating with shape [..., D] and at least one batch dimension.")
    batch_shape = reference.shape[:-1]
    sampled_shape = batch_shape[:-1] + (1, 1) if shared_across_calibrations and len(batch_shape) >= 2 else batch_shape + (1,)
    sampled = torch.rand(sampled_shape, device=reference.device, dtype=reference.dtype, generator=generator)
    return sampled.expand(*batch_shape, 1)


@torch.no_grad()
def make_calibration_flow_batch(target_delta_xi: torch.Tensor, target_delta_tau: torch.Tensor, *, time: torch.Tensor | None = None, generator: torch.Generator | None = None, shared_across_calibrations: bool = False) -> CalibrationFlowBatch:
    """Construct the paper's exponential path without integrating a model.

    All inputs share the same leading batch dimensions. The target is kept in
    physical units (rad, m, s). The model/loss can normalize its input and
    velocity independently in later modules.

    Use the same `time` for all views when coupling losses across augmented
    original/rotated/opposite windows is desired.
    """

    if target_delta_xi.ndim < 2 or target_delta_xi.shape[-1] != 6 or not target_delta_xi.is_floating_point():
        raise ValueError("target_delta_xi must be floating with shape [..., 6].")
    if target_delta_tau.shape != target_delta_xi.shape[:-1] + (1,) or not target_delta_tau.is_floating_point():
        raise ValueError("target_delta_tau must have shape [..., 1] with the same leading dimensions as target_delta_xi.")
    if target_delta_xi.device != target_delta_tau.device or target_delta_xi.dtype != target_delta_tau.dtype:
        raise ValueError("target_delta_xi and target_delta_tau must have the same dtype and device.")

    if time is None:
        time = sample_flow_times(target_delta_xi, generator=generator, shared_across_calibrations=shared_across_calibrations)
    else:
        if time.shape != target_delta_tau.shape or time.device != target_delta_xi.device or time.dtype != target_delta_xi.dtype:
            raise ValueError("time must have shape [..., 1], dtype and device matching the targets.")
        if not bool(torch.isfinite(time).all()) or bool(((time < 0) | (time > 1)).any()):
            raise ValueError("time must contain finite values in [0, 1].")

    correction_xi = time * target_delta_xi
    correction_tau = time * target_delta_tau
    correction_transform = se3_exp(correction_xi.reshape(-1, 6)).reshape(*target_delta_xi.shape[:-1], 4, 4)
    target_velocity = torch.cat((target_delta_xi, target_delta_tau), dim=-1)
    return CalibrationFlowBatch(correction_transform=correction_transform, correction_xi=correction_xi, correction_tau=correction_tau, time=time, target_velocity=target_velocity)


__all__ = ["CalibrationFlowBatch", "sample_flow_times", "make_calibration_flow_batch"]
