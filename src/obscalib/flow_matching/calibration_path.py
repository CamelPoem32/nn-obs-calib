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

from obscalib.geometry.lie import se3_exp, se3_log


@dataclass(frozen=True)
class CalibrationFlowBatch:
    """One exactly supervised flow query, with arbitrary leading batch dimensions.

    correction_transform: [..., 4, 4], intermediate correction transform.
    correction_xi: [..., 6], Log(correction_transform).
    correction_tau: [..., 1], intermediate clock correction.
    time: [..., 1], flow time.
    target_velocity: [..., 7], right-composed velocity supervised at this state.

    The six-dimensional target velocity uses the right-composed convention: a solver advances
    the correction by D_next = D @ Exp(dt * velocity_xi).
    """

    correction_transform: torch.Tensor
    correction_xi: torch.Tensor
    correction_tau: torch.Tensor
    time: torch.Tensor
    target_velocity: torch.Tensor
    state_mode: torch.Tensor | None = None  # 0=on-path, 1=off-path, 2=identity, optional


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




@torch.no_grad()
def make_correction_to_go_flow_batch(
    target_delta_xi: torch.Tensor,
    target_delta_tau: torch.Tensor,
    *,
    time: torch.Tensor,
    mode: str = "mixed",
    off_path_probability: float = 0.5,
    identity_probability: float = 0.1,
    rotation_noise_std_rad: float = 0.08726646259971647,
    translation_noise_std_m: float = 0.02,
    tau_noise_std_s: float = 0.02,
    generator: torch.Generator | None = None,
    max_time: float = 0.9,
) -> CalibrationFlowBatch:
    """Create correction-to-go supervision on SE(3) x R.

    The integrator advances D_next = D @ Exp(dt * v_xi), hence the exact
    right-composed velocity for recovering D_star from D at time t is
        v_xi = Log(D^{-1} D_star) / (1-t),
        v_tau = (delta_tau_star - delta_tau) / (1-t).

    On-path: D = Exp(t * delta_xi_star). Off-path: D is right-perturbed by
    Exp((1-t) * epsilon), with similarly scaled clock noise, so the target
    remains well conditioned close to the end of the interval. Identity mode
    supervises the actual inference start D=I, tau=0 at t=0.

    For mode='mixed', state modes are drawn independently per physical example
    but shared across that example's calibration keys. The returned state_mode
    has values 0=on-path, 1=off-path, 2=identity.

    `time` is explicitly supplied and must match the targets' leading shape
    [...,1]. It must be <= max_time (<1) to bound the remaining-time inverse.
    Existing analytic make_calibration_flow_batch remains unchanged.
    """
    if mode not in {"mixed", "on_path", "off_path", "identity"}:
        raise ValueError(f"Unknown flow state mode: {mode!r}")
    if target_delta_xi.ndim < 2 or target_delta_xi.shape[-1] != 6 or not target_delta_xi.is_floating_point():
        raise ValueError("target_delta_xi must have floating shape [...,6].")
    if target_delta_tau.shape != target_delta_xi.shape[:-1] + (1,) or time.shape != target_delta_tau.shape:
        raise ValueError("target_delta_tau and time must both have shape [...,1] matching target_delta_xi.")
    if any(value.device != target_delta_xi.device or value.dtype != target_delta_xi.dtype for value in (target_delta_tau, time)):
        raise ValueError("Target and time tensors must share device and dtype.")
    if not (0.0 < max_time < 1.0):
        raise ValueError("max_time must be strictly between zero and one.")
    if not bool(torch.isfinite(time).all()) or bool(((time < 0.0) | (time > max_time + 1e-7)).any()):
        raise ValueError("Flow time must be finite and in [0, max_time].")
    if not (0.0 <= off_path_probability <= 1.0 and 0.0 <= identity_probability <= 1.0 and off_path_probability + identity_probability <= 1.0):
        raise ValueError("Mode probabilities must be nonnegative and sum to at most one.")
    if any(not (0.0 <= value < float("inf")) for value in (rotation_noise_std_rad, translation_noise_std_m, tau_noise_std_s)):
        raise ValueError("Noise standard deviations must be finite and nonnegative.")

    # Mode sampling: all calibration keys of the same physical example share
    # the same state mode. This also works with an optional leading time grid.
    draw_shape = time.shape[:-2] + (1, 1) if time.ndim >= 3 else time.shape
    if mode == "mixed":
        draws = torch.rand(draw_shape, device=time.device, dtype=time.dtype, generator=generator)
        identity_mask = (draws < identity_probability).expand_as(time)
        off_path_mask = ((draws >= identity_probability) & (draws < identity_probability + off_path_probability)).expand_as(time)
    else:
        identity_mask = torch.full_like(time, mode == "identity", dtype=torch.bool)
        off_path_mask = torch.full_like(time, mode == "off_path", dtype=torch.bool)

    actual_time = torch.where(identity_mask, torch.zeros_like(time), time)
    remaining = 1.0 - actual_time
    # Exactly identity at t=0, reflecting the inference initial condition.
    off_path_mask = off_path_mask & (actual_time > 0.0)
    state_mode = torch.where(identity_mask, 2, torch.where(off_path_mask, 1, 0)).to(torch.int64)

    ideal_xi = actual_time * target_delta_xi
    ideal_tau = actual_time * target_delta_tau
    batch_shape = target_delta_xi.shape[:-1]
    ideal_transform = se3_exp(ideal_xi.reshape(-1, 6)).reshape(*batch_shape, 4, 4)
    target_transform = se3_exp(target_delta_xi.reshape(-1, 6)).reshape(*batch_shape, 4, 4)

    # Right perturbations correspond to the same multiplication convention as
    # obscalib.flow_matching.lie_integrator.integrate_calibration_flow.
    epsilon = torch.randn(target_delta_xi.shape, device=time.device, dtype=time.dtype, generator=generator)
    scales = epsilon.new_tensor([rotation_noise_std_rad] * 3 + [translation_noise_std_m] * 3)
    perturbation_xi = epsilon * scales * remaining * off_path_mask
    perturbation_transform = se3_exp(perturbation_xi.reshape(-1, 6)).reshape(*batch_shape, 4, 4)
    correction_transform = ideal_transform @ perturbation_transform
    epsilon_tau = torch.randn(target_delta_tau.shape, device=time.device, dtype=time.dtype, generator=generator)
    correction_tau = ideal_tau + remaining * tau_noise_std_s * epsilon_tau * off_path_mask
    correction_xi = se3_log(correction_transform.reshape(-1, 4, 4)).reshape(*batch_shape, 6)

    remaining_transform = torch.linalg.solve(correction_transform, target_transform)
    remaining_xi = se3_log(remaining_transform.reshape(-1, 4, 4)).reshape(*batch_shape, 6)
    target_velocity = torch.cat((remaining_xi / remaining, (target_delta_tau - correction_tau) / remaining), dim=-1)
    if not bool(torch.isfinite(target_velocity).all()):
        raise FloatingPointError("Nonfinite correction-to-go target; inspect SE(3) branch or calibration targets.")
    return CalibrationFlowBatch(correction_transform, correction_xi, correction_tau, actual_time, target_velocity, state_mode)


__all__ = ["CalibrationFlowBatch", "sample_flow_times", "make_calibration_flow_batch", "make_correction_to_go_flow_batch"]
