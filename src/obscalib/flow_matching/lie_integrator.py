"""Lie-group ODE integration of a predicted calibration *correction* velocity.

The evolving state is (D_t, delta_tau_t) in SE(3) x R, initialized at (I, 0).
The velocity is left-trivialized and composed on the right of D_t. The final
correction is still applied on the left of the calibration extrinsic:
    T_new = D_1 @ T_current.

The 7-vector convention is [phi_x, phi_y, phi_z, rho_x, rho_y, rho_z, delta_tau].
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

from obscalib.geometry.lie import se3_exp, se3_log


# Callback: (current_delta_xi [..., 6], current_delta_tau [..., 1], time [..., 1])
#            -> predicted physical velocity [..., 7].
CalibrationVelocityField = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class CalibrationFlowState:
    """A sampled correction state at an integration time."""

    time: float
    correction_transform: torch.Tensor
    correction_tau: torch.Tensor


@dataclass(frozen=True)
class CalibrationFlowResult:
    """Final integrated correction, ready for CalibrationPrediction."""

    delta_xi: torch.Tensor
    delta_tau: torch.Tensor
    correction_transform: torch.Tensor
    trajectory: tuple[CalibrationFlowState, ...] | None = None


def integrate_calibration_flow(velocity_field: CalibrationVelocityField, reference: torch.Tensor, *, num_steps: int = 8, method: Literal["euler", "midpoint"] = "midpoint", record_trajectory: bool = False) -> CalibrationFlowResult:
    """Integrate from zero correction to time 1, without requiring a noise level.

    `reference` provides shape/device/dtype, normally pooled features [B, C, H]
    or flattened features [B*C, H]; it is not fed to the field automatically.
    The callback closes over the transformer features, original prior and any
    additional conditioning. This avoids reevaluating the transformer each step.

    This function is differentiable in principle, but the *primary FM loss*
    trains the velocity at analytic path samples instead of backpropagating
    through this solver. For validation/inference use torch.no_grad().

    Lie-Euler: one field evaluation per step. Lie-midpoint: two per step.
    """

    if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps <= 0:
        raise ValueError("num_steps must be a positive integer.")
    if method not in ("euler", "midpoint"):
        raise ValueError("method must be 'euler' or 'midpoint'.")
    if reference.ndim < 2 or not reference.is_floating_point() or reference.shape[-1] < 1:
        raise ValueError("reference must be floating with shape [..., H] and at least one batch dimension.")

    batch_shape = reference.shape[:-1]
    identity = torch.eye(4, dtype=reference.dtype, device=reference.device)
    correction_transform = identity.expand(*batch_shape, 4, 4).clone()
    correction_tau = reference.new_zeros(*batch_shape, 1)
    dt = 1.0 / num_steps
    trajectory = [CalibrationFlowState(0.0, correction_transform, correction_tau)] if record_trajectory else None

    def evaluate(transform: torch.Tensor, tau: torch.Tensor, time_value: float, *, identity_state: bool = False) -> torch.Tensor:
        current_xi = reference.new_zeros(*batch_shape, 6) if identity_state else se3_log(transform.reshape(-1, 4, 4)).reshape(*batch_shape, 6)
        time = reference.new_full((*batch_shape, 1), time_value)
        velocity = velocity_field(current_xi, tau, time)
        if not isinstance(velocity, torch.Tensor) or velocity.shape != batch_shape + (7,):
            raise ValueError(f"velocity_field must return a tensor of shape {batch_shape + (7,)}.")
        if velocity.dtype != reference.dtype or velocity.device != reference.device:
            raise ValueError("velocity_field output must match reference dtype and device.")
        return velocity

    for step in range(num_steps):
        t = step * dt
        velocity_start = evaluate(correction_transform, correction_tau, t, identity_state=(step == 0))
        if method == "midpoint":
            halfway_transform = correction_transform @ se3_exp(((0.5 * dt) * velocity_start[..., :6]).reshape(-1, 6)).reshape(*batch_shape, 4, 4)
            halfway_tau = correction_tau + (0.5 * dt) * velocity_start[..., 6:]
            velocity = evaluate(halfway_transform, halfway_tau, (step + 0.5) * dt)
        else:
            velocity = velocity_start

        correction_transform = correction_transform @ se3_exp((dt * velocity[..., :6]).reshape(-1, 6)).reshape(*batch_shape, 4, 4)
        correction_tau = correction_tau + dt * velocity[..., 6:]
        if trajectory is not None:
            trajectory.append(CalibrationFlowState((step + 1) * dt, correction_transform, correction_tau))

    return CalibrationFlowResult(delta_xi=se3_log(correction_transform.reshape(-1, 4, 4)).reshape(*batch_shape, 6), delta_tau=correction_tau, correction_transform=correction_transform, trajectory=tuple(trajectory) if trajectory is not None else None)


__all__ = ["CalibrationVelocityField", "CalibrationFlowState", "CalibrationFlowResult", "integrate_calibration_flow"]
