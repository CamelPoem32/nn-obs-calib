"""Differentiable Euler/midpoint integration of a ten-dimensional 6D rotation flow.

The state is Euclidean during ODE integration. Orthogonalization is performed
only on the final rotation output, never on intermediate states.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

from obscalib.flow_matching.rotation_6d_path import identity_6d_flow_state
from obscalib.geometry.rotation_6d import rotation_6d_to_matrix


Rotation6DVelocityField = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class Rotation6DFlowResult:
    """Integrated correction, with a 6D SO(3) embedding and additive t/tau."""

    state: torch.Tensor  # [...,10]
    trajectory: tuple[torch.Tensor, ...] | None = None

    @property
    def rotation_6d(self) -> torch.Tensor:
        return self.state[..., :6]

    @property
    def delta_rotation(self) -> torch.Tensor:
        return rotation_6d_to_matrix(self.rotation_6d)

    @property
    def delta_translation(self) -> torch.Tensor:
        return self.state[..., 6:9]

    @property
    def delta_tau(self) -> torch.Tensor:
        return self.state[..., 9:10]


def integrate_rotation_6d_flow(
    velocity_field: Rotation6DVelocityField,
    reference: torch.Tensor,
    *,
    num_steps: int = 16,
    method: Literal['euler', 'midpoint'] = 'midpoint',
    record_trajectory: bool = False,
) -> Rotation6DFlowResult:
    """Integrate from the identity correction, keeping autograd for endpoint loss.

    The velocity callback takes (state [...,10], time [...,1]) and returns [...,10].
    A straight 6D interpolation may be rank-deficient at intermediate times;
    the velocity field sees *raw embedding coordinates*, not a rotation matrix.
    """
    if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps < 1:
        raise ValueError('num_steps must be a positive integer')
    if method not in ('euler', 'midpoint'):
        raise ValueError('method must be euler or midpoint')
    state = identity_6d_flow_state(reference)
    batch_shape = state.shape[:-1]
    dt = 1.0 / num_steps
    trajectory = [state] if record_trajectory else None

    def evaluate(current_state: torch.Tensor, t: float) -> torch.Tensor:
        velocity = velocity_field(current_state, current_state.new_full(batch_shape + (1,), t))
        if not isinstance(velocity, torch.Tensor) or velocity.shape != batch_shape + (10,):
            raise ValueError(f'velocity_field must return {batch_shape + (10,)}')
        if velocity.device != current_state.device or velocity.dtype != current_state.dtype:
            raise ValueError('velocity_field output device and dtype must match reference')
        if not bool(torch.isfinite(velocity).all()):
            raise FloatingPointError('Nonfinite velocity during 6D FM integration')
        return velocity

    for step in range(num_steps):
        start_time = step * dt
        start_velocity = evaluate(state, start_time)
        if method == 'midpoint':
            midpoint = state + 0.5 * dt * start_velocity
            velocity = evaluate(midpoint, (step + 0.5) * dt)
        else:
            velocity = start_velocity
        state = state + dt * velocity
        if trajectory is not None:
            trajectory.append(state)

    if not bool(torch.isfinite(state).all()):
        raise FloatingPointError('Nonfinite endpoint after 6D FM integration')
    return Rotation6DFlowResult(state, tuple(trajectory) if trajectory is not None else None)


__all__ = ['Rotation6DVelocityField', 'Rotation6DFlowResult', 'integrate_rotation_6d_flow']
