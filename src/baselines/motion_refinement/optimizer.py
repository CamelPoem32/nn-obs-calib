"""Small, explicit Levenberg–Marquardt on SO(3), with bounded nuisance states.

Implements LM's damped normal equations, relinearization, acceptance/rejection,
robust factor weights, and optional TSVD projection of rotational updates.
This is a local optimizer; it does not estimate a global Wahba initialization.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .geometry import exp_so3, log_so3
from .observability import analyze_rotation_observability
from .residuals import MotionFactors, RefinementState, flatten_blocks, factor_cost, robust_weights
from .structures import IterationTrace, MotionRefinementConfig, RotationObservability


@dataclass(frozen=True)
class OptimizationOutput:
    state: RefinementState
    success: bool
    message: str
    trace: tuple[IterationTrace, ...]
    initial_cost: float
    final_cost: float
    observability: RotationObservability


def parameter_count(config: MotionRefinementConfig) -> int:
    return 3 + (3 if config.optimize_gyro_bias else 0) + (1 if config.optimize_time_offset else 0)


def retract(state: RefinementState, dx: np.ndarray, config: MotionRefinementConfig) -> RefinementState:
    """Left-multiply SO(3) correction and add Euclidean bias/time increments."""
    if dx.shape != (parameter_count(config),):
        raise ValueError('Wrong tangent-step dimension')
    j = 3
    b = state.gyro_bias.copy()
    tau = state.time_offset
    if config.optimize_gyro_bias:
        b += dx[j:j+3]
        j += 3
    if config.optimize_time_offset:
        tau += dx[j]
    return RefinementState(exp_so3(dx[:3]) @ state.R_LI, b, float(tau))


def state_in_bounds(state: RefinementState, config: MotionRefinementConfig, initial_bias: np.ndarray) -> bool:
    if config.optimize_time_offset and not (config.time_offset_bounds[0] <= state.time_offset <= config.time_offset_bounds[1]):
        return False
    if config.optimize_gyro_bias and np.linalg.norm(state.gyro_bias - initial_bias, ord=np.inf) > config.gyro_bias_bounds_radps:
        return False
    return True


def finite_difference_jacobian(factors: MotionFactors, state: RefinementState, config: MotionRefinementConfig, *, include_prior: bool, frozen_weights: list[np.ndarray] | None = None) -> np.ndarray:
    """Numeric tangent-space Jacobian; can be upgraded to analytic derivatives.

    Factors have a constant layout: all interval measurements are retained
    across all trial states. For IRLS, the current state's weights stay frozen.
    """
    base_blocks = factors.evaluate(state, include_prior=include_prior)
    base = flatten_blocks(base_blocks, frozen_weights)
    J = np.empty((base.size, parameter_count(config)), dtype=np.float64)
    eps = config.finite_difference_epsilon
    for col in range(J.shape[1]):
        dx = np.zeros(J.shape[1], dtype=np.float64)
        dx[col] = eps
        plus = retract(state, dx, config)
        # Near bounds use one-sided finite differences instead of invalid trials.
        if not state_in_bounds(plus, config, factors.data.initial_gyro_bias):
            dx[col] = -eps
            minus = retract(state, dx, config)
            if not state_in_bounds(minus, config, factors.data.initial_gyro_bias):
                J[:, col] = 0
                continue
            v = flatten_blocks(factors.evaluate(minus, include_prior=include_prior), frozen_weights)
            J[:, col] = (base - v) / eps
            continue
        pos = flatten_blocks(factors.evaluate(plus, include_prior=include_prior), frozen_weights)
        if config.finite_difference_method == 'central':
            dx[col] = -eps
            minus = retract(state, dx, config)
            if state_in_bounds(minus, config, factors.data.initial_gyro_bias):
                neg = flatten_blocks(factors.evaluate(minus, include_prior=include_prior), frozen_weights)
                J[:, col] = (pos - neg) / (2 * eps)
                continue
        J[:, col] = (pos - base) / eps
    return J


def run_lm(factors: MotionFactors, config: MotionRefinementConfig) -> OptimizationOutput:
    """Minimize independent gyro-rotation, LiDAR-pair and optional prior factors."""
    data = factors.data
    state = RefinementState(data.T_I_L_prior[:3, :3].copy(), data.initial_gyro_bias.copy(), float(data.initial_time_offset))
    if config.optimize_time_offset and not config.time_offset_bounds[0] <= state.time_offset <= config.time_offset_bounds[1]:
        raise ValueError('initial_time_offset outside time_offset_bounds')
    if config.optimize_gyro_bias and not config.use_gyro_factor:
        raise ValueError('Cannot optimize gyro bias when gyro factor is off')
    if config.optimize_time_offset and not config.use_gyro_factor:
        raise ValueError('Cannot optimize time offset when gyro factor is off')

    initial_blocks = factors.evaluate(state)
    cost = factor_cost(initial_blocks, config)
    first_cost = cost
    damping = config.initial_damping
    trace: list[IterationTrace] = []
    success = True
    message = 'maximum iterations reached'

    for iteration in range(1, config.max_iterations + 1):
        blocks = factors.evaluate(state)
        weights = robust_weights(blocks, config)
        r = flatten_blocks(blocks, weights)
        J = finite_difference_jacobian(factors, state, config, include_prior=True, frozen_weights=weights)
        # Measurement blocks always precede Gaussian priors. Reuse the Jacobian
        # already computed for LM, avoiding a second expensive preintegration
        # pass per tangent parameter. The robust weights remain frozen.
        n_meas = sum(b.values.size for b in blocks if b.name in ('gyro_rotation', 'lidar_pair'))
        J_m = J[:n_meas]
        # Measurement-only observability; no prior can create excitation.
        obs = analyze_rotation_observability(J_m, config)
        g = J.T @ r
        grad = float(np.linalg.norm(g, ord=np.inf))
        if grad <= config.gradient_tolerance:
            message = 'gradient tolerance reached'
            break
        if config.project_unobservable_updates and obs.rank == 0:
            message = 'rotation not observable; retained supplied prior'
            break

        H = J.T @ J
        diag = np.maximum(np.diag(H), 1e-6)
        accepted = False
        cost_before = cost
        rotation_step_deg = 0.0
        note = 'LM trial limit reached'
        for trial in range(config.max_trials_per_iteration):
            try:
                step = np.linalg.solve(H + damping * np.diag(diag), -g)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H + damping * np.diag(diag), -g, rcond=None)[0]
            if config.project_unobservable_updates:
                step[:3] = obs.projector @ step[:3]
            a = np.linalg.norm(step[:3])
            limit = np.deg2rad(config.max_rotation_step_deg)
            if a > limit:
                step[:3] *= limit / a
            # Constrain nuisance updates for stable relinearization.
            j = 3
            if config.optimize_gyro_bias:
                s = np.linalg.norm(step[j:j+3], ord=np.inf)
                if s > 0.02:
                    step[j:j+3] *= 0.02 / s
                j += 3
            if config.optimize_time_offset:
                step[j] = np.clip(step[j], -0.02, 0.02)
            rotation_step_deg = float(np.rad2deg(np.linalg.norm(step[:3])))
            candidate = retract(state, step, config)
            if not state_in_bounds(candidate, config, data.initial_gyro_bias):
                damping = min(config.max_damping, damping * config.damping_increase)
                note = 'nuisance parameter bounds'
                continue
            try:
                candidate_cost = factor_cost(factors.evaluate(candidate), config)
            except (ValueError, FloatingPointError):
                candidate_cost = np.inf
            if np.isfinite(candidate_cost) and candidate_cost < cost - 1e-13:
                accepted = True
                state = candidate
                cost = candidate_cost
                damping = max(config.min_damping, damping / config.damping_decrease)
                note = 'accepted'
                break
            damping = min(config.max_damping, damping * config.damping_increase)
            note = 'objective did not decrease'

        total_delta = float(np.rad2deg(np.linalg.norm(log_so3(state.R_LI @ data.T_I_L_prior[:3, :3].T))))
        trace.append(IterationTrace(iteration, accepted, cost_before, cost, damping, rotation_step_deg, grad, total_delta, note))
        if not accepted:
            if rotation_step_deg < np.rad2deg(config.step_tolerance) or damping >= config.max_damping:
                message = 'no improving step; converged or constrained'
                break
            continue
        if abs(cost_before - cost) <= config.cost_tolerance * max(1.0, cost_before):
            message = 'cost tolerance reached'
            break
        if rotation_step_deg < np.rad2deg(config.step_tolerance) and not config.optimize_gyro_bias and not config.optimize_time_offset:
            message = 'rotation step tolerance reached'
            break

    # Diagnostics at the FINAL solution; measurement prior not counted.
    J_final = finite_difference_jacobian(factors, state, config, include_prior=False)
    obs = analyze_rotation_observability(J_final, config)
    return OptimizationOutput(state=state, success=bool(success and np.isfinite(cost)), message=message, trace=tuple(trace), initial_cost=first_cost, final_cost=cost, observability=obs)
