"""Whitened rotation-preintegration and two-LiDAR SE(3) factors.

Gyro factor: Log(DeltaR_imu^-1 * R_IL DeltaR_lidar R_IL^T).
Pair factor: Log((T_IR DeltaT_R T_IR^-1)^-1 * (T_IL DeltaT_L T_IL^-1)).

The pair factor includes both rotation and translation with separate sigmas.
Only left extrinsic *rotation* is variable; its IMU-frame lever arm is fixed.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .geometry import invert_T, log_se3, log_so3
from .lidar_pair import LidarMotionPairs
from .structures import MotionRefinementConfig, MotionRefinementInput
from .preintegration import integrate_gyro_intervals


@dataclass(frozen=True)
class RefinementState:
    R_LI: np.ndarray
    gyro_bias: np.ndarray
    time_offset: float


@dataclass(frozen=True)
class ResidualBlock:
    name: str
    values: np.ndarray  # [num_factors, dim], already divided by standard deviation
    robust: bool = True


class MotionFactors:
    """Stateful factor assembly with fixed geometry and cached IMU integration.

    Residual counts are fixed across nonlinear iterations; time bounds are
    checked before optimization, not by silently dropping out-of-range edges.
    """
    def __init__(self, data: MotionRefinementInput, config: MotionRefinementConfig, pair: LidarMotionPairs | None):
        self.data = data
        self.config = config
        self.pair = pair
        self.R_prior = data.T_I_L_prior[:3, :3]
        self.bias_prior = data.initial_gyro_bias
        self.tau_prior = data.initial_time_offset
        left_edges = data.lidar.delta_T
        if data.lidar.increment_convention == 'end_to_start':
            left_edges = invert_T(left_edges)
        self.left_rotations = left_edges[:, :3, :3].copy()
        if config.use_gyro_factor:
            if len(self.left_rotations) < config.min_gyro_intervals:
                raise ValueError('Insufficient LiDAR intervals for gyro factors')
            if config.optimize_time_offset:
                lo, hi = config.time_offset_bounds
                if data.lidar.t_start.min() + lo < data.imu.t[0] - 1e-8 or data.lidar.t_end.max() + hi > data.imu.t[-1] + 1e-8:
                    raise ValueError('Extra IMU padding is required for the entire time_offset_bounds')
            self.cached_imu_rotations = None
            if not (config.optimize_gyro_bias or config.optimize_time_offset):
                self.cached_imu_rotations = self._integrate(data.initial_gyro_bias, data.initial_time_offset)

    def _integrate(self, gyro_bias: np.ndarray, tau: float) -> np.ndarray:
        return integrate_gyro_intervals(self.data.imu, self.data.lidar.t_start, self.data.lidar.t_end, time_offset=tau, gyro_bias=gyro_bias, max_sample_gap_s=self.config.imu_max_sample_gap_s)

    def evaluate(self, state: RefinementState, *, include_prior: bool = True) -> list[ResidualBlock]:
        cfg = self.config
        out: list[ResidualBlock] = []
        if cfg.use_gyro_factor:
            d_i = self.cached_imu_rotations
            if d_i is None:
                d_i = self._integrate(state.gyro_bias, state.time_offset)
            d_pred = state.R_LI @ self.left_rotations @ state.R_LI.T
            r = log_so3(np.swapaxes(d_i, -1, -2) @ d_pred)
            out.append(ResidualBlock('gyro_rotation', r / cfg.gyro_rotation_sigma_rad))

        if self.pair is not None:
            T_I_L = self.data.T_I_L_prior.copy()
            T_I_L[:3, :3] = state.R_LI
            T_I_R = self.data.T_I_R
            dl = T_I_L @ self.pair.left @ invert_T(T_I_L)
            dr = T_I_R @ self.pair.right @ invert_T(T_I_R)
            r = log_se3(invert_T(dr) @ dl)
            whiten = np.array([cfg.lidar_pair_rotation_sigma_rad] * 3 + [cfg.lidar_pair_translation_sigma_m] * 3)
            out.append(ResidualBlock('lidar_pair', r / whiten))

        if include_prior and cfg.use_prior:
            r_prior = log_so3(state.R_LI @ self.R_prior.T) / np.deg2rad(cfg.prior_rotation_sigma_deg)
            out.append(ResidualBlock('rotation_prior', r_prior.reshape(1, 3), robust=False))
            if cfg.optimize_gyro_bias:
                r_bias = (state.gyro_bias - self.bias_prior) / cfg.prior_gyro_bias_sigma_radps
                out.append(ResidualBlock('gyro_bias_prior', r_bias.reshape(1, 3), robust=False))
            if cfg.optimize_time_offset:
                r_tau = (state.time_offset - self.tau_prior) / cfg.prior_time_offset_sigma_s
                out.append(ResidualBlock('time_offset_prior', np.array([[r_tau]], dtype=float), robust=False))
        return out


def robust_weights(blocks: list[ResidualBlock], config: MotionRefinementConfig) -> list[np.ndarray]:
    """IRLS weights on factor *norms*, never on individual SO(3) coordinates."""
    weights = []
    for block in blocks:
        n = np.linalg.norm(block.values, axis=1)
        if not block.robust or config.loss == 'linear':
            weight = np.ones_like(n)
        elif config.loss == 'huber':
            weight = np.minimum(1.0, config.robust_delta / np.maximum(n, 1e-12))
        else:
            weight = 1.0 / (1.0 + (n / config.robust_delta) ** 2)
        weights.append(np.sqrt(weight))
    return weights


def flatten_blocks(blocks: list[ResidualBlock], sqrt_weights: list[np.ndarray] | None = None) -> np.ndarray:
    if sqrt_weights is None:
        return np.concatenate([b.values.reshape(-1) for b in blocks])
    if len(blocks) != len(sqrt_weights):
        raise ValueError('Factor weight layout changed')
    return np.concatenate([(b.values * w[:, None]).reshape(-1) for b, w in zip(blocks, sqrt_weights)])


def factor_cost(blocks: list[ResidualBlock], config: MotionRefinementConfig) -> float:
    """Actual robust objective (rather than frozen-IRLS quadratic surrogate)."""
    cost = 0.0
    for block in blocks:
        a = np.linalg.norm(block.values, axis=1)
        if not block.robust or config.loss == 'linear':
            rho = 0.5 * a * a
        elif config.loss == 'huber':
            d = config.robust_delta
            rho = np.where(a <= d, 0.5 * a * a, d * (a - 0.5 * d))
        else:
            d = config.robust_delta
            rho = 0.5 * d * d * np.log1p((a / d) ** 2)
        cost += float(rho.sum())
    return cost


def measurement_rms(factors: MotionFactors, state: RefinementState) -> tuple[float, float | None, float | None]:
    """Report raw units, independently of optimization weights."""
    blocks = {b.name: b.values for b in factors.evaluate(state, include_prior=False)}
    gyro = blocks.get('gyro_rotation')
    pair = blocks.get('lidar_pair')
    a = float(np.sqrt(np.mean(np.sum((gyro * factors.config.gyro_rotation_sigma_rad) ** 2, axis=1)))) if gyro is not None else np.nan
    b = float(np.sqrt(np.mean(np.sum((pair[:, :3] * factors.config.lidar_pair_rotation_sigma_rad) ** 2, axis=1)))) if pair is not None else None
    c = float(np.sqrt(np.mean(np.sum((pair[:, 3:] * factors.config.lidar_pair_translation_sigma_m) ** 2, axis=1)))) if pair is not None else None
    return a, b, c
