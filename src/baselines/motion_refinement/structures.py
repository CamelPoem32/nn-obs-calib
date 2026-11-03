"""Inputs, hyperparameters and diagnostics for local multi-sensor rotation refinement.

The public LI-Init IMUSeries and LidarIncrements types are intentionally reused.
The unknown R_LI maps LiDAR-frame vectors into the reference IMU frame; optional
T_I_R maps RIGHT LiDAR into reference IMU, and T_I_L contains R_LI plus a fixed
left-LiDAR lever arm in IMU coordinates. Clock sign: t_imu=t_lidar+tau.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from baselines.li_init.structures import IMUSeries, LidarIncrements
from .geometry import validate_pose, validate_rotation


@dataclass(frozen=True)
class MotionRefinementConfig:
    """Independent LM, preintegration, factor and observability parameters.

    Noise sigmas are *assumptions*, not measured KAIST/New College covariances;
    calibrate them on training/validation data before accuracy comparisons.
    """
    max_iterations: int = 8
    initial_damping: float = 1e-2
    min_damping: float = 1e-9
    max_damping: float = 1e9
    damping_increase: float = 5.0
    damping_decrease: float = 3.0
    max_trials_per_iteration: int = 8
    gradient_tolerance: float = 1e-7
    step_tolerance: float = 1e-7
    cost_tolerance: float = 1e-9
    max_rotation_step_deg: float = 15.0
    finite_difference_epsilon: float = 1e-5
    finite_difference_method: Literal['forward', 'central'] = 'forward'

    # Rotation-only by default. Bias/time can be optimized using extra IMU padding.
    optimize_gyro_bias: bool = False
    optimize_time_offset: bool = False
    time_offset_bounds: tuple[float, float] = (-0.30, 0.30)
    gyro_bias_bounds_radps: float = 0.20
    imu_max_sample_gap_s: float = 0.15

    # Independent factor sigmas; whitened residuals are dimensionless.
    gyro_rotation_sigma_rad: float = 0.025
    lidar_pair_rotation_sigma_rad: float = 0.025
    lidar_pair_translation_sigma_m: float = 0.25
    use_lidar_pair: Literal['auto', 'off', 'required'] = 'auto'
    pair_max_scan_gap_s: float = 0.75
    pair_min_intervals: int = 3
    use_gyro_factor: bool = True

    # Proper prior on TOTAL correction, not per LM iteration.
    use_prior: bool = True
    prior_rotation_sigma_deg: float = 20.0
    prior_gyro_bias_sigma_radps: float = 0.02
    prior_time_offset_sigma_s: float = 0.05

    # Huber IRLS is applied to individual vector-factor norms; priors remain Gaussian.
    loss: Literal['linear', 'huber', 'cauchy'] = 'huber'
    robust_delta: float = 3.0

    # TSVD acts only along measurement-observable directions after nuisance marginalization.
    project_unobservable_updates: bool = True
    observability_rtol: float = 1e-3
    observability_atol: float = 1e-8
    min_gyro_intervals: int = 3

    def __post_init__(self) -> None:
        positive = ('initial_damping', 'min_damping', 'max_damping', 'damping_increase', 'damping_decrease', 'gradient_tolerance', 'step_tolerance', 'cost_tolerance', 'max_rotation_step_deg', 'finite_difference_epsilon', 'gyro_bias_bounds_radps', 'imu_max_sample_gap_s', 'gyro_rotation_sigma_rad', 'lidar_pair_rotation_sigma_rad', 'lidar_pair_translation_sigma_m', 'pair_max_scan_gap_s', 'prior_rotation_sigma_deg', 'prior_gyro_bias_sigma_radps', 'prior_time_offset_sigma_s', 'robust_delta', 'observability_rtol', 'observability_atol')
        for key in positive:
            x = getattr(self, key)
            if not np.isfinite(x) or x <= 0:
                raise ValueError(f'{key} must be positive and finite')
        if self.max_iterations < 1 or self.max_trials_per_iteration < 1 or self.pair_min_intervals < 1 or self.min_gyro_intervals < 1:
            raise ValueError('Iteration and minimum-factor counts must be positive')
        if not (self.min_damping < self.max_damping and self.damping_increase > 1 and self.damping_decrease > 1):
            raise ValueError('Invalid LM damping range/multipliers')
        if not (0 < self.observability_rtol < 1):
            raise ValueError('observability_rtol must be in (0,1)')
        lo, hi = self.time_offset_bounds
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError('time_offset_bounds must be ordered finite bounds')
        if self.loss not in ('linear', 'huber', 'cauchy') or self.finite_difference_method not in ('forward', 'central') or self.use_lidar_pair not in ('auto', 'off', 'required'):
            raise ValueError('Unsupported solver option')
        if not self.use_gyro_factor and self.use_lidar_pair == 'off':
            raise ValueError('At least one measurement factor must be enabled')


@dataclass(frozen=True)
class MotionRefinementInput:
    """One local calibration task; references/ground truth are never included.

    ``T_I_L_prior`` gives the initial left-LiDAR pose in IMU coordinates.
    The optimization changes only its SO(3) block, not its translation.
    ``T_I_R`` fixes the right-LiDAR pose in the SAME IMU frame.
    ``right_lidar`` and ``T_I_R`` must be supplied together.
    """
    imu: IMUSeries
    lidar: LidarIncrements
    T_I_L_prior: ArrayLike
    right_lidar: LidarIncrements | None = None
    T_I_R: ArrayLike | None = None
    initial_time_offset: float = 0.0
    right_time_offset_s: float = 0.0  # t_right = t_left + this fixed clock shift
    initial_gyro_bias: ArrayLike = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def __post_init__(self) -> None:
        if not isinstance(self.imu, IMUSeries) or not isinstance(self.lidar, LidarIncrements):
            raise TypeError('imu/lidar must be baselines.li_init.structures.IMUSeries/LidarIncrements')
        if (self.right_lidar is None) != (self.T_I_R is None):
            raise ValueError('right_lidar and T_I_R must be supplied together')
        object.__setattr__(self, 'T_I_L_prior', validate_pose(self.T_I_L_prior, 'T_I_L_prior'))
        if self.T_I_R is not None:
            object.__setattr__(self, 'T_I_R', validate_pose(self.T_I_R, 'T_I_R'))
        b = np.asarray(self.initial_gyro_bias, dtype=np.float64)
        if b.shape != (3,) or not np.isfinite(b).all() or not np.isfinite(self.initial_time_offset) or not np.isfinite(self.right_time_offset_s):
            raise ValueError('Invalid initial gyro bias / time offset')
        object.__setattr__(self, 'initial_gyro_bias', b.copy())


@dataclass(frozen=True)
class IterationTrace:
    iteration: int
    accepted: bool
    cost_before: float
    cost_after: float
    damping: float
    rotation_step_deg: float
    gradient_norm: float
    rotation_error_to_prior_deg: float
    note: str


@dataclass(frozen=True)
class RotationObservability:
    singular_values: np.ndarray
    rank: int
    projector: np.ndarray
    condition_number: float
    num_nuisance_parameters: int


@dataclass(frozen=True)
class MotionRefinementResult:
    R_LI: np.ndarray
    T_I_L: np.ndarray
    delta_rotvec: np.ndarray
    gyro_bias: np.ndarray
    time_offset: float
    success: bool
    observable: bool
    message: str
    iterations: int
    accepted_iterations: int
    initial_cost: float
    final_cost: float
    elapsed_ms: float
    gyro_rotation_rms_rad: float
    lidar_pair_rotation_rms_rad: float | None
    lidar_pair_translation_rms_m: float | None
    gyro_factor_count: int
    lidar_pair_factor_count: int
    observability: RotationObservability
    trace: tuple[IterationTrace, ...]


__all__ = ['IMUSeries', 'LidarIncrements', 'MotionRefinementConfig', 'MotionRefinementInput', 'IterationTrace', 'RotationObservability', 'MotionRefinementResult']
