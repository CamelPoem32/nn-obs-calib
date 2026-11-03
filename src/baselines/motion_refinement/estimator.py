"""Public local extrinsic rotation refinement API. No ground truth or NN dependencies."""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter
import numpy as np

from baselines.li_init.structures import IMUSeries, LidarIncrements
from .geometry import log_so3
from .lidar_pair import align_lidar_motion_pairs
from .optimizer import run_lm
from .residuals import MotionFactors, measurement_rms
from .structures import MotionRefinementConfig, MotionRefinementInput, MotionRefinementResult


class MotionRefinementCalibrator:
    """Locally refine a supplied LiDAR->IMU rotation on short motion windows.

    Estimation is deterministic, holds LiDAR extrinsic translations fixed, and
    can optionally add a fixed-right-LiDAR SE(3) relative-motion factor.
    """
    def __init__(self, config: MotionRefinementConfig | None = None):
        self.config = config or MotionRefinementConfig()

    def fit(self, imu: IMUSeries, lidar: LidarIncrements, *, initial_R_LI: np.ndarray | None = None, initial_T_I_L: np.ndarray | None = None, right_lidar: LidarIncrements | None = None, T_I_R: np.ndarray | None = None, initial_time_offset: float = 0.0, initial_gyro_bias: np.ndarray | None = None, p_I_L: np.ndarray | None = None, right_time_offset_s: float = 0.0) -> MotionRefinementResult:
        """Return the refined rotation and cumulative correction from the prior.

        Supply either initial_R_LI with p_I_L (default lever arm: 0), or the
        full initial_T_I_L. When two LiDARs are used, T_I_R must be known and
        expressed in the same IMU reference frame.
        """
        if (initial_R_LI is None) == (initial_T_I_L is None):
            raise ValueError('Supply exactly one of initial_R_LI and initial_T_I_L')
        if initial_T_I_L is None:
            R = np.asarray(initial_R_LI, dtype=np.float64)
            p = np.zeros(3) if p_I_L is None else np.asarray(p_I_L, dtype=np.float64)
            if p.shape != (3,) or not np.isfinite(p).all():
                raise ValueError('p_I_L must be a finite 3-vector')
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = p
        else:
            if p_I_L is not None:
                raise ValueError('p_I_L cannot be supplied with initial_T_I_L')
            T = np.asarray(initial_T_I_L, dtype=np.float64)
        inp = MotionRefinementInput(imu=imu, lidar=lidar, T_I_L_prior=T, right_lidar=right_lidar, T_I_R=T_I_R, initial_time_offset=initial_time_offset, initial_gyro_bias=np.zeros(3) if initial_gyro_bias is None else initial_gyro_bias, right_time_offset_s=right_time_offset_s)
        return self.refine(inp)

    def refine(self, data: MotionRefinementInput) -> MotionRefinementResult:
        """Refine a prevalidated task and report objective, factors and LM trace."""
        cfg = self.config
        use_pair = (data.right_lidar is not None and cfg.use_lidar_pair != 'off')
        if cfg.use_lidar_pair == 'required' and not use_pair:
            raise ValueError('Two-LiDAR mode requires right_lidar and T_I_R')
        pair = None
        tic = perf_counter()
        if use_pair:
            pair = align_lidar_motion_pairs(data.lidar, data.right_lidar, max_gap_s=cfg.pair_max_scan_gap_s, min_intervals=cfg.pair_min_intervals, right_time_offset_s=data.right_time_offset_s)
        if pair is None and not cfg.use_gyro_factor:
            raise ValueError('No measurement factors enabled')
        factors = MotionFactors(data, cfg, pair)
        optimized = run_lm(factors, cfg)
        R = optimized.state.R_LI
        T = data.T_I_L_prior.copy()
        T[:3, :3] = R
        rotation_rms, pair_rotation_rms, pair_translation_rms = measurement_rms(factors, optimized.state)
        delta = log_so3(R @ data.T_I_L_prior[:3, :3].T)
        elapsed = 1000.0 * (perf_counter() - tic)
        return MotionRefinementResult(R_LI=R.copy(), T_I_L=T, delta_rotvec=delta, gyro_bias=optimized.state.gyro_bias.copy(), time_offset=optimized.state.time_offset, success=optimized.success, observable=optimized.observability.rank == 3, message=optimized.message, iterations=len(optimized.trace), accepted_iterations=sum(v.accepted for v in optimized.trace), initial_cost=optimized.initial_cost, final_cost=optimized.final_cost, elapsed_ms=elapsed, gyro_rotation_rms_rad=rotation_rms, lidar_pair_rotation_rms_rad=pair_rotation_rms, lidar_pair_translation_rms_m=pair_translation_rms, gyro_factor_count=len(data.lidar.t_start) if cfg.use_gyro_factor else 0, lidar_pair_factor_count=0 if pair is None else len(pair.left), observability=optimized.observability, trace=optimized.trace)
