"""ROS-free LI-Init-inspired orchestration of odometry/IMU calibration.

Original LI_Initialization stages: interpolate, filter, correlate timing,
differentiate, solve rotation, joint gyro bias/time refinement, then translate.
For 5-second windows we skip fixed head/tail discards, avoid normalizing
accelerometer by default, and never modify the original measurement clocks.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .lidar_motion import lidar_increments_to_motion
from .preprocessing import normalize_acc, zero_phase_filt
from .rotation import solve_Rotation_only, solve_Rot_bias_gyro
from .structures import IMUSeries, LidarIncrements, LIInitConfig, LIInitializationResult, RotationCalibrationResult, TranslationCalibrationResult
from .temporal import xcorr_temporal_init
from .translation import solve_trans_biasacc_grav


def LI_Initialization(lidar: LidarIncrements, imu: IMUSeries, config: LIInitConfig | None = None) -> LIInitializationResult:
    """Estimate a LiDAR-to-IMU rotation and relative clock offset, optionally p_IL.

    Relative LiDAR rotations suffice for the primary estimate. Translation
    requires consecutive relative SE(3) poses and IMU accelerometer samples.
    Default initial rotation, Ceres-like loss and filter coefficients follow
    LI-Init choices. Enable ``rotation_initialization='svd'`` for large angles.
    """
    cfg = config or LIInitConfig()
    motion = lidar_increments_to_motion(lidar)
    imu_processed = imu
    if cfg.filtering_enabled:
        motion = zero_phase_filt(motion, cfg)
        imu_processed = zero_phase_filt(imu, cfg)
    if cfg.normalize_acceleration:
        imu_processed = normalize_acc(imu_processed, cfg.gravity_norm)

    temporal = xcorr_temporal_init(motion, imu_processed, cfg) if cfg.estimate_time_offset else None
    candidates = temporal.candidate_offsets if temporal is not None and temporal.success else np.array([cfg.initial_time_offset])
    if temporal is not None and not temporal.success:
        candidates = np.array([cfg.initial_time_offset])

    # The SVD + centered-bias initializer is our short-window extension.
    # Identity initialization preserves the original rotation-only -> joint
    # refinement sequence and accepts large-angle failures as baseline outcomes.
    stage_results: list[tuple[RotationCalibrationResult, RotationCalibrationResult]] = []
    errors = []
    for coarse in np.unique(candidates):
        coarse = float(coarse)
        lo, hi = cfg.time_offset_bounds
        if cfg.temporal_residual == "taylor" and cfg.estimate_time_offset:
            lo = max(lo, coarse - cfg.temporal_local_half_width)
            hi = min(hi, coarse + cfg.temporal_local_half_width)
            if hi <= lo:
                continue
        candidate_cfg = replace(cfg, filtering_enabled=False, initial_time_offset=coarse, time_offset_bounds=(lo, hi))
        try:
            rotation_only = solve_Rotation_only(motion, imu_processed, candidate_cfg, time_offset=coarse)
            initial_rotation = rotation_only.R_LI if cfg.rotation_initialization == "identity" else None
            joint = solve_Rot_bias_gyro(motion, imu_processed, candidate_cfg, initial_R_LI=initial_rotation)
            stage_results.append((joint, rotation_only))
        except (ValueError, np.linalg.LinAlgError) as exc:
            errors.append(f"time_offset={coarse:.6f}: {exc}")

    if not stage_results:
        raise ValueError("No feasible rotation/time calibration candidate. " + "; ".join(errors))

    # Rank/observability takes precedence over RMS. It is important that a
    # low-residual but single-axis fit not be chosen over informative motion.
    joint, rotation_only = min(stage_results, key=lambda pair: (not pair[0].success, pair[0].rms_residual))
    translation: TranslationCalibrationResult | None = None
    if cfg.estimate_translation:
        translation = solve_trans_biasacc_grav(lidar, imu_processed, joint.R_LI, joint.time_offset, cfg)
    return LIInitializationResult(rotation=joint, temporal=temporal, translation=translation, rotation_only=rotation_only)


class LIInitCalibrator:
    """Small state-free front end for ``LI_Initialization`` (no ROS runtime)."""

    def __init__(self, config: LIInitConfig | None = None) -> None:
        self.config = config or LIInitConfig()

    def fit(self, imu: IMUSeries, lidar: LidarIncrements) -> LIInitializationResult:
        """Calibrate LiDAR/IMU extrinsics from relative SE(3) and raw IMU."""
        return LI_Initialization(lidar=lidar, imu=imu, config=self.config)
