"""LI-Init-style coarse temporal synchronization without ROS or resampling drift.

The sign convention is ``t_IMU = t_LiDAR + offset``. Unlike the upstream
integer-lag zero-centered correlation, this adaptation handles arbitrary
sample times and uses overlap-normalized Pearson correlation. Low excitation
and ambiguous offsets are reported, not silently accepted.
"""

from __future__ import annotations

import numpy as np

from .preprocessing import interpolate_vectors
from .structures import AngularVelocitySeries, IMUSeries, LIInitConfig, TemporalCalibrationResult


def xcorr_temporal_init(lidar: AngularVelocitySeries, imu: IMUSeries | AngularVelocitySeries, config: LIInitConfig | None = None) -> TemporalCalibrationResult:
    """Estimate coarse IMU-to-LiDAR clock offset using angular-speed correlation.

    Upstream uses a zero-centered discrete cross-correlation with lag step
    ``1 / odom_freq``. This version interpolates the native IMU clock and
    searches a configurable range with the median LiDAR interval as its
    default step. A candidate requires sufficient overlap and signal variance.
    """
    cfg = config or LIInitConfig()
    imu_t = imu.t
    imu_gyr = imu.data_gyr if isinstance(imu, IMUSeries) else imu.ang_vel
    lo, hi = cfg.time_offset_bounds
    dt = cfg.xcorr_step if cfg.xcorr_step is not None else float(np.median(np.diff(lidar.t)))
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("Invalid xcorr search step")
    offsets = np.arange(lo, hi + dt * 0.5, dt)
    offsets = np.unique(np.clip(np.r_[offsets, cfg.initial_time_offset, hi], lo, hi))
    lidar_norm = np.linalg.norm(lidar.ang_vel, axis=1)
    scores = np.full(offsets.shape, np.nan)
    overlap_counts = np.zeros(offsets.shape, dtype=int)
    min_overlap = max(cfg.min_samples, int(np.ceil(cfg.xcorr_min_overlap_fraction * lidar.t.size)))
    for k, tau in enumerate(offsets):
        query = lidar.t + tau
        valid = (query >= imu_t[0]) & (query <= imu_t[-1])
        n = int(valid.sum())
        overlap_counts[k] = n
        if n < min_overlap:
            continue
        x = lidar_norm[valid]
        y = np.linalg.norm(interpolate_vectors(imu_t, imu_gyr, query[valid]), axis=1)
        x = x - x.mean()
        y = y - y.mean()
        den = float(np.linalg.norm(x) * np.linalg.norm(y))
        if np.std(x) < cfg.xcorr_min_std or np.std(y) < cfg.xcorr_min_std or den < 1e-12:
            continue
        scores[k] = float(x @ y / den)

    eligible = np.flatnonzero(np.isfinite(scores))
    if eligible.size == 0:
        return TemporalCalibrationResult(cfg.initial_time_offset, False, "No informative, overlapping angular-speed pairs", np.array([]), offsets, scores, overlap_counts)

    ordered = sorted(eligible, key=lambda k: (-scores[k], abs(offsets[k] - cfg.initial_time_offset)))
    selected = []
    min_sep = cfg.xcorr_peak_separation if cfg.xcorr_peak_separation is not None else max(dt * 1.5, 1e-6)
    for idx in ordered:
        if all(abs(offsets[idx] - offsets[j]) >= min_sep for j in selected):
            selected.append(int(idx))
        if len(selected) >= cfg.xcorr_num_peaks:
            break
    best = selected[0]
    success = bool(scores[best] >= cfg.xcorr_min_score)
    return TemporalCalibrationResult(float(offsets[best]), success, "Peak found" if success else "Weak angular-speed correlation", offsets[selected], offsets, scores, overlap_counts)


def IMU_time_compensate(imu: IMUSeries, lag_time: float, is_discard: bool = False, discard_initial: int = 10) -> IMUSeries:
    """Shift the IMU clock into LiDAR time (LI-Init function name).

    Upstream subtracts lag from IMU timestamps and optionally discards 10
    initial states. Here discarding is OFF by default for 5-second windows.
    For estimation, prefer evaluating raw IMU at ``t_LiDAR + lag_time`` so
    the reported offset remains in the original timestamp convention.
    """
    if not np.isfinite(lag_time) or discard_initial < 0:
        raise ValueError("Invalid time compensation parameters")
    i = discard_initial if is_discard else 0
    if imu.t.size - i < 2:
        raise ValueError("Not enough IMU samples after initial discard")
    return IMUSeries(imu.t[i:] - lag_time, imu.data_gyr[i:], None if imu.data_acc is None else imu.data_acc[i:])


def acc_interpolate(imu: IMUSeries, t_lidar: np.ndarray, time_offset: float) -> np.ndarray:
    """Interpolate IMU specific force at LiDAR times after clock correction."""
    if imu.data_acc is None:
        raise ValueError("No accelerometer data available")
    return interpolate_vectors(imu.t, imu.data_acc, np.asarray(t_lidar, dtype=float) + time_offset)
