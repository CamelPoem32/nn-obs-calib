"""IMU interpolation and optional zero-phase filtering for ROS-free LI-Init.

Function names follow upstream LI_Init. Unlike the original, all filtering is
opt-in, timestamps are never silently extrapolated, and no fixed samples are
removed from short windows. Array inputs represent one or more channels.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

from .structures import AngularVelocitySeries, IMUSeries, LIInitConfig


def interpolate_vectors(t: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Interpolate a 3-axis signal without ever extrapolating outside its support."""
    t = np.asarray(t, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if query.ndim != 1 or not np.all(np.isfinite(query)):
        raise ValueError("query timestamps must be a finite 1D array")
    if np.any(query < t[0] - 1e-10) or np.any(query > t[-1] + 1e-10):
        raise ValueError("Requested timestamps extend beyond IMU coverage; interpolation never extrapolates")
    return np.column_stack([np.interp(np.clip(query, t[0], t[-1]), t, values[:, i]) for i in range(values.shape[1])])


def downsample_interpolate_IMU(imu: IMUSeries, lidar_t: np.ndarray, config: LIInitConfig | None = None) -> IMUSeries:
    """Interpolate raw IMU data to LiDAR timestamps (LI-Init function name).

    The upstream routine also applies a 5-sample mean filter to accelerometer
    data. Here it is enabled only through ``config.filtering_enabled`` and
    leaves gyro measurements untouched before interpolation.
    """
    cfg = config or LIInitConfig()
    target = np.asarray(lidar_t, dtype=np.float64)
    if target.ndim != 1 or target.size < 2 or np.any(np.diff(target) <= 0):
        raise ValueError("lidar_t must contain at least two increasing timestamps")
    acc = imu.data_acc
    if acc is not None and cfg.filtering_enabled and cfg.mean_filter_half_window:
        h = cfg.mean_filter_half_window
        weights = np.ones(2 * h + 1, dtype=np.float64) / (2 * h + 1)
        padded = np.pad(acc, ((h, h), (0, 0)), mode="reflect")
        acc = np.column_stack([np.convolve(padded[:, i], weights, mode="valid") for i in range(3)])
    return IMUSeries(t=target, data_gyr=interpolate_vectors(imu.t, imu.data_gyr, target), data_acc=None if acc is None else interpolate_vectors(imu.t, acc, target))


def Butter_filt(values: np.ndarray, config: LIInitConfig | None = None) -> np.ndarray:
    """One-pass low-pass Butterworth filtering, the upstream LI-Init function name.

    SciPy designs exact coefficients from the configured order and normalized
    cutoff, rather than copying upstream rounded coefficient literals.
    This function filters regardless of ``filtering_enabled``: calling it is
    an explicit request. The pipeline calls it only if filtering is enabled.
    """
    cfg = config or LIInitConfig()
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim not in (1, 2) or arr.shape[0] < 2:
        raise ValueError("values must have shape [T] or [T,C], T >= 2")
    sos = butter(cfg.filter_order, cfg.filter_normalized_cutoff, btype="lowpass", output="sos")
    return sosfilt(sos, arr, axis=0)


def zero_phase_filt(signal: np.ndarray | IMUSeries | AngularVelocitySeries, config: LIInitConfig | None = None) -> np.ndarray | IMUSeries | AngularVelocitySeries:
    """Forward-backward Butterworth low-pass filter (LI-Init function name).

    The upstream pipeline invokes Butter_filt twice with reversal. SciPy's
    second-order-section implementation is more stable. Filtering is invoked
    directly or by the estimator only when explicitly requested; defaults in
    LIInitConfig keep filtering off. Short sequences fail explicitly instead
    of being silently truncated/padded with fabricated data.
    """
    cfg = config or LIInitConfig()
    sos = butter(cfg.filter_order, cfg.filter_normalized_cutoff, btype="lowpass", output="sos")

    def filt(x: np.ndarray) -> np.ndarray:
        try:
            return sosfiltfilt(sos, np.asarray(x, dtype=np.float64), axis=0)
        except ValueError as exc:
            raise ValueError(f"Window too short for order={cfg.filter_order} zero-phase filtering: {exc}") from exc

    if isinstance(signal, IMUSeries):
        return IMUSeries(signal.t, filt(signal.data_gyr), None if signal.data_acc is None else filt(signal.data_acc))
    if isinstance(signal, AngularVelocitySeries):
        return AngularVelocitySeries(signal.t, filt(signal.ang_vel), None if signal.ang_acc is None else filt(signal.ang_acc))
    return filt(np.asarray(signal, dtype=np.float64))


def normalize_acc(imu: IMUSeries, gravity_norm: float = 9.81, head_samples: int = 9) -> IMUSeries:
    """Apply LI-Init-style accelerometer scale normalization; opt-in only.

    Assumes the initial samples are approximately stationary. It should NOT
    normally be used for arbitrary 5 s motion windows.
    """
    if imu.data_acc is None:
        raise ValueError("IMU accelerometer samples are required")
    if gravity_norm <= 0 or head_samples < 1:
        raise ValueError("gravity_norm and head_samples must be positive")
    mean_acc = imu.data_acc[:head_samples].mean(axis=0)
    norm = float(np.linalg.norm(mean_acc))
    if norm < 1e-8:
        raise ValueError("Cannot normalize accelerometer: initial mean is near zero")
    return IMUSeries(imu.t, imu.data_gyr, imu.data_acc * (gravity_norm / norm))
