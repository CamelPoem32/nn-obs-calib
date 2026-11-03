"""Rotation-only on-manifold gyroscope preintegration, inspired by GTSAM/Forster.

The gyro is integrated on the SO(3) manifold, not by summing rotation vectors.
Every LiDAR interval [a,b] is evaluated at IMU clock times [a+tau,b+tau].
Between IMU samples angular velocity is linearly interpolated; midpoint
integration is exact for its interval integral in the commuting limit.

Not a copy of GTSAM's full 9D NavState preintegration, covariance recursion or
first-order bias Jacobian; bias/time changes trigger *exact re-integration*.
"""
from __future__ import annotations

import numpy as np

from baselines.li_init.structures import IMUSeries
from .geometry import exp_so3


def integrate_gyro_intervals(imu: IMUSeries, starts: np.ndarray, ends: np.ndarray, *, time_offset: float = 0.0, gyro_bias: np.ndarray | None = None, max_sample_gap_s: float = 0.15) -> np.ndarray:
    """Integrate rotations in the IMU's moving body axes over many intervals.

    Returns shape [N,3,3], with delta_R[k]=R_WI(a)^T R_WI(b).
    No extrapolation; invalid IMU coverage or sample gaps are errors.
    The union of all sample and interval endpoints is integrated once.
    """
    a = np.asarray(starts, dtype=np.float64) + float(time_offset)
    b = np.asarray(ends, dtype=np.float64) + float(time_offset)
    t = imu.t
    w = imu.data_gyr
    bias = np.zeros(3) if gyro_bias is None else np.asarray(gyro_bias, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or not a.size or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or np.any(b <= a):
        raise ValueError('IMU integration intervals must be finite and strictly positive')
    if bias.shape != (3,) or not np.all(np.isfinite(bias)):
        raise ValueError('gyro_bias must be a finite 3-vector')
    if a.min() < t[0] - 1e-8 or b.max() > t[-1] + 1e-8:
        raise ValueError('IMU does not cover shifted LiDAR interval boundaries (padding or clock bounds required)')
    a = np.clip(a, t[0], t[-1])
    b = np.clip(b, t[0], t[-1])
    low, high = a.min(), b.max()
    within = t[(t > low) & (t < high)]
    grid = np.unique(np.concatenate([a, b, within]))
    if grid.size < 2:
        raise ValueError('Not enough points for integration')

    # Check actual IMU sampling gaps *over the requested recording region*.
    i0 = max(0, int(np.searchsorted(t, low, side='right') - 1))
    i1 = min(len(t), int(np.searchsorted(t, high, side='left') + 1))
    if np.any(np.diff(t[i0:i1]) > max_sample_gap_s):
        raise ValueError('IMU sample gap exceeds max_sample_gap_s')

    mid = (grid[1:] + grid[:-1]) * 0.5
    wmid = np.column_stack([np.interp(mid, t, w[:, d]) for d in range(3)]) - bias
    phi = wmid * np.diff(grid)[:, None]
    increments = exp_so3(phi)
    cumulative = np.empty((grid.size, 3, 3), dtype=np.float64)
    cumulative[0] = np.eye(3)
    for k in range(len(increments)):
        cumulative[k + 1] = cumulative[k] @ increments[k]
    ia = np.searchsorted(grid, a)
    ib = np.searchsorted(grid, b)
    if np.any(ia >= len(grid)) or np.any(ib >= len(grid)) or np.max(np.abs(grid[ia] - a)) > 1e-8 or np.max(np.abs(grid[ib] - b)) > 1e-8:
        raise RuntimeError('IMU integration endpoint alignment error')
    return np.swapaxes(cumulative[ia], 1, 2) @ cumulative[ib]
