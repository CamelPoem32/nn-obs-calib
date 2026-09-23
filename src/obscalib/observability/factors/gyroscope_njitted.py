"""Numba kernels for the analytic gyroscope observability factor.

The compiled core accepts NumPy float64 arrays and returns raw residual/Jacobian
blocks. Public PyTorch/reference dataclasses remain in ``gyroscope.py`` and
``linearization.py``.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from obscalib.geometry.lie_njitted import so3_exp_njitted, so3_left_jacobian_inverse_njitted, so3_left_jacobian_njitted, so3_log_njitted


@njit 
def interpolate_gyroscope_linear_njitted(timestamps: np.ndarray, samples: np.ndarray, query_time: float) -> np.ndarray:
    """Linearly interpolate a sorted 3D gyroscope signal without extrapolation."""

    index = int(np.searchsorted(timestamps, query_time))

    if index < timestamps.shape[0] and timestamps[index] == query_time:
        return samples[index].copy()

    if index == 0 or index >= timestamps.shape[0]:
        return np.empty(0, dtype=np.float64)

    left = index - 1
    right = index
    alpha = (query_time - timestamps[left]) / (timestamps[right] - timestamps[left])
    return (1.0 - alpha) * samples[left] + alpha * samples[right]


@njit 
def gyroscope_interval_is_supported_njitted(timestamps: np.ndarray, true_start_time: float, true_end_time: float, time_offset: float, time_offset_sign: float = -1.0) -> bool:
    """Return whether both shifted integration endpoints lie inside measured support."""

    lower = true_start_time + time_offset_sign * time_offset
    upper = true_end_time + time_offset_sign * time_offset
    return lower >= timestamps[0] and upper <= timestamps[-1] and upper > lower


@njit 
def integrate_gyroscope_signal_linear_njitted(timestamps: np.ndarray, samples: np.ndarray, lower_time: float, upper_time: float, bias: np.ndarray) -> np.ndarray:
    """Exact trapezoidal integral of a piecewise-linear gyroscope signal."""

    result = np.zeros(3, dtype=np.float64)
    omega_left = interpolate_gyroscope_linear_njitted(timestamps, samples, lower_time)

    if omega_left.shape[0] != 3:
        return np.empty(0, dtype=np.float64)

    previous_time = lower_time
    previous_omega = omega_left - bias
    first_interior = int(np.searchsorted(timestamps, lower_time, side="right"))
    last_interior = int(np.searchsorted(timestamps, upper_time, side="left"))

    for sample_index in range(first_interior, last_interior):
        current_time = timestamps[sample_index]
        current_omega = samples[sample_index] - bias
        dt = current_time - previous_time
        result += 0.5 * dt * (previous_omega + current_omega)
        previous_time = current_time
        previous_omega = current_omega

    omega_right = interpolate_gyroscope_linear_njitted(timestamps, samples, upper_time)

    if omega_right.shape[0] != 3:
        return np.empty(0, dtype=np.float64)

    dt = upper_time - previous_time
    result += 0.5 * dt * (previous_omega + omega_right - bias)
    return result


@njit 
def linearize_gyroscope_factor_njitted(start_body_pose: np.ndarray, end_body_pose: np.ndarray, body_from_imu: np.ndarray, gyro_bias: np.ndarray, imu_time_offset: float, true_start_time: float, true_end_time: float, imu_sensor_timestamps: np.ndarray, gyroscope_samples: np.ndarray, time_offset_sign: float = -1.0):
    """Return residual, H_start, H_end, H_T_B_I, H_b_g, H_tau_I for one analytic gyroscope factor."""

    shifted_start = true_start_time + time_offset_sign * imu_time_offset
    shifted_end = true_end_time + time_offset_sign * imu_time_offset
    phi = integrate_gyroscope_signal_linear_njitted(imu_sensor_timestamps, gyroscope_samples, shifted_start, shifted_end, gyro_bias)

    if phi.shape[0] != 3:
        empty = np.empty((0, 0), dtype=np.float64)
        return np.empty(0, dtype=np.float64), empty, empty, empty, empty, empty

    R_k = start_body_pose[:3, :3]
    R_k1 = end_body_pose[:3, :3]
    C = body_from_imu[:3, :3]
    delta_R = so3_exp_njitted(phi)
    Q = C @ delta_R @ C.T
    E_I = R_k @ Q @ R_k1.T
    residual = so3_log_njitted(E_I)

    omega_start = interpolate_gyroscope_linear_njitted(imu_sensor_timestamps, gyroscope_samples, shifted_start) - gyro_bias
    omega_end = interpolate_gyroscope_linear_njitted(imu_sensor_timestamps, gyroscope_samples, shifted_end) - gyro_bias
    J_l_inv_r = so3_left_jacobian_inverse_njitted(residual)
    J_l_phi = so3_left_jacobian_njitted(phi)

    H_start = np.zeros((3, 6), dtype=np.float64)
    H_start[:, :3] = J_l_inv_r

    H_end = np.zeros((3, 6), dtype=np.float64)
    H_end[:, :3] = -(J_l_inv_r @ E_I)

    H_extrinsic = np.zeros((3, 6), dtype=np.float64)
    H_extrinsic[:, :3] = J_l_inv_r @ R_k @ (np.eye(3, dtype=np.float64) - Q)

    delta_t = true_end_time - true_start_time
    H_bias = -(J_l_inv_r @ R_k @ C @ J_l_phi) * delta_t

    q = time_offset_sign * (omega_end - omega_start)
    H_tau_vector = J_l_inv_r @ R_k @ C @ J_l_phi @ q
    H_tau = H_tau_vector.reshape((3, 1))

    return residual, H_start, H_end, H_extrinsic, H_bias, H_tau


__all__ = [
    "gyroscope_interval_is_supported_njitted",
    "integrate_gyroscope_signal_linear_njitted",
    "interpolate_gyroscope_linear_njitted",
    "linearize_gyroscope_factor_njitted",
]
