"""Numba kernels for the analytic LiDAR relative-pose observability factor."""

from __future__ import annotations

import numpy as np
from numba import njit

from obscalib.geometry.lie_njitted import se3_adjoint_njitted, se3_inverse_njitted, se3_left_jacobian_inverse_njitted, se3_log_njitted


@njit 
def linearize_lidar_factor_njitted(start_body_pose: np.ndarray, end_body_pose: np.ndarray, body_from_lidar: np.ndarray, lidar_measurement: np.ndarray, start_spatial_twist: np.ndarray, end_spatial_twist: np.ndarray):
    """Return residual, H_start, H_end, H_T_B_L, H_tau_L for one analytic LiDAR factor."""

    inverse_start = se3_inverse_njitted(start_body_pose)
    inverse_extrinsic = se3_inverse_njitted(body_from_lidar)
    inverse_measurement = se3_inverse_njitted(lidar_measurement)

    relative_body_motion = inverse_start @ end_body_pose
    predicted_lidar_motion = inverse_extrinsic @ relative_body_motion @ body_from_lidar
    error_transform = predicted_lidar_motion @ inverse_measurement
    residual = se3_log_njitted(error_transform)
    J_l_inv_r = se3_left_jacobian_inverse_njitted(residual)

    transport = inverse_extrinsic @ inverse_start
    common = J_l_inv_r @ se3_adjoint_njitted(transport)

    H_start = -common
    H_end = common
    H_extrinsic = J_l_inv_r @ se3_adjoint_njitted(inverse_extrinsic) @ (se3_adjoint_njitted(relative_body_motion) - np.eye(6, dtype=np.float64))
    H_tau = (common @ (end_spatial_twist - start_spatial_twist)).reshape((6, 1))

    return residual, H_start, H_end, H_extrinsic, H_tau


__all__ = ["linearize_lidar_factor_njitted"]
