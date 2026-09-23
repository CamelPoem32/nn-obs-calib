"""Numba-compiled NumPy Lie-group kernels used by CPU observability factors.

The canonical public/reference implementation remains ``obscalib.geometry.lie``.
This module is intentionally small, allocation-conscious, float64-oriented, and
contains no PyTorch objects so Numba can compile complete factor kernels around it.

Tangent convention is rotation first:
    SO(3): phi = [phi_x, phi_y, phi_z]
    SE(3): xi = [phi_x, phi_y, phi_z, rho_x, rho_y, rho_z]
"""

from __future__ import annotations

import numpy as np
from numba import njit


_SO3_EPS = 1e-10
_SO3_NEAR_PI_EPS = 1e-5
_SE3_SERIES_TOLERANCE = 1e-14
_SE3_SERIES_MAX_TERMS = 80


@njit 
def so3_hat_njitted(vector: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix of one length-three vector."""

    x, y, z = vector[0], vector[1], vector[2]
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=np.float64)


@njit 
def so3_vee_njitted(matrix: np.ndarray) -> np.ndarray:
    """Return the vector associated with one 3x3 skew-symmetric matrix."""

    return np.array((matrix[2, 1], matrix[0, 2], matrix[1, 0]), dtype=np.float64)


@njit 
def so3_exp_njitted(phi: np.ndarray) -> np.ndarray:
    """SO(3) exponential with stable small-angle coefficients."""

    theta2 = float(phi @ phi)
    hat = so3_hat_njitted(phi)
    hat2 = hat @ hat

    if theta2 < 1e-16:
        A = 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0
        B = 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0
    else:
        theta = np.sqrt(theta2)
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / theta2

    return np.eye(3, dtype=np.float64) + A * hat + B * hat2


@njit 
def _so3_axis_near_pi_njitted(rotation: np.ndarray) -> np.ndarray:
    """Recover a stable unit rotation axis when theta is close to pi."""

    diagonal = np.diag(rotation)
    axis = np.empty(3, dtype=np.float64)
    index = int(np.argmax(diagonal))

    if index == 0:
        axis[0] = np.sqrt(max(0.0, 0.5 * (rotation[0, 0] + 1.0)))
        denominator = max(4.0 * axis[0], 1e-15)
        axis[1] = (rotation[0, 1] + rotation[1, 0]) / denominator
        axis[2] = (rotation[0, 2] + rotation[2, 0]) / denominator
    elif index == 1:
        axis[1] = np.sqrt(max(0.0, 0.5 * (rotation[1, 1] + 1.0)))
        denominator = max(4.0 * axis[1], 1e-15)
        axis[0] = (rotation[0, 1] + rotation[1, 0]) / denominator
        axis[2] = (rotation[1, 2] + rotation[2, 1]) / denominator
    else:
        axis[2] = np.sqrt(max(0.0, 0.5 * (rotation[2, 2] + 1.0)))
        denominator = max(4.0 * axis[2], 1e-15)
        axis[0] = (rotation[0, 2] + rotation[2, 0]) / denominator
        axis[1] = (rotation[1, 2] + rotation[2, 1]) / denominator

    norm = np.sqrt(float(axis @ axis))

    if norm < 1e-12:
        skew = so3_vee_njitted(rotation - rotation.T)
        skew_norm = np.sqrt(float(skew @ skew))
        if skew_norm < 1e-12:
            return np.array((1.0, 0.0, 0.0), dtype=np.float64)
        return skew / skew_norm

    axis /= norm

    skew = so3_vee_njitted(rotation - rotation.T)
    if float(axis @ skew) < 0.0:
        axis = -axis

    return axis


@njit 
def so3_log_njitted(rotation: np.ndarray) -> np.ndarray:
    """SO(3) logarithm with explicit small-angle and near-pi branches."""

    cos_theta = 0.5 * (float(np.trace(rotation)) - 1.0)
    cos_theta = min(1.0, max(-1.0, cos_theta))
    theta = np.arccos(cos_theta)

    if theta < _SO3_EPS:
        return 0.5 * so3_vee_njitted(rotation - rotation.T)

    if np.pi - theta < _SO3_NEAR_PI_EPS:
        return theta * _so3_axis_near_pi_njitted(rotation)

    return (theta / (2.0 * np.sin(theta))) * so3_vee_njitted(rotation - rotation.T)


@njit 
def so3_left_jacobian_njitted(phi: np.ndarray) -> np.ndarray:
    """SO(3) left Jacobian."""

    theta2 = float(phi @ phi)
    hat = so3_hat_njitted(phi)
    hat2 = hat @ hat

    if theta2 < 1e-16:
        A = 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0
        B = 1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0
    else:
        theta = np.sqrt(theta2)
        A = (1.0 - np.cos(theta)) / theta2
        B = (theta - np.sin(theta)) / (theta2 * theta)

    return np.eye(3, dtype=np.float64) + A * hat + B * hat2


@njit 
def so3_left_jacobian_inverse_njitted(phi: np.ndarray) -> np.ndarray:
    """Inverse SO(3) left Jacobian."""

    theta2 = float(phi @ phi)
    hat = so3_hat_njitted(phi)
    hat2 = hat @ hat

    if theta2 < 1e-16:
        coefficient = 1.0 / 12.0 + theta2 / 720.0 + theta2 * theta2 / 30240.0
    else:
        theta = np.sqrt(theta2)
        half_theta = 0.5 * theta
        coefficient = (1.0 - half_theta / np.tan(half_theta)) / theta2

    return np.eye(3, dtype=np.float64) - 0.5 * hat + coefficient * hat2


@njit 
def se3_inverse_njitted(transform: np.ndarray) -> np.ndarray:
    """Inverse one homogeneous SE(3) transform."""

    result = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    rotation_transpose = rotation.T
    result[:3, :3] = rotation_transpose
    result[:3, 3] = -(rotation_transpose @ translation)
    return result


@njit 
def se3_adjoint_njitted(transform: np.ndarray) -> np.ndarray:
    """SE(3) adjoint for the rotation-first [phi, rho] convention."""

    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    result = np.zeros((6, 6), dtype=np.float64)
    result[:3, :3] = rotation
    result[3:, :3] = so3_hat_njitted(translation) @ rotation
    result[3:, 3:] = rotation
    return result


@njit 
def se3_little_adjoint_njitted(xi: np.ndarray) -> np.ndarray:
    """Lie-algebra adjoint ad_xi for xi=[phi,rho]."""

    phi_hat = so3_hat_njitted(xi[:3])
    rho_hat = so3_hat_njitted(xi[3:])
    result = np.zeros((6, 6), dtype=np.float64)
    result[:3, :3] = phi_hat
    result[3:, :3] = rho_hat
    result[3:, 3:] = phi_hat
    return result


@njit 
def se3_exp_njitted(xi: np.ndarray) -> np.ndarray:
    """SE(3) exponential for xi=[phi,rho]."""

    phi = xi[:3]
    rho = xi[3:]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = so3_exp_njitted(phi)
    result[:3, 3] = so3_left_jacobian_njitted(phi) @ rho
    return result


@njit 
def se3_log_njitted(transform: np.ndarray) -> np.ndarray:
    """SE(3) logarithm returning xi=[phi,rho]."""

    phi = so3_log_njitted(transform[:3, :3])
    rho = so3_left_jacobian_inverse_njitted(phi) @ transform[:3, 3]
    result = np.empty(6, dtype=np.float64)
    result[:3] = phi
    result[3:] = rho
    return result


@njit 
def se3_left_jacobian_njitted(xi: np.ndarray, tolerance: float = _SE3_SERIES_TOLERANCE, max_terms: int = _SE3_SERIES_MAX_TERMS) -> np.ndarray:
    """SE(3) left Jacobian from the same algebra-adjoint power series as the reference implementation."""

    ad = se3_little_adjoint_njitted(xi)
    result = np.eye(6, dtype=np.float64)
    power = np.eye(6, dtype=np.float64)
    factorial = 1.0

    for n in range(1, max_terms):
        power = power @ ad
        factorial *= float(n + 1)
        term = power / factorial
        result += term

        term_norm_squared = 0.0
        for row in range(6):
            for column in range(6):
                term_norm_squared += term[row, column] * term[row, column]

        if np.sqrt(term_norm_squared) < tolerance:
            return result

    return result


@njit 
def se3_left_jacobian_inverse_njitted(xi: np.ndarray) -> np.ndarray:
    """Inverse SE(3) left Jacobian."""

    return np.linalg.solve(se3_left_jacobian_njitted(xi), np.eye(6, dtype=np.float64))


__all__ = [
    "se3_adjoint_njitted",
    "se3_exp_njitted",
    "se3_inverse_njitted",
    "se3_left_jacobian_inverse_njitted",
    "se3_left_jacobian_njitted",
    "se3_little_adjoint_njitted",
    "se3_log_njitted",
    "so3_exp_njitted",
    "so3_hat_njitted",
    "so3_left_jacobian_inverse_njitted",
    "so3_left_jacobian_njitted",
    "so3_log_njitted",
    "so3_vee_njitted",
]
