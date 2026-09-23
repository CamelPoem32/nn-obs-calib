"""Reference-equivalence tests for Numba observability kernels."""

from __future__ import annotations

import numpy as np
import torch

from obscalib.geometry.lie import se3_adjoint, se3_exp, se3_inverse, se3_left_jacobian, se3_left_jacobian_inverse, se3_log, so3_exp, so3_left_jacobian, so3_left_jacobian_inverse, so3_log
from obscalib.geometry.lie_njitted import se3_adjoint_njitted, se3_exp_njitted, se3_inverse_njitted, se3_left_jacobian_inverse_njitted, se3_left_jacobian_njitted, se3_log_njitted, so3_exp_njitted, so3_left_jacobian_inverse_njitted, so3_left_jacobian_njitted, so3_log_njitted
from obscalib.observability.factors.gyroscope import linearize_gyroscope_factor
from obscalib.observability.factors.gyroscope_njitted import linearize_gyroscope_factor_njitted
from obscalib.observability.factors.lidar import linearize_lidar_factor
from obscalib.observability.factors.lidar_njitted import linearize_lidar_factor_njitted


DTYPE = torch.float64
ATOL = 2e-9
RTOL = 2e-8


def _np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def test_lie_njitted_matches_reference() -> None:
    phi = torch.tensor([0.31, -0.22, 0.17], dtype=DTYPE)
    xi = torch.tensor([0.31, -0.22, 0.17, 0.8, -0.3, 0.2], dtype=DTYPE)
    R = so3_exp(phi)
    T = se3_exp(xi)

    np.testing.assert_allclose(so3_exp_njitted(_np(phi)), _np(R), atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(so3_log_njitted(_np(R)), _np(so3_log(R)), atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(so3_left_jacobian_njitted(_np(phi)), _np(so3_left_jacobian(phi)), atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(so3_left_jacobian_inverse_njitted(_np(phi)), _np(so3_left_jacobian_inverse(phi)), atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(se3_exp_njitted(_np(xi)), _np(T), atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(se3_log_njitted(_np(T)), _np(se3_log(T)), atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(se3_inverse_njitted(_np(T)), _np(se3_inverse(T)), atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(se3_adjoint_njitted(_np(T)), _np(se3_adjoint(T)), atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(se3_left_jacobian_njitted(_np(xi)), _np(se3_left_jacobian(xi)), atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(se3_left_jacobian_inverse_njitted(_np(xi)), _np(se3_left_jacobian_inverse(xi)), atol=ATOL, rtol=RTOL)


def test_gyroscope_factor_njitted_matches_reference() -> None:
    timestamps = torch.linspace(0.0, 1.0, 101, dtype=DTYPE)
    samples = torch.stack((0.2 + 0.03 * timestamps, -0.1 + 0.02 * timestamps, 0.15 - 0.01 * timestamps), dim=1)
    T0 = torch.eye(4, dtype=DTYPE)
    T1 = torch.eye(4, dtype=DTYPE)
    T1[:3, :3] = so3_exp(torch.tensor([0.018, -0.009, 0.014], dtype=DTYPE))
    X = torch.eye(4, dtype=DTYPE)
    X[:3, :3] = so3_exp(torch.tensor([0.03, -0.02, 0.01], dtype=DTYPE))
    bias = torch.tensor([0.001, -0.002, 0.0005], dtype=DTYPE)
    reference = linearize_gyroscope_factor(T0, T1, X, bias, 0.002, 0.2, 0.8, timestamps, samples)
    actual = linearize_gyroscope_factor_njitted(_np(T0), _np(T1), _np(X), _np(bias), 0.002, 0.2, 0.8, _np(timestamps), _np(samples))

    for actual_block, reference_block in zip(actual, (reference.residual, reference.H_start_pose, reference.H_end_pose, reference.H_T_B_I, reference.H_b_g, reference.H_tau_I)):
        np.testing.assert_allclose(actual_block, _np(reference_block), atol=ATOL, rtol=RTOL)


def test_lidar_factor_njitted_matches_reference() -> None:
    T0 = se3_exp(torch.tensor([0.1, -0.05, 0.03, 1.0, -0.4, 0.2], dtype=DTYPE))
    T1 = se3_exp(torch.tensor([0.13, -0.01, 0.08, 1.4, -0.2, 0.35], dtype=DTYPE))
    X = se3_exp(torch.tensor([-0.2, 0.1, 0.05, 0.3, -0.15, 0.8], dtype=DTYPE))
    Z = se3_inverse(X) @ se3_inverse(T0) @ T1 @ X
    perturbation = se3_exp(torch.tensor([2e-3, -1e-3, 3e-3, 1e-2, -5e-3, 4e-3], dtype=DTYPE))
    Z = perturbation @ Z
    twist0 = torch.tensor([0.02, -0.01, 0.03, 1.0, 0.2, -0.1], dtype=DTYPE)
    twist1 = torch.tensor([0.03, -0.005, 0.025, 1.1, 0.25, -0.08], dtype=DTYPE)
    reference = linearize_lidar_factor(T0, T1, X, Z, twist0, twist1)
    actual = linearize_lidar_factor_njitted(_np(T0), _np(T1), _np(X), _np(Z), _np(twist0), _np(twist1))

    for actual_block, reference_block in zip(actual, (reference.residual, reference.H_start_pose, reference.H_end_pose, reference.H_T_B_L, reference.H_tau_L)):
        np.testing.assert_allclose(actual_block, _np(reference_block), atol=5e-9, rtol=5e-8)
