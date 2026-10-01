"""Tests for frame invariance of dimensionless soft-projector observability."""

from __future__ import annotations

import torch

from obscalib.geometry.lie import se3_exp


DTYPE = torch.float64

ROTATION_SCALE_RAD = torch.deg2rad(torch.tensor(5.0, dtype=DTYPE)).item()
TRANSLATION_SCALE_M = 0.05
TIME_OFFSET_SCALE_S = 0.015
LAMBDA_VALUE_SETTING = 0.25

ATOL = 1e-10
RTOL = 1e-9


##################################################
# Test helpers
##################################################

def _rotation_matrix(axis: torch.Tensor, angle_rad: float) -> torch.Tensor:
    """Construct one deterministic SO(3) rotation."""

    axis = axis.to(dtype=DTYPE)
    axis = axis / torch.linalg.vector_norm(axis)

    phi = axis * angle_rad

    xi = torch.zeros(6, dtype=DTYPE)
    xi[:3] = phi

    return se3_exp(xi)[:3, :3]


def _make_spd_matrix(dimension: int, seed: int) -> torch.Tensor:
    """Construct one deterministic positive-definite covariance matrix."""

    generator = torch.Generator().manual_seed(seed)

    A = torch.randn(
        dimension,
        dimension,
        generator=generator,
        dtype=DTYPE,
    )

    return (
        A @ A.T
        + 0.5 * torch.eye(
            dimension,
            dtype=DTYPE,
        )
    )


def _make_calibration_scale_vector(
    number_calibrations: int,
) -> torch.Tensor:
    """Return repeated [phi, rho, tau] physical scales."""

    one_sensor_scale = torch.tensor(
        [
            ROTATION_SCALE_RAD,
            ROTATION_SCALE_RAD,
            ROTATION_SCALE_RAD,
            TRANSLATION_SCALE_M,
            TRANSLATION_SCALE_M,
            TRANSLATION_SCALE_M,
            TIME_OFFSET_SCALE_S,
        ],
        dtype=DTYPE,
    )

    return one_sensor_scale.repeat(
        number_calibrations
    )


def _compute_fisher(
    J: torch.Tensor,
    covariance: torch.Tensor,
) -> torch.Tensor:
    """Compute J.T Sigma^-1 J without forming Sigma^-1 explicitly."""

    whitened_solution = torch.linalg.solve(
        covariance,
        J,
    )

    F = (
        J.T
        @ whitened_solution
    )

    return 0.5 * (
        F
        + F.T
    )


def _dimensionless_soft_projector(
    fisher_information_matrix: torch.Tensor,
    number_calibrations: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Compute the dimensionless Fisher matrix, soft projector, and its diagonal.

    The metric is

        F_u = D.T F D

        g_i = lambda_i / (lambda_i + lambda_setting)

        P_soft = V diag(g_i) V.T

        o = diag(P_soft)
    """

    scale_vector = _make_calibration_scale_vector(
        number_calibrations
    )

    D = torch.diag(
        scale_vector
    )

    F_u = (
        D.T
        @ fisher_information_matrix
        @ D
    )

    F_u = 0.5 * (
        F_u
        + F_u.T
    )

    eigenvalues, eigenvectors = torch.linalg.eigh(
        F_u
    )

    eigenvalues = torch.clamp(
        eigenvalues,
        min=0.0,
    )

    soft_mode_strength = (
        eigenvalues
        / (
            eigenvalues
            + LAMBDA_VALUE_SETTING
        )
    )

    P_soft = (
        eigenvectors
        @ torch.diag(
            soft_mode_strength
        )
        @ eigenvectors.T
    )

    P_soft = 0.5 * (
        P_soft
        + P_soft.T
    )

    coordinate_observability = torch.diagonal(
        P_soft
    )

    return (
        F_u,
        P_soft,
        coordinate_observability,
    )


def _block_diag(
    *blocks: torch.Tensor,
) -> torch.Tensor:
    """Construct a block-diagonal matrix from square blocks."""

    return torch.block_diag(
        *blocks
    )


##################################################
# Left-perturbation convention
##################################################

def test_left_calibration_perturbation_is_unchanged_by_right_sensor_frame_relabeling() -> None:
    """
    Verify the calibration tangent convention used by the invariance argument.

    Original sensor calibration:

        T_B_S

    Relabeled sensor frame:

        T_B_Sp = T_B_S A

    With a left perturbation,

        Exp(delta_xi) T_B_Sp
        =
        (Exp(delta_xi) T_B_S) A

    so the same delta_xi describes the same physical calibration perturbation
    before and after the sensor-coordinate relabeling.
    """

    T_B_S = se3_exp(
        torch.tensor(
            [
                0.30,
                -0.15,
                0.08,
                0.40,
                -0.20,
                0.10,
            ],
            dtype=DTYPE,
        )
    )

    A = torch.eye(
        4,
        dtype=DTYPE,
    )

    A[:3, :3] = _rotation_matrix(
        torch.tensor(
            [
                1.0,
                2.0,
                -0.5,
            ],
            dtype=DTYPE,
        ),
        angle_rad=0.9,
    )

    delta_xi = torch.tensor(
        [
            0.02,
            -0.03,
            0.01,
            0.005,
            -0.004,
            0.002,
        ],
        dtype=DTYPE,
    )

    delta_T = se3_exp(
        delta_xi
    )

    T_B_S_randomized = (
        T_B_S
        @ A
    )

    perturbed_after_relabeling = (
        delta_T
        @ T_B_S_randomized
    )

    relabeled_after_perturbation = (
        delta_T
        @ T_B_S
        @ A
    )

    torch.testing.assert_close(
        perturbed_after_relabeling,
        relabeled_after_perturbation,
        atol=ATOL,
        rtol=RTOL,
    )


##################################################
# Fisher-information invariance
##################################################

def test_fisher_information_is_invariant_to_consistent_sensor_frame_rotation() -> None:
    """
    Verify Fisher invariance under a pure change of residual coordinates.

    If

        r' = H r
        J' = H J
        Sigma' = H Sigma H.T

    then

        J'.T Sigma'^-1 J' = J.T Sigma^-1 J.

    The blocks below mimic:
        gyro  -> R
        accel -> R
        lidar -> blockdiag(R, R)

    with anisotropic covariances so this test is not accidentally passing only
    because identity/isotropic covariance commutes with every rotation.
    """

    number_calibrations = 2
    calibration_dimension = (
        7
        * number_calibrations
    )

    ##################################################
    # Deterministic synthetic calibration Jacobian
    ##################################################

    generator = torch.Generator().manual_seed(
        1234
    )

    J_gyro = torch.randn(
        3,
        calibration_dimension,
        generator=generator,
        dtype=DTYPE,
    )

    J_accel = torch.randn(
        3,
        calibration_dimension,
        generator=generator,
        dtype=DTYPE,
    )

    J_lidar = torch.randn(
        6,
        calibration_dimension,
        generator=generator,
        dtype=DTYPE,
    )

    J = torch.cat(
        (
            J_gyro,
            J_accel,
            J_lidar,
        ),
        dim=0,
    )

    ##################################################
    # Anisotropic residual covariance
    ##################################################

    Sigma_gyro = _make_spd_matrix(
        3,
        seed=1,
    )

    Sigma_accel = _make_spd_matrix(
        3,
        seed=2,
    )

    Sigma_lidar = _make_spd_matrix(
        6,
        seed=3,
    )

    Sigma = _block_diag(
        Sigma_gyro,
        Sigma_accel,
        Sigma_lidar,
    )

    ##################################################
    # One arbitrary physical frame rotation
    ##################################################

    R = _rotation_matrix(
        torch.tensor(
            [
                0.4,
                -1.0,
                0.7,
            ],
            dtype=DTYPE,
        ),
        angle_rad=1.1,
    )

    H_gyro = R
    H_accel = R

    H_lidar = _block_diag(
        R,
        R,
    )

    H = _block_diag(
        H_gyro,
        H_accel,
        H_lidar,
    )

    ##################################################
    # Consistently transform residual Jacobians and covariance
    ##################################################

    J_randomized = (
        H
        @ J
    )

    Sigma_randomized = (
        H
        @ Sigma
        @ H.T
    )

    ##################################################
    # Fisher matrices must represent exactly the same information
    ##################################################

    F = _compute_fisher(
        J,
        Sigma,
    )

    F_randomized = _compute_fisher(
        J_randomized,
        Sigma_randomized,
    )

    torch.testing.assert_close(
        F_randomized,
        F,
        atol=ATOL,
        rtol=RTOL,
    )


##################################################
# Soft-projector invariance
##################################################

def test_dimensionless_soft_projector_is_invariant_to_consistent_sensor_frame_rotation() -> None:
    """
    Verify invariance of the complete dimensionless soft projector.

    This is the main scientific property needed for using raw-window
    soft-projector values as frame-randomization-invariant training targets.
    """

    number_calibrations = 2
    calibration_dimension = (
        7
        * number_calibrations
    )

    generator = torch.Generator().manual_seed(
        5678
    )

    ##################################################
    # Synthetic multimodal residual system
    ##################################################

    J_gyro = torch.randn(
        3,
        calibration_dimension,
        generator=generator,
        dtype=DTYPE,
    )

    J_accel = torch.randn(
        3,
        calibration_dimension,
        generator=generator,
        dtype=DTYPE,
    )

    J_lidar = torch.randn(
        6,
        calibration_dimension,
        generator=generator,
        dtype=DTYPE,
    )

    J = torch.cat(
        (
            J_gyro,
            J_accel,
            J_lidar,
        ),
        dim=0,
    )

    Sigma = _block_diag(
        _make_spd_matrix(
            3,
            seed=11,
        ),
        _make_spd_matrix(
            3,
            seed=12,
        ),
        _make_spd_matrix(
            6,
            seed=13,
        ),
    )

    ##################################################
    # Pure sensor-coordinate rotation
    ##################################################

    R = _rotation_matrix(
        torch.tensor(
            [
                -0.3,
                0.8,
                1.2,
            ],
            dtype=DTYPE,
        ),
        angle_rad=0.75,
    )

    H = _block_diag(
        R,
        R,
        _block_diag(
            R,
            R,
        ),
    )

    J_randomized = (
        H
        @ J
    )

    Sigma_randomized = (
        H
        @ Sigma
        @ H.T
    )

    ##################################################
    # Fisher information
    ##################################################

    F = _compute_fisher(
        J,
        Sigma,
    )

    F_randomized = _compute_fisher(
        J_randomized,
        Sigma_randomized,
    )

    ##################################################
    # Dimensionless soft projector
    ##################################################

    (
        F_u,
        P_soft,
        observability,
    ) = _dimensionless_soft_projector(
        F,
        number_calibrations,
    )

    (
        F_u_randomized,
        P_soft_randomized,
        observability_randomized,
    ) = _dimensionless_soft_projector(
        F_randomized,
        number_calibrations,
    )

    ##################################################
    # Check every level, not only the final seven-value diagonal
    ##################################################

    torch.testing.assert_close(
        F_u_randomized,
        F_u,
        atol=ATOL,
        rtol=RTOL,
    )

    torch.testing.assert_close(
        P_soft_randomized,
        P_soft,
        atol=ATOL,
        rtol=RTOL,
    )

    torch.testing.assert_close(
        observability_randomized,
        observability,
        atol=ATOL,
        rtol=RTOL,
    )


##################################################
# Important negative control
##################################################

def test_frame_rotation_without_covariance_rotation_is_not_generally_invariant() -> None:
    """
    Verify that rotating measurements/Jacobians while leaving anisotropic
    residual covariance in the old coordinate frame changes Fisher information.

    This negative control is important: if the real augmentation path changes
    sensor coordinates but residual covariance is not transformed consistently,
    the resulting observability calculation can change even though the physical
    experiment has not changed.
    """

    number_calibrations = 2
    calibration_dimension = (
        7
        * number_calibrations
    )

    generator = torch.Generator().manual_seed(
        9012
    )

    J = torch.randn(
        12,
        calibration_dimension,
        generator=generator,
        dtype=DTYPE,
    )

    Sigma = _block_diag(
        _make_spd_matrix(
            3,
            seed=21,
        ),
        _make_spd_matrix(
            3,
            seed=22,
        ),
        _make_spd_matrix(
            6,
            seed=23,
        ),
    )

    R = _rotation_matrix(
        torch.tensor(
            [
                0.7,
                0.2,
                -1.0,
            ],
            dtype=DTYPE,
        ),
        angle_rad=1.25,
    )

    H = _block_diag(
        R,
        R,
        _block_diag(
            R,
            R,
        ),
    )

    ##################################################
    # Deliberately wrong transformation:
    # Jacobian changes frame, covariance does not.
    ##################################################

    J_randomized = (
        H
        @ J
    )

    F = _compute_fisher(
        J,
        Sigma,
    )

    F_inconsistent = _compute_fisher(
        J_randomized,
        Sigma,
    )

    (
        _,
        P_soft,
        _,
    ) = _dimensionless_soft_projector(
        F,
        number_calibrations,
    )

    (
        _,
        P_soft_inconsistent,
        _,
    ) = _dimensionless_soft_projector(
        F_inconsistent,
        number_calibrations,
    )

    fisher_difference = torch.linalg.matrix_norm(
        F_inconsistent
        - F
    )

    projector_difference = torch.linalg.matrix_norm(
        P_soft_inconsistent
        - P_soft
    )

    assert float(
        fisher_difference
    ) > 1e-5

    assert float(
        projector_difference
    ) > 1e-5