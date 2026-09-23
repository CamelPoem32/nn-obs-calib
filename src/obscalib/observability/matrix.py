"""Canonical one-window calibration Fisher-matrix construction."""

from __future__ import annotations

import torch

from obscalib.observability.linearization import WindowLinearization
from obscalib.observability.numerics import ObservabilityNumericsConfig, singular_value_tolerance, stabilize_psd
from obscalib.observability.structures import WindowObservabilityMatrix
from obscalib.observability.whitening import WhitenedWindowLinearization, whiten_window_linearization


def _combined_marginalized_jacobian(whitened: WhitenedWindowLinearization) -> torch.Tensor:
    """
    Concatenate trajectory and explicit nuisance columns into one matrix.

    Marginalizing the complete ``[J_T, J_N]`` column space in one operation avoids
    order-dependent sequential projections.
    """

    whitened.validate()

    return torch.cat((whitened.trajectory_jacobian, whitened.nuisance_jacobian), dim=1)


def project_calibration_jacobian(whitened: WhitenedWindowLinearization, numerics: ObservabilityNumericsConfig) -> torch.Tensor:
    """
    Remove all calibration sensitivity explainable by trajectory or nuisance variables.

    If ``J_M = [J_T, J_N]`` and ``U_r`` is an orthonormal basis for its numerical
    column space, the projected calibration Jacobian is

        O_C = J_C - U_r (U_r.T J_C).

    SVD is used only to obtain the rank-revealing orthonormal basis. No explicit
    projector and no normal-equation pseudoinverse are formed.

    This function is intentionally self-contained because it is a natural target
    for a future NumPy/Numba backend with the same inputs and outputs.
    """

    whitened.validate()

    marginalized_jacobian = _combined_marginalized_jacobian(whitened)
    calibration_jacobian = whitened.calibration_jacobian

    if marginalized_jacobian.shape[1] == 0:
        return calibration_jacobian.clone()

    U, singular_values, _ = torch.linalg.svd(marginalized_jacobian, full_matrices=False)
    tolerance = singular_value_tolerance(singular_values, tuple(marginalized_jacobian.shape), numerics)
    rank = int(torch.count_nonzero(singular_values > tolerance).item())

    if rank == 0:
        return calibration_jacobian.clone()

    nuisance_basis = U[:, :rank]

    return calibration_jacobian - nuisance_basis @ (nuisance_basis.transpose(-1, -2) @ calibration_jacobian)


def fisher_information_from_projected_jacobian(projected_calibration_jacobian: torch.Tensor, numerics: ObservabilityNumericsConfig) -> torch.Tensor:
    """
    Construct the physical calibration Fisher matrix from ``O_C``.

    The matrix

        F_C = O_C.T @ O_C

    remains in the physical calibration parameterization. Diagnostic parameter
    scaling from ``ObservabilityNumericsConfig`` is deliberately not applied
    here.
    """

    if not isinstance(projected_calibration_jacobian, torch.Tensor):
        raise TypeError("projected_calibration_jacobian must be a torch.Tensor.")
    if projected_calibration_jacobian.ndim != 2:
        raise ValueError("projected_calibration_jacobian must have shape [M, D].")
    if projected_calibration_jacobian.device.type != "cpu":
        raise ValueError("projected_calibration_jacobian must be stored on CPU.")
    if projected_calibration_jacobian.requires_grad:
        raise ValueError("projected_calibration_jacobian must be detached from autograd.")
    if projected_calibration_jacobian.dtype != torch.float64:
        raise ValueError("projected_calibration_jacobian must use torch.float64.")
    if not torch.isfinite(projected_calibration_jacobian).all():
        raise ValueError("projected_calibration_jacobian must contain only finite values.")

    fisher_information = projected_calibration_jacobian.transpose(-1, -2) @ projected_calibration_jacobian

    return stabilize_psd(fisher_information, numerics)


def compute_observability_matrix_single_window(linearization: WindowLinearization, numerics: ObservabilityNumericsConfig) -> WindowObservabilityMatrix:
    """
    Compute the canonical calibration Fisher information matrix for one window.

    The complete calculation is

        factor linearization
            -> covariance whitening
            -> stack [J_T, J_N, J_C]
            -> marginalize span([J_T, J_N])
            -> projected calibration Jacobian O_C
            -> F_C = O_C.T @ O_C.

    ``O_C`` is retained for rank and conditioning diagnostics because forming
    ``F_C`` squares its singular values and therefore worsens numerical dynamic
    range. The Fisher matrix remains the canonical physical matrix output.
    """

    if not isinstance(numerics, ObservabilityNumericsConfig):
        raise TypeError("numerics must be an ObservabilityNumericsConfig.")

    linearization.validate()
    whitened = whiten_window_linearization(linearization)
    projected_calibration_jacobian = project_calibration_jacobian(whitened, numerics)
    fisher_information_matrix = fisher_information_from_projected_jacobian(projected_calibration_jacobian, numerics)

    return WindowObservabilityMatrix(
        fisher_information_matrix=fisher_information_matrix,
        layout=linearization.calibration_layout,
        reference_timebase=linearization.reference_timebase,
        projected_calibration_jacobian=projected_calibration_jacobian,
    )


__all__ = [
    "compute_observability_matrix_single_window",
    "fisher_information_from_projected_jacobian",
    "project_calibration_jacobian",
]