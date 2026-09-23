"""Observability diagnostics derived from a completed one-window Fisher result."""

from __future__ import annotations

import torch

from obscalib.observability.numerics import ObservabilityNumericsConfig, matrix_singular_values, psd_pseudoinverse, singular_value_tolerance, stabilize_psd, validated_parameter_scales
from obscalib.observability.structures import WindowObservabilityMatrix


def _diagnostic_projected_jacobian(result: WindowObservabilityMatrix, numerics: ObservabilityNumericsConfig) -> torch.Tensor | None:
    """
    Return the projected calibration Jacobian with optional diagnostic scaling.

    ``parameter_scales`` multiply calibration columns only for spectral
    diagnostics. They never modify the physical Fisher matrix stored in
    ``WindowObservabilityMatrix``.
    """

    projected = result.projected_calibration_jacobian

    if projected is None:
        return None

    scales = validated_parameter_scales(result.layout.total_dimension, numerics)

    if scales is None:
        return projected

    return projected * scales.to(dtype=projected.dtype, device=projected.device)[None, :]


def _diagnostic_fisher_matrix(result: WindowObservabilityMatrix, numerics: ObservabilityNumericsConfig) -> torch.Tensor:
    """
    Return a Fisher matrix corresponding to optional diagnostic parameter scaling.

    If ``S = diag(parameter_scales)``, the returned diagnostic matrix is
    ``S F_C S``. The physical Fisher matrix stored in ``result`` is unchanged.
    """

    fisher_information = result.fisher_information_matrix
    scales = validated_parameter_scales(result.layout.total_dimension, numerics)

    if scales is None:
        return fisher_information

    scales = scales.to(dtype=fisher_information.dtype, device=fisher_information.device)

    return scales[:, None] * fisher_information * scales[None, :]


def _pad_singular_values(singular_values: torch.Tensor, dimension: int) -> torch.Tensor:
    """
    Pad an observability spectrum with exact zeros to the calibration dimension.

    A projected Jacobian with fewer residual rows than calibration columns has
    implicit zero singular values corresponding to unobservable directions.
    """

    if singular_values.numel() > dimension:
        raise ValueError("The singular-value count cannot exceed the calibration dimension.")

    if singular_values.numel() == dimension:
        return singular_values

    return torch.cat((singular_values, torch.zeros(dimension - singular_values.numel(), dtype=singular_values.dtype, device=singular_values.device)), dim=0)


def _observability_spectrum(result: WindowObservabilityMatrix, numerics: ObservabilityNumericsConfig) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Return the singular values of the projected calibration Jacobian.

    ``O_C`` is preferred because forming ``F_C = O_C.T @ O_C`` squares the
    singular values and worsens numerical dynamic range. If ``O_C`` is absent,
    the same spectrum is reconstructed from the eigenvalues of the diagnostic
    Fisher matrix.
    """

    calibration_dimension = result.layout.total_dimension
    projected = _diagnostic_projected_jacobian(result, numerics)

    if projected is not None:
        singular_values = matrix_singular_values(projected)
        singular_values = _pad_singular_values(singular_values, calibration_dimension)

        return singular_values, tuple(projected.shape)

    diagnostic_fisher = stabilize_psd(_diagnostic_fisher_matrix(result, numerics), numerics)
    eigenvalues = torch.linalg.eigvalsh(diagnostic_fisher)
    singular_values = torch.sqrt(torch.clamp(torch.flip(eigenvalues, dims=(0,)), min=0.0))

    return singular_values, tuple(diagnostic_fisher.shape)


def compute_singular_values(result: WindowObservabilityMatrix, numerics: ObservabilityNumericsConfig) -> torch.Tensor:
    """
    Return calibration-observability singular values in descending order.

    The returned vector always has length ``D_C``. Missing singular directions
    caused by a short projected Jacobian are represented by exact zeros.
    """

    singular_values, _ = _observability_spectrum(result, numerics)

    return singular_values


def compute_numerical_rank(result: WindowObservabilityMatrix, numerics: ObservabilityNumericsConfig) -> int:
    """
    Return the numerical calibration-observability rank.

    Rank is thresholded on singular values of ``O_C`` rather than eigenvalues of
    ``F_C`` so numerical dynamic range is not squared unnecessarily.
    """

    singular_values, matrix_shape = _observability_spectrum(result, numerics)
    tolerance = singular_value_tolerance(singular_values, matrix_shape, numerics)

    return int(torch.count_nonzero(singular_values > tolerance).item())


def compute_condition_number(result: WindowObservabilityMatrix, numerics: ObservabilityNumericsConfig) -> torch.Tensor:
    """
    Return the condition number of the projected calibration Jacobian.

    Rank-deficient observability returns positive infinity. The condition number
    is based on ``O_C`` rather than ``F_C`` so it is not artificially squared.
    """

    singular_values, matrix_shape = _observability_spectrum(result, numerics)

    if singular_values.numel() == 0:
        return torch.tensor(float("inf"), dtype=result.fisher_information_matrix.dtype)

    tolerance = singular_value_tolerance(singular_values, matrix_shape, numerics)
    rank = int(torch.count_nonzero(singular_values > tolerance).item())
    calibration_dimension = result.layout.total_dimension

    if rank < calibration_dimension:
        return torch.tensor(float("inf"), dtype=singular_values.dtype, device=singular_values.device)

    smallest = singular_values[calibration_dimension - 1]

    if smallest <= tolerance:
        return torch.tensor(float("inf"), dtype=singular_values.dtype, device=singular_values.device)

    return singular_values[0] / smallest


def compute_crlb(result: WindowObservabilityMatrix, numerics: ObservabilityNumericsConfig) -> torch.Tensor:
    """
    Return principal-direction CRLB variances ``1 / sigma_i**2``.

    The result has shape ``[D_C]`` and follows descending observability singular
    values. A numerically unobservable singular direction receives ``+inf``,
    which represents an unbounded uncertainty lower bound rather than the zero
    gain used by a matrix pseudoinverse.

    When diagnostic ``parameter_scales`` are configured, these values correspond
    to the scaled principal coordinates used by the spectral diagnostics.
    """

    singular_values, matrix_shape = _observability_spectrum(result, numerics)
    tolerance = singular_value_tolerance(singular_values, matrix_shape, numerics)
    observable = singular_values > tolerance
    crlb = torch.full_like(singular_values, float("inf"))

    crlb[observable] = singular_values[observable].pow(-2)

    return crlb


def compute_crlb_standard_deviations(result: WindowObservabilityMatrix, numerics: ObservabilityNumericsConfig) -> torch.Tensor:
    """
    Return principal-direction one-sigma lower bounds ``1 / sigma_i``.

    This is simply ``sqrt(compute_crlb(...))`` and preserves ``+inf`` for
    unobservable directions.
    """

    return torch.sqrt(compute_crlb(result, numerics))


def compute_crlb_matrix(result: WindowObservabilityMatrix, numerics: ObservabilityNumericsConfig) -> torch.Tensor:
    """
    Return the full physical-coordinate observable-subspace CRLB matrix.

    This is the Moore-Penrose pseudoinverse ``F_C^+``. For a full-rank Fisher
    matrix it equals ``F_C^{-1}``. For a rank-deficient Fisher matrix it is only
    the covariance on the observable subspace; nullspace directions remain
    unbounded even though the pseudoinverse has zero gain there.

    ``parameter_scales`` are deliberately ignored so this matrix always stays in
    the physical calibration coordinates defined by ``CalibrationParameterLayout``.
    """

    return psd_pseudoinverse(result.fisher_information_matrix, numerics)


__all__ = [
    "compute_condition_number",
    "compute_crlb",
    "compute_crlb_matrix",
    "compute_crlb_standard_deviations",
    "compute_numerical_rank",
    "compute_singular_values",
]