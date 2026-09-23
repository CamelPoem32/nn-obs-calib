"""Reference tests for calibration observability metrics."""

from __future__ import annotations

import torch

from obscalib.observability.layout import CalibrationParameterLayout
from obscalib.observability.metrics import compute_condition_number, compute_crlb, compute_crlb_matrix, compute_crlb_standard_deviations, compute_numerical_rank, compute_singular_values
from obscalib.observability.numerics import ObservabilityNumericsConfig
from obscalib.observability.structures import WindowObservabilityMatrix
from obscalib.observability.timebase import ReferenceTimebase


DTYPE = torch.float64
ATOL = 1e-11
RTOL = 1e-10


def _make_result(singular_values: torch.Tensor) -> tuple[WindowObservabilityMatrix, torch.Tensor]:
    """
    Build a synthetic result with a known nontrivial right-singular basis.
    """

    dimension = singular_values.numel()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    Q, R = torch.linalg.qr(torch.randn((dimension, dimension), dtype=DTYPE, generator=generator))
    signs = torch.where(torch.sign(torch.diag(R)) == 0.0, torch.ones(dimension, dtype=DTYPE), torch.sign(torch.diag(R)))
    V = Q * signs.unsqueeze(0)

    projected = torch.diag(singular_values) @ V.transpose(-1, -2)
    fisher = projected.transpose(-1, -2) @ projected

    result = WindowObservabilityMatrix(
        fisher_information_matrix=fisher,
        layout=CalibrationParameterLayout.from_calibration_keys(("sensor",)),
        reference_timebase=ReferenceTimebase(stream_key="lidar", timestamps=torch.tensor([0.0, 0.1], dtype=DTYPE)),
        projected_calibration_jacobian=projected,
    )

    return result, V


def test_every_metric_matches_full_rank_reference() -> None:
    """
    Call every public metric and compare it against its direct reference formula.
    """

    sigma = torch.tensor([8.0, 5.0, 3.0, 2.0, 1.5, 0.75, 0.25], dtype=DTYPE)
    result, V = _make_result(sigma)
    numerics = ObservabilityNumericsConfig(relative_tolerance=1e-12)

    singular_values = compute_singular_values(result, numerics)
    rank = compute_numerical_rank(result, numerics)
    condition_number = compute_condition_number(result, numerics)
    principal_crlb = compute_crlb(result, numerics)
    principal_std = compute_crlb_standard_deviations(result, numerics)
    full_crlb = compute_crlb_matrix(result, numerics)

    expected_singular_values = torch.linalg.svdvals(result.projected_calibration_jacobian)
    expected_principal_crlb = expected_singular_values.pow(-2)
    expected_principal_std = expected_singular_values.reciprocal()
    expected_full_crlb = torch.linalg.inv(result.fisher_information_matrix)
    expected_full_crlb_from_principal_axes = V @ torch.diag(sigma.pow(-2)) @ V.transpose(-1, -2)

    torch.testing.assert_close(singular_values, expected_singular_values, atol=ATOL, rtol=RTOL)
    assert rank == 7
    torch.testing.assert_close(condition_number, expected_singular_values[0] / expected_singular_values[-1], atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(principal_crlb, expected_principal_crlb, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(principal_std, expected_principal_std, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(full_crlb, expected_full_crlb, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(full_crlb, expected_full_crlb_from_principal_axes, atol=ATOL, rtol=RTOL)


def test_rank_deficiency_gives_infinite_principal_crlb_but_pseudoinverse_matrix() -> None:
    """
    Verify the intentional distinction between compact and full CRLB outputs.
    """

    sigma = torch.tensor([7.0, 4.0, 2.0, 1.0, 0.5, 0.1, 0.0], dtype=DTYPE)
    result, _ = _make_result(sigma)
    numerics = ObservabilityNumericsConfig(relative_tolerance=1e-10)

    principal_crlb = compute_crlb(result, numerics)
    full_crlb = compute_crlb_matrix(result, numerics)

    assert compute_numerical_rank(result, numerics) == 6
    assert torch.isinf(compute_condition_number(result, numerics))
    assert torch.isinf(principal_crlb[-1])
    torch.testing.assert_close(principal_crlb[:-1], torch.linalg.svdvals(result.projected_calibration_jacobian)[:-1].pow(-2), atol=1e-10, rtol=1e-9)

    expected_pseudoinverse = torch.linalg.pinv(result.fisher_information_matrix, rtol=1e-10, atol=0.0, hermitian=True)
    torch.testing.assert_close(full_crlb, expected_pseudoinverse, atol=1e-10, rtol=1e-9)


def test_short_projected_jacobian_pads_missing_principal_directions() -> None:
    """
    Verify that missing singular directions are represented explicitly.
    """

    layout = CalibrationParameterLayout.from_calibration_keys(("sensor",))
    projected = torch.zeros((3, layout.total_dimension), dtype=DTYPE)
    projected[:, :3] = torch.diag(torch.tensor([3.0, 2.0, 1.0], dtype=DTYPE))
    fisher = projected.transpose(-1, -2) @ projected
    result = WindowObservabilityMatrix(
        fisher_information_matrix=fisher,
        layout=layout,
        reference_timebase=ReferenceTimebase(stream_key="lidar", timestamps=torch.tensor([0.0, 0.1], dtype=DTYPE)),
        projected_calibration_jacobian=projected,
    )
    numerics = ObservabilityNumericsConfig(relative_tolerance=1e-12)

    singular_values = compute_singular_values(result, numerics)
    principal_crlb = compute_crlb(result, numerics)

    assert singular_values.shape == (7,)
    torch.testing.assert_close(singular_values[:3], torch.tensor([3.0, 2.0, 1.0], dtype=DTYPE))
    torch.testing.assert_close(singular_values[3:], torch.zeros(4, dtype=DTYPE))
    assert torch.all(torch.isinf(principal_crlb[3:]))


def test_parameter_scaling_affects_principal_diagnostics_not_full_physical_crlb() -> None:
    """
    Verify the diagnostic-only scaling contract.
    """

    sigma = torch.tensor([6.0, 4.5, 3.0, 2.0, 1.2, 0.7, 0.3], dtype=DTYPE)
    result, _ = _make_result(sigma)
    scales = (1.0, 2.0, 0.5, 4.0, 1.5, 0.25, 3.0)
    unscaled_numerics = ObservabilityNumericsConfig(relative_tolerance=1e-12)
    scaled_numerics = ObservabilityNumericsConfig(relative_tolerance=1e-12, parameter_scales=scales)

    principal_unscaled = compute_crlb(result, unscaled_numerics)
    principal_scaled = compute_crlb(result, scaled_numerics)
    full_unscaled = compute_crlb_matrix(result, unscaled_numerics)
    full_scaled_config = compute_crlb_matrix(result, scaled_numerics)

    assert not torch.allclose(principal_unscaled, principal_scaled)
    torch.testing.assert_close(full_unscaled, full_scaled_config, atol=ATOL, rtol=RTOL)