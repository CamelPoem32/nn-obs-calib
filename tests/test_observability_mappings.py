"""Tests for observability-to-network feature mappings."""

from __future__ import annotations

import torch

from obscalib.observability.layout import CalibrationParameterLayout
from obscalib.observability.mappings import CRLBTanhObservabilityMapper, CombinedObservabilityMapper, FlattenObservabilityMapper, LogConditionObservabilityMapper, SoftRankObservabilityMapper
from obscalib.observability.metrics import compute_crlb, compute_crlb_matrix, compute_singular_values
from obscalib.observability.numerics import ObservabilityNumericsConfig
from obscalib.observability.structures import ObservabilityResult, WindowObservabilityMatrix
from obscalib.observability.timebase import ReferenceTimebase


DTYPE = torch.float64


def _make_observability_result(sigma: torch.Tensor) -> ObservabilityResult:
    """
    Build one diagonal synthetic observability result for mapping tests.
    """

    projected = torch.diag(sigma)
    fisher = projected.transpose(-1, -2) @ projected
    raw = WindowObservabilityMatrix(
        fisher_information_matrix=fisher,
        layout=CalibrationParameterLayout.from_calibration_keys(("sensor",)),
        reference_timebase=ReferenceTimebase(stream_key="lidar", timestamps=torch.tensor([0.0, 0.1], dtype=DTYPE)),
        projected_calibration_jacobian=projected,
    )

    return ObservabilityResult(raw=raw)


def test_flatten_mapper_uses_full_crlb_matrix() -> None:
    """
    Verify that the full mapper flattens ``F_C^+`` rather than the Fisher matrix.
    """

    result = _make_observability_result(torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.25], dtype=DTYPE))
    numerics = ObservabilityNumericsConfig(relative_tolerance=1e-12)
    mapped = FlattenObservabilityMapper(numerics)(result)

    expected = compute_crlb_matrix(result.raw, numerics).reshape(1, -1)

    torch.testing.assert_close(mapped.features, expected)
    assert mapped.raw is result.raw
    assert mapped.features.shape == (1, 49)


def test_crlb_tanh_mapper_uses_principal_crlb_values() -> None:
    """
    Verify the default compact uncertainty mapping.
    """

    result = _make_observability_result(torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.25], dtype=DTYPE))
    numerics = ObservabilityNumericsConfig(relative_tolerance=1e-12)
    alpha = 2.5
    mapped = CRLBTanhObservabilityMapper(alpha=alpha, numerics=numerics)(result)

    expected_crlb = compute_crlb(result.raw, numerics)
    expected = (1.0 - torch.tanh(torch.sqrt(expected_crlb) / alpha)).reshape(1, -1)

    torch.testing.assert_close(mapped.features, expected)
    assert mapped.features.shape == (1, 7)


def test_crlb_tanh_maps_unobservable_direction_to_zero() -> None:
    """
    Verify that infinite principal CRLB becomes zero bounded observability.
    """

    result = _make_observability_result(torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.0], dtype=DTYPE))
    mapped = CRLBTanhObservabilityMapper(alpha=1.0, numerics=ObservabilityNumericsConfig(relative_tolerance=1e-10))(result)

    assert mapped.features[0, -1] == 0.0
    assert torch.isfinite(mapped.features).all()


def test_soft_rank_mapper_matches_reference_formula() -> None:
    """
    Verify the smooth-rank equation exactly.
    """

    result = _make_observability_result(torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.25], dtype=DTYPE))
    numerics = ObservabilityNumericsConfig(relative_tolerance=1e-12)
    threshold = 1.5
    temperature = 0.2
    mapped = SoftRankObservabilityMapper(threshold=threshold, temperature=temperature, numerics=numerics)(result)

    sigma = compute_singular_values(result.raw, numerics)
    expected = torch.sigmoid((sigma - threshold) / temperature).sum().reshape(1, 1)

    torch.testing.assert_close(mapped.features, expected)


def test_log_condition_mapper_is_finite_for_rank_deficiency() -> None:
    """
    Verify that an infinite raw condition number maps to a finite feature.
    """

    result = _make_observability_result(torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.0], dtype=DTYPE))
    mapped = LogConditionObservabilityMapper(ObservabilityNumericsConfig(relative_tolerance=1e-10))(result)

    torch.testing.assert_close(mapped.features, torch.ones((1, 1), dtype=DTYPE))


def test_combined_mapper_concatenates_in_configured_order() -> None:
    """
    Verify deterministic concatenation of heterogeneous observability features.
    """

    result = _make_observability_result(torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.25], dtype=DTYPE))
    numerics = ObservabilityNumericsConfig(relative_tolerance=1e-12)
    first = CRLBTanhObservabilityMapper(alpha=2.0, numerics=numerics)
    second = SoftRankObservabilityMapper(threshold=1.0, temperature=0.25, numerics=numerics)
    combined = CombinedObservabilityMapper((first, second))(result)

    expected = torch.cat((first(result).features, second(result).features), dim=-1)

    torch.testing.assert_close(combined.features, expected)
    assert combined.features.shape == (1, 8)
