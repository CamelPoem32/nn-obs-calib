"""Whitening of one-window observability linearizations with per-stream Cholesky reuse.

The key optimization is structural: the current linearization API assigns one
constant residual covariance to a stream. Therefore Cholesky factorization is
performed once per stream per window, and all factors from that stream are
whitened in one batched triangular solve.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch

from obscalib.observability.linearization import SensorFactorLinearization, WindowLinearization


@dataclass(frozen=True)
class WhitenedFactorLinearization:
    """One factor after covariance whitening."""

    residual: torch.Tensor
    trajectory_jacobian: torch.Tensor
    nuisance_jacobian: torch.Tensor
    calibration_jacobian: torch.Tensor


@dataclass(frozen=True)
class WhitenedWindowLinearization:
    """All whitened rows stacked in original factor order."""

    residual: torch.Tensor
    trajectory_jacobian: torch.Tensor
    nuisance_jacobian: torch.Tensor
    calibration_jacobian: torch.Tensor


def _validate_covariance_once(covariance: torch.Tensor, residual_dimension: int, stream_key: str) -> torch.Tensor:
    """Validate and symmetrize one stream covariance before its single Cholesky factorization."""

    if covariance is None:
        raise ValueError(f"Factor stream {stream_key!r} has no residual covariance; Fisher construction must not assume identity noise.")
    if covariance.device.type != "cpu" or covariance.requires_grad or covariance.dtype != torch.float64:
        raise ValueError(f"Residual covariance for stream {stream_key!r} must be detached CPU float64.")
    if covariance.shape != (residual_dimension, residual_dimension):
        raise ValueError(f"Residual covariance for stream {stream_key!r} must have shape [{residual_dimension}, {residual_dimension}].")
    if not torch.isfinite(covariance).all():
        raise ValueError(f"Residual covariance for stream {stream_key!r} must contain finite values.")

    scale = max(float(torch.max(torch.abs(covariance)).item()), 1.0)
    asymmetry = float(torch.max(torch.abs(covariance - covariance.T)).item())
    tolerance = 100.0 * torch.finfo(covariance.dtype).eps * scale

    if asymmetry > tolerance:
        raise ValueError(f"Residual covariance for stream {stream_key!r} is not symmetric within numerical tolerance.")

    return 0.5 * (covariance + covariance.T)


def _cholesky_for_covariance(covariance: torch.Tensor, stream_key: str) -> torch.Tensor:
    """Factor one validated SPD covariance."""

    cholesky, info = torch.linalg.cholesky_ex(covariance)

    if int(info.item()) != 0:
        raise ValueError(f"Residual covariance for stream {stream_key!r} must be positive definite.")

    return cholesky


def whiten_dense_blocks(residual: torch.Tensor, trajectory_jacobian: torch.Tensor, nuisance_jacobian: torch.Tensor, calibration_jacobian: torch.Tensor, covariance: torch.Tensor) -> WhitenedFactorLinearization:
    """Whiten one dense factor using its covariance.

    This compatibility entry point performs one Cholesky factorization. Window-level code should use ``whiten_window_linearization`` so the factorization is reused across every factor from the same stream.
    """

    covariance = _validate_covariance_once(covariance, residual.shape[0], "<dense>")
    cholesky = _cholesky_for_covariance(covariance, "<dense>")
    return whiten_dense_blocks_with_cholesky(residual, trajectory_jacobian, nuisance_jacobian, calibration_jacobian, cholesky)


def whiten_dense_blocks_with_cholesky(residual: torch.Tensor, trajectory_jacobian: torch.Tensor, nuisance_jacobian: torch.Tensor, calibration_jacobian: torch.Tensor, cholesky: torch.Tensor) -> WhitenedFactorLinearization:
    """Whiten one already-validated factor using a precomputed lower Cholesky factor."""

    dense = torch.cat((residual[:, None], trajectory_jacobian, nuisance_jacobian, calibration_jacobian), dim=1)
    whitened = torch.linalg.solve_triangular(cholesky, dense, upper=False)
    trajectory_dimension = trajectory_jacobian.shape[1]
    nuisance_dimension = nuisance_jacobian.shape[1]
    trajectory_start = 1
    nuisance_start = trajectory_start + trajectory_dimension
    calibration_start = nuisance_start + nuisance_dimension

    return WhitenedFactorLinearization(residual=whitened[:, 0], trajectory_jacobian=whitened[:, trajectory_start:nuisance_start], nuisance_jacobian=whitened[:, nuisance_start:calibration_start], calibration_jacobian=whitened[:, calibration_start:])


def whiten_factor_linearization(factor: SensorFactorLinearization, cholesky: torch.Tensor | None = None) -> WhitenedFactorLinearization:
    """Whiten one factor; optionally reuse a caller-supplied Cholesky factor."""

    nuisance_jacobian = factor.nuisance_jacobian if factor.nuisance_jacobian is not None else torch.zeros((factor.residual.shape[0], 0), dtype=factor.residual.dtype, device=factor.residual.device)

    if cholesky is None:
        covariance = _validate_covariance_once(factor.residual_covariance, factor.residual.shape[0], factor.stream_key)
        cholesky = _cholesky_for_covariance(covariance, factor.stream_key)

    return whiten_dense_blocks_with_cholesky(factor.residual, factor.trajectory_jacobian, nuisance_jacobian, factor.calibration_jacobian, cholesky)


def whiten_window_linearization(linearization: WindowLinearization) -> WhitenedWindowLinearization:
    """Whiten every factor, factorizing covariance once per stream and solving once per stream family."""

    if not linearization.factors:
        raise ValueError("A window linearization requires at least one factor.")

    factors = linearization.factors
    first = factors[0]
    trajectory_dimension = first.trajectory_jacobian.shape[1]
    nuisance_dimension = max((0 if factor.nuisance_jacobian is None else factor.nuisance_jacobian.shape[1]) for factor in factors)
    calibration_dimension = first.calibration_jacobian.shape[1]
    total_rows = sum(int(factor.residual.shape[0]) for factor in factors)

    residual_out = torch.empty(total_rows, dtype=torch.float64)
    trajectory_out = torch.empty((total_rows, trajectory_dimension), dtype=torch.float64)
    nuisance_out = torch.empty((total_rows, nuisance_dimension), dtype=torch.float64)
    calibration_out = torch.empty((total_rows, calibration_dimension), dtype=torch.float64)

    row_starts = []
    row_cursor = 0

    for factor in factors:
        residual_dimension = int(factor.residual.shape[0])
        row_starts.append(row_cursor)
        row_cursor += residual_dimension

    factor_indices_by_stream = defaultdict(list)

    for factor_index, factor in enumerate(factors):
        factor_indices_by_stream[factor.stream_key].append(factor_index)

    for stream_key, factor_indices in factor_indices_by_stream.items():
        representative = factors[factor_indices[0]]
        residual_dimension = int(representative.residual.shape[0])
        covariance = _validate_covariance_once(representative.residual_covariance, residual_dimension, stream_key)
        cholesky = _cholesky_for_covariance(covariance, stream_key)

        for factor_index in factor_indices[1:]:
            factor = factors[factor_index]
            if factor.residual.shape[0] != residual_dimension:
                raise ValueError(f"All factors from stream {stream_key!r} must share residual dimension when one stream covariance is reused.")
            if factor.trajectory_jacobian.shape[1] != trajectory_dimension or factor.calibration_jacobian.shape[1] != calibration_dimension:
                raise ValueError("All factor Jacobians must share global trajectory and calibration widths.")
            current_nuisance_dimension = 0 if factor.nuisance_jacobian is None else factor.nuisance_jacobian.shape[1]
            if current_nuisance_dimension != nuisance_dimension:
                raise ValueError("All factor Jacobians must share global nuisance width.")

        dense_width = 1 + trajectory_dimension + nuisance_dimension + calibration_dimension
        dense_batch = torch.empty((len(factor_indices), residual_dimension, dense_width), dtype=torch.float64)

        for batch_index, factor_index in enumerate(factor_indices):
            factor = factors[factor_index]
            nuisance_jacobian = factor.nuisance_jacobian if factor.nuisance_jacobian is not None else torch.zeros((residual_dimension, nuisance_dimension), dtype=torch.float64)
            dense_batch[batch_index] = torch.cat((factor.residual[:, None], factor.trajectory_jacobian, nuisance_jacobian, factor.calibration_jacobian), dim=1)

        cholesky_batch = cholesky.unsqueeze(0).expand(len(factor_indices), -1, -1)
        whitened_batch = torch.linalg.solve_triangular(cholesky_batch, dense_batch, upper=False)
        trajectory_start = 1
        nuisance_start = trajectory_start + trajectory_dimension
        calibration_start = nuisance_start + nuisance_dimension

        for batch_index, factor_index in enumerate(factor_indices):
            row_start = row_starts[factor_index]
            row_stop = row_start + residual_dimension
            whitened = whitened_batch[batch_index]
            residual_out[row_start:row_stop] = whitened[:, 0]
            trajectory_out[row_start:row_stop] = whitened[:, trajectory_start:nuisance_start]
            nuisance_out[row_start:row_stop] = whitened[:, nuisance_start:calibration_start]
            calibration_out[row_start:row_stop] = whitened[:, calibration_start:]

    return WhitenedWindowLinearization(residual=residual_out, trajectory_jacobian=trajectory_out, nuisance_jacobian=nuisance_out, calibration_jacobian=calibration_out)


__all__ = ["WhitenedFactorLinearization", "WhitenedWindowLinearization", "whiten_dense_blocks", "whiten_dense_blocks_with_cholesky", "whiten_factor_linearization", "whiten_window_linearization"]
