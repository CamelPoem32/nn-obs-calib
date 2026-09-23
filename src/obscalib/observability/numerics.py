"""Shared numerical policy and stable linear-algebra primitives for observability."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ObservabilityNumericsConfig:
    """
    Configure rank, pseudoinverse, PSD, and diagnostic scaling behavior.

    ``absolute_tolerance`` and ``relative_tolerance`` control singular-value
    decisions. When omitted, the relative tolerance follows the usual
    dimension-scaled machine-epsilon rule.

    ``psd_tolerance`` controls how much negative eigenvalue magnitude is accepted
    as numerical roundoff when a matrix is expected to be positive semidefinite.

    ``parameter_scales`` is diagnostic-only. It may be used to construct
    dimensionless matrices for rank or conditioning analysis, but it must never
    silently replace the physical Fisher matrix used for covariance or CRLB.
    """

    absolute_tolerance: float | None = None
    relative_tolerance: float | None = None
    psd_tolerance: float | None = None
    parameter_scales: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        for name, value in (("absolute_tolerance", self.absolute_tolerance), ("relative_tolerance", self.relative_tolerance), ("psd_tolerance", self.psd_tolerance)):
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be nonnegative.")

        if self.parameter_scales is not None:
            if not self.parameter_scales:
                raise ValueError("parameter_scales must not be empty.")
            if any(scale <= 0.0 for scale in self.parameter_scales):
                raise ValueError("parameter_scales must be strictly positive.")


def _validate_float_matrix(matrix: torch.Tensor, *, name: str, square: bool = False) -> None:
    """
    Validate one finite floating-point matrix.

    Scientific observability code currently converts its inputs to CPU float64
    before reaching these helpers, but the routines remain dtype-aware so they
    can also be unit-tested independently.
    """

    if not isinstance(matrix, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if matrix.ndim != 2:
        raise ValueError(f"{name} must have shape [M, N].")
    if square and matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square.")
    if not torch.is_floating_point(matrix):
        raise TypeError(f"{name} must use a floating-point dtype.")
    if not torch.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values.")


def _resolved_relative_tolerance(matrix_shape: tuple[int, int], dtype: torch.dtype, config: ObservabilityNumericsConfig) -> float:
    """
    Resolve the relative singular-value tolerance.

    The default ``max(M, N) * eps`` matches the standard scale-aware numerical
    rank convention used by common dense linear-algebra implementations.
    """

    if config.relative_tolerance is not None:
        return float(config.relative_tolerance)

    return float(max(matrix_shape) * torch.finfo(dtype).eps)


def singular_value_tolerance(singular_values: torch.Tensor, matrix_shape: tuple[int, int], config: ObservabilityNumericsConfig | None = None) -> float:
    """
    Resolve the absolute singular-value threshold for one matrix spectrum.

    The threshold is the maximum of the configured absolute tolerance and the
    relative tolerance multiplied by the largest singular value.
    """

    if config is None:
        config = ObservabilityNumericsConfig()

    if singular_values.ndim != 1:
        raise ValueError("singular_values must have shape [K].")
    if not torch.is_floating_point(singular_values):
        raise TypeError("singular_values must use a floating-point dtype.")
    if not torch.isfinite(singular_values).all():
        raise ValueError("singular_values must contain only finite values.")

    absolute_tolerance = 0.0 if config.absolute_tolerance is None else float(config.absolute_tolerance)
    largest_singular_value = float(singular_values[0].item()) if singular_values.numel() else 0.0
    relative_tolerance = _resolved_relative_tolerance(matrix_shape, singular_values.dtype, config)

    return max(absolute_tolerance, relative_tolerance * largest_singular_value)


def matrix_singular_values(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute singular values of one finite matrix in descending order.

    Keeping this operation centralized ensures rank and conditioning diagnostics
    use exactly the same spectrum.
    """

    _validate_float_matrix(matrix, name="matrix")

    return torch.linalg.svdvals(matrix)


def numerical_rank(matrix: torch.Tensor, config: ObservabilityNumericsConfig | None = None) -> int:
    """
    Compute numerical matrix rank from an explicit singular-value threshold.

    This is intended for observability diagnostics and avoids scattering
    independent ``matrix_rank`` tolerance choices throughout the package.
    """

    singular_values = matrix_singular_values(matrix)
    tolerance = singular_value_tolerance(singular_values, tuple(matrix.shape), config)

    return int(torch.count_nonzero(singular_values > tolerance).item())


def pseudoinverse(matrix: torch.Tensor, config: ObservabilityNumericsConfig | None = None) -> torch.Tensor:
    """
    Compute an SVD pseudoinverse using the shared observability rank policy.

    Singular directions at or below the resolved threshold are assigned zero
    reciprocal gain.
    """

    _validate_float_matrix(matrix, name="matrix")

    if config is None:
        config = ObservabilityNumericsConfig()

    U, singular_values, Vh = torch.linalg.svd(matrix, full_matrices=False)
    tolerance = singular_value_tolerance(singular_values, tuple(matrix.shape), config)
    inverse_singular_values = torch.where(singular_values > tolerance, singular_values.reciprocal(), torch.zeros_like(singular_values))

    return (Vh.transpose(-1, -2) * inverse_singular_values.unsqueeze(0)) @ U.transpose(-1, -2)


def symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    """
    Return the symmetric part of a square matrix.

    Fisher and covariance matrices are theoretically symmetric; this helper
    removes only floating-point antisymmetry before eigenvalue-based diagnostics.
    """

    _validate_float_matrix(matrix, name="matrix", square=True)

    return 0.5 * (matrix + matrix.transpose(-1, -2))


def psd_tolerance(eigenvalues: torch.Tensor, config: ObservabilityNumericsConfig | None = None) -> float:
    """
    Resolve the allowed negative-eigenvalue magnitude for a PSD matrix.

    The default combines matrix dimension, machine precision, and spectral
    magnitude. An explicit ``psd_tolerance`` can only make this allowance larger.
    """

    if config is None:
        config = ObservabilityNumericsConfig()

    if eigenvalues.ndim != 1:
        raise ValueError("eigenvalues must have shape [N].")
    if not torch.is_floating_point(eigenvalues):
        raise TypeError("eigenvalues must use a floating-point dtype.")
    if not torch.isfinite(eigenvalues).all():
        raise ValueError("eigenvalues must contain only finite values.")

    spectral_scale = float(torch.max(torch.abs(eigenvalues)).item()) if eigenvalues.numel() else 0.0
    default_tolerance = float(max(1, eigenvalues.numel()) * torch.finfo(eigenvalues.dtype).eps * max(1.0, spectral_scale))

    if config.psd_tolerance is None:
        return default_tolerance

    return max(default_tolerance, float(config.psd_tolerance))


def stabilize_psd(matrix: torch.Tensor, config: ObservabilityNumericsConfig | None = None) -> torch.Tensor:
    """
    Symmetrize a numerically PSD matrix and clip roundoff-scale negative eigenvalues.

    A materially negative eigenvalue raises instead of being silently repaired.
    This prevents an invalid information matrix from masquerading as a valid
    Fisher matrix.
    """

    symmetric = symmetrize(matrix)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    tolerance = psd_tolerance(eigenvalues, config)

    if eigenvalues.numel() and float(eigenvalues[0].item()) < -tolerance:
        raise ValueError(f"Matrix is not positive semidefinite within tolerance: minimum eigenvalue={float(eigenvalues[0].item()):.6e}, tolerance={tolerance:.6e}.")

    clipped_eigenvalues = torch.clamp(eigenvalues, min=0.0)

    return (eigenvectors * clipped_eigenvalues.unsqueeze(0)) @ eigenvectors.transpose(-1, -2)


def psd_pseudoinverse(matrix: torch.Tensor, config: ObservabilityNumericsConfig | None = None) -> torch.Tensor:
    """
    Compute the pseudoinverse of a symmetric positive-semidefinite matrix.

    This is suitable for observable-subspace covariance diagnostics. A
    rank-deficient Fisher matrix still has unobservable directions; zero
    pseudoinverse gain in those directions must not be interpreted as finite
    physical uncertainty.
    """

    stabilized = stabilize_psd(matrix, config)
    eigenvalues, eigenvectors = torch.linalg.eigh(stabilized)
    singular_values_descending = torch.flip(eigenvalues, dims=(0,))
    tolerance = singular_value_tolerance(singular_values_descending, tuple(stabilized.shape), config)
    inverse_eigenvalues = torch.where(eigenvalues > tolerance, eigenvalues.reciprocal(), torch.zeros_like(eigenvalues))

    return (eigenvectors * inverse_eigenvalues.unsqueeze(0)) @ eigenvectors.transpose(-1, -2)


def validated_parameter_scales(dimension: int, config: ObservabilityNumericsConfig | None = None) -> torch.Tensor | None:
    """
    Return diagnostic parameter scales as a CPU float64 vector.

    ``None`` means no dimensionless diagnostic scaling is requested. The
    physical Fisher matrix must remain unscaled regardless of this setting.
    """

    if dimension < 0:
        raise ValueError("dimension must be nonnegative.")

    if config is None or config.parameter_scales is None:
        return None

    if len(config.parameter_scales) != dimension:
        raise ValueError(f"parameter_scales has length {len(config.parameter_scales)}, expected {dimension}.")

    return torch.tensor(config.parameter_scales, dtype=torch.float64)


__all__ = [
    "ObservabilityNumericsConfig",
    "matrix_singular_values",
    "numerical_rank",
    "psd_pseudoinverse",
    "psd_tolerance",
    "pseudoinverse",
    "singular_value_tolerance",
    "stabilize_psd",
    "symmetrize",
    "validated_parameter_scales",
]