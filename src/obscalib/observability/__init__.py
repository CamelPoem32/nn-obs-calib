"""Injectable observability estimation, matrix diagnostics, and feature mappings."""

from obscalib.observability.estimators import ObservabilityEstimator, ObservabilityMatrixEstimator
from obscalib.observability.mappings import CRLBTanhObservabilityMapper, CombinedObservabilityMapper, FlattenObservabilityMapper, LogConditionObservabilityMapper, ObservabilityMapper, SoftRankObservabilityMapper
from obscalib.observability.linearization import linearize_gyroscope_factors_single_window, linearize_gyroscope_stream, linearize_lidar_stream, linearize_single_window
from obscalib.observability.whitening import WhitenedFactorLinearization, WhitenedWindowLinearization, whiten_factor_linearization, whiten_window_linearization
from obscalib.observability.metrics import compute_condition_number, compute_crlb, compute_crlb_matrix, compute_crlb_standard_deviations, compute_numerical_rank, compute_singular_values
from obscalib.observability.structures import ObservabilityResult, WindowObservabilityMatrix

__all__ = [
    "CRLBTanhObservabilityMapper",
    "CombinedObservabilityMapper",
    "FlattenObservabilityMapper",
    "LogConditionObservabilityMapper",
    "ObservabilityEstimator",
    "ObservabilityMapper",
    "ObservabilityMatrixEstimator",
    "ObservabilityResult",
    "SoftRankObservabilityMapper",
    "WindowObservabilityMatrix",
    "WhitenedFactorLinearization",
    "WhitenedWindowLinearization",
    "compute_condition_number",
    "compute_crlb",
    "compute_crlb_matrix",
    "compute_crlb_standard_deviations",
    "compute_numerical_rank",
    "compute_singular_values",
    "linearize_gyroscope_factors_single_window",
    "linearize_gyroscope_stream",
    "linearize_lidar_stream",
    "linearize_single_window",
    "whiten_factor_linearization",
    "whiten_window_linearization",
]
