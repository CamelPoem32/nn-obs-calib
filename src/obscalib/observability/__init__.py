"""Injectable observability estimation, matrix diagnostics, and feature mappings."""

from obscalib.observability.estimators import ObservabilityEstimator, ObservabilityMatrixEstimator
from obscalib.observability.mappings import CRLBTanhObservabilityMapper, CombinedObservabilityMapper, FlattenObservabilityMapper, LogConditionObservabilityMapper, ObservabilityMapper, SoftRankObservabilityMapper
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
    "compute_condition_number",
    "compute_crlb",
    "compute_crlb_matrix",
    "compute_crlb_standard_deviations",
    "compute_numerical_rank",
    "compute_singular_values",
]
