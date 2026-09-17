"""Interfaces for mapping scientific observability results to NN features."""

from abc import ABC, abstractmethod

from torch import nn

from obscalib.observability.structures import ObservabilityResult


class ObservabilityMapper(nn.Module, ABC):
    """Convert estimator-specific raw quantities to standardized features.

    Estimators may preserve matrices, ranks, or uncertainty structures in raw.
    Mappers produce optional features with shape [B, N, d_obs] for tokenization.
    """

    @abstractmethod
    def forward(
        self,
        result: ObservabilityResult,
    ) -> ObservabilityResult:
        """Return the result with NN-ready features while preserving raw data."""
