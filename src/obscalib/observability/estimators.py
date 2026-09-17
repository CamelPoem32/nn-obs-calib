"""Research boundaries for interchangeable observability estimators."""

from abc import ABC, abstractmethod
from collections.abc import Mapping

from torch import nn

from obscalib.calibration.state import CalibrationState

from obscalib.data.structures import (
    CanonicalMeasurements,
    ObservabilityResult,
)


class ObservabilityEstimator(nn.Module, ABC):
    """Common injectable interface for future observability strategies."""

    @abstractmethod
    def forward(
        self,
        measurements: CanonicalMeasurements,
        calibration: Mapping[str, CalibrationState],
    ) -> ObservabilityResult:
        """Return a scientific result whose raw structure may be strategy-specific."""


class ObservabilityMatrixEstimator(ObservabilityEstimator):
    """Future observability/information-matrix scientific result estimator."""

    def forward(
        self,
        measurements: CanonicalMeasurements,
        calibration: Mapping[str, CalibrationState],
    ) -> ObservabilityResult:
        """Compute a matrix-derived raw result after validated code is ported."""

        # TODO: Port the existing information/observability matrix pipeline.
        raise NotImplementedError("Observability-matrix features are not implemented.")


class SoftRankObservabilityEstimator(ObservabilityEstimator):
    """Future differentiable or smoothly thresholded rank-like estimator."""

    def forward(
        self,
        measurements: CanonicalMeasurements,
        calibration: Mapping[str, CalibrationState],
    ) -> ObservabilityResult:
        """Compute a raw soft-rank result once its definition is specified."""

        # TODO: Define the singular-value scaling and smooth threshold.
        raise NotImplementedError("Soft-rank observability is not implemented.")


class CRLBTanhObservabilityEstimator(ObservabilityEstimator):
    """Future CRLB estimator; bounded feature mapping belongs to a mapper."""

    def __init__(self, alpha: float) -> None:
        super().__init__()
        if alpha <= 0.0:
            raise ValueError("alpha must be positive.")
        self.alpha = alpha

    def forward(
        self,
        measurements: CanonicalMeasurements,
        calibration: Mapping[str, CalibrationState],
    ) -> ObservabilityResult:
        """Compute CRLB features after the uncertainty pipeline is specified."""

        # TODO: Port the validated raw CRLB calculation. alpha is retained for
        # compatibility with this public placeholder; bounded transforms such
        # as 1 - tanh(sigma / alpha) belong in an ObservabilityMapper.
        raise NotImplementedError("CRLB-like observability is not implemented.")


class RawObservabilityEstimator(ObservabilityEstimator):
    """Future estimator preserving a raw quantity without feature mapping."""

    def forward(
        self,
        measurements: CanonicalMeasurements,
        calibration: Mapping[str, CalibrationState],
    ) -> ObservabilityResult:
        """Return a raw result after its scientific contract is specified."""

        # TODO: Define the raw feature shape and normalization.
        raise NotImplementedError("Raw observability features are not implemented.")
