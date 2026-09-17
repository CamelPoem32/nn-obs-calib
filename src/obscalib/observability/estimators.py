"""Research boundary for raw-window observability-matrix computation."""

from abc import ABC, abstractmethod
from collections.abc import Mapping

from torch import nn

from obscalib.calibration import CalibrationState
from obscalib.data.structures import SensorStreamBatch
from obscalib.observability.structures import ObservabilityResult


class ObservabilityEstimator(nn.Module, ABC):
    """Common interface for future raw-window observability computation."""

    @abstractmethod
    def forward(self, measurements: Mapping[str, SensorStreamBatch], calibration: Mapping[str, CalibrationState]) -> ObservabilityResult:
        """
        Compute one scientific observability result per window.

        Measurements are deliberately consumed before geometry preprocessing so
        future NumPy/Numba implementations can operate directly on raw sensor
        streams and calibration priors.
        """


class ObservabilityMatrixEstimator(ObservabilityEstimator):
    """
    Future observability/information-matrix estimator.

    The intended output is the complete matrix for every window. Feature
    representations such as flattening, soft rank, CRLB-like uncertainties and
    log condition number belong to ObservabilityMapper implementations.
    """

    def forward(self, measurements: Mapping[str, SensorStreamBatch], calibration: Mapping[str, CalibrationState]) -> ObservabilityResult:
        # TODO: Port and benchmark the validated observability-matrix pipeline.
        # TODO: Compare plain NumPy with a Numba-compiled implementation.
        # TODO: Keep this branch outside autograd; only the resulting small matrix
        # or feature vector needs to be transferred to the learned model device.
        raise NotImplementedError("Observability-matrix computation is not implemented.")