"""Current observability-estimator interface tests.

The concrete ``ObservabilityMatrixEstimator`` in the repository is still a
scientific placeholder. Detailed batching/Fisher orchestration tests belong here
once that implementation is landed. Until then, these tests cover only the
stable injectable estimator contract and deliberately avoid testing code that
does not yet exist.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from obscalib.calibration.state import CalibrationState
from obscalib.observability.estimators import ObservabilityEstimator, ObservabilityMatrixEstimator
from obscalib.observability.structures import ObservabilityResult


class _ConcreteEstimator(ObservabilityEstimator):
    """Minimal concrete estimator used to verify the injectable module boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.last_measurements = None
        self.last_calibration = None

    def forward(
        self,
        measurements: Mapping,
        calibration: Mapping[str, CalibrationState],
        *args,
        **kwargs,
    ) -> ObservabilityResult:
        """Record the supplied objects and return a strategy-agnostic raw result."""

        self.last_measurements = measurements
        self.last_calibration = calibration

        return ObservabilityResult(raw={"status": "recorded"}, features=None)


def test_observability_estimator_is_abstract() -> None:
    """The common estimator interface must not be instantiated directly."""

    with pytest.raises(TypeError):
        ObservabilityEstimator()


def test_concrete_observability_estimator_uses_nn_module_call_boundary() -> None:
    """A concrete estimator can be injected and called like any other nn.Module."""

    estimator = _ConcreteEstimator()
    measurements = {"stream": object()}
    calibration = {}

    result = estimator(measurements, calibration)

    assert estimator.last_measurements is measurements
    assert estimator.last_calibration is calibration
    assert result.raw == {"status": "recorded"}
    assert result.features is None


def test_observability_matrix_estimator_is_still_an_explicit_placeholder() -> None:
    """
    Document the current repository state without pretending the matrix estimator exists.

    Replace this test with numerical batching/Fisher tests when
    ``ObservabilityMatrixEstimator`` receives its real implementation.
    """

    estimator = ObservabilityMatrixEstimator()

    # The current placeholder has historically used either the two-argument
    # estimator boundary or the newer metadata-aware three-argument boundary.
    # Both are implementation placeholders and must fail explicitly rather than
    # return fabricated observability.
    try:
        result = estimator({}, {})
    except TypeError:
        with pytest.raises(NotImplementedError):
            estimator({}, {}, {})
    else:
        raise AssertionError(f"Placeholder ObservabilityMatrixEstimator unexpectedly returned {result!r}.")
