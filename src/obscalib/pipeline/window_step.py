"""Dependency-injected boundary for processing one temporal window."""

from obscalib.calibration.update import CalibrationUpdater
from obscalib.geometry.maps import GeometryMap
from obscalib.models.obs_calib_model import ObsCalibModel
from obscalib.observability.estimators import ObservabilityEstimator
from obscalib.observability.mappings import ObservabilityMapper
from obscalib.tokenization.tokenizer import Tokenizer


class WindowStep:
    """Coordinate one future geometry-to-state-update pipeline step."""

    def __init__(
        self,
        geometry_map: GeometryMap,
        observability_estimator: ObservabilityEstimator,
        observability_mapper: ObservabilityMapper,
        tokenizer: Tokenizer,
        model: ObsCalibModel,
        calibration_updater: CalibrationUpdater,
    ) -> None:
        self.geometry_map = geometry_map
        self.observability_estimator = observability_estimator
        self.observability_mapper = observability_mapper
        self.tokenizer = tokenizer
        self.model = model
        self.calibration_updater = calibration_updater

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Process one window after heterogeneous merge semantics are defined."""

        # TODO: Map each stream, merge values, estimate and map observability,
        # prepare calibration contexts, invoke the model, and update each state.
        raise NotImplementedError("Complete window processing is not implemented.")
