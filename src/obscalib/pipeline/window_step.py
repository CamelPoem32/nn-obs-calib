"""Execution of one complete temporal calibration window."""

from __future__ import annotations

from dataclasses import dataclass

from obscalib.calibration.context import CALIBRATION_CONTEXT_DIM, calibration_state_to_context
from obscalib.calibration.state import CalibrationState
from obscalib.calibration.update import CalibrationUpdater
from obscalib.data.structures import TokenBatch, WindowBatch
from obscalib.geometry.processing import GeometryProcessor
from obscalib.models.obs_calib_model import ObsCalibModel
from obscalib.models.structures import ModelOutput
from obscalib.observability.estimators import ObservabilityEstimator
from obscalib.observability.mappings import ObservabilityMapper
from obscalib.observability.structures import ObservabilityResult
from obscalib.tokenization.tokenizer import Tokenizer

import torch

@dataclass
class WindowStepResult:
    """
    Result of processing one complete temporal window.

    model_output:
        Raw deterministic predictions produced by all configured calibration
        heads.

    next_calibration:
        Calibration states after applying the predicted left-multiplicative
        spatial corrections and additive temporal corrections.

    tokens:
        Final chronologically sorted tokens supplied to the Transformer.

    observability:
        Optional scientific and mapped observability result used for this
        window. None means the current experiment does not use observability.
    """

    model_output: ModelOutput
    next_calibration: dict[str, CalibrationState]
    tokens: TokenBatch
    observability: ObservabilityResult | None = None


class WindowStep:
    """
    Coordinate all deterministic and learned operations for one temporal window.

    The processing order is

        raw window
            |
            +-> optional observability computation from raw streams
            |
            -> GeometryProcessor
                 spatial calibration prior
                 Log().vee() / vector mapping
                 additive timestamp correction
            -> Tokenizer
                 zero-pad canonical vectors
                 append timestamp / type / optional observability
                 concatenate streams
                 chronological sort
            -> calibration contexts [phi, rho, tau]
            -> ObsCalibModel
            -> CalibrationUpdater.

    During teacher-forced training, calibration=None uses
    window.current_calibration.

    During sequential rollout, an explicitly supplied calibration dictionary
    overrides window.current_calibration.
    """

    def __init__(
        self,
        geometry_processor: GeometryProcessor,
        tokenizer: Tokenizer,
        model: ObsCalibModel,
        calibration_updater: CalibrationUpdater,
        observability_estimator: ObservabilityEstimator | None = None,
        observability_mapper: ObservabilityMapper | None = None,
    ) -> None:
        if (observability_estimator is None) != (observability_mapper is None):
            raise ValueError("observability_estimator and observability_mapper must either both be provided or both be None.")

        if model.config.calibration_head.calibration_context_dim != CALIBRATION_CONTEXT_DIM:
            raise ValueError(f"Calibration heads must use calibration_context_dim={CALIBRATION_CONTEXT_DIM} for context [phi, rho, tau].")

        self.geometry_processor = geometry_processor
        self.tokenizer = tokenizer
        self.model = model
        self.calibration_updater = calibration_updater
        self.observability_estimator = observability_estimator
        self.observability_mapper = observability_mapper

    def _resolve_calibration(self, window: WindowBatch, calibration: dict[str, CalibrationState] | None) -> dict[str, CalibrationState]:
        """Select teacher-forced or externally carried calibration states."""

        current_calibration = dict(window.current_calibration if calibration is None else calibration)

        if not current_calibration:
            raise ValueError("At least one calibration state is required.")

        for calibration_key, state in current_calibration.items():
            try:
                state.validate()
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid calibration state {calibration_key!r}.") from error

        return current_calibration

    def _compute_observability(self, window: WindowBatch, calibration: dict[str, CalibrationState]) -> ObservabilityResult | None:
        """Compute and map optional observability features from the raw window."""

        if self.observability_estimator is None:
            return None

        if self.observability_mapper is None:
            raise RuntimeError("Observability mapper is missing although an estimator is configured.")

        scientific_result = self.observability_estimator(window.streams, calibration)

        return self.observability_mapper(scientific_result)

    def _build_calibration_context(self, calibration: dict[str, CalibrationState]) -> dict[str, torch.Tensor]:
        """Build one [B, 7] calibration context for every learned head."""

        head_keys = tuple(self.model.heads.keys())
        missing_calibration = set(head_keys) - set(calibration)

        if missing_calibration:
            raise KeyError(f"Missing calibration states for model heads: {sorted(missing_calibration)}")

        return {head_key: calibration_state_to_context(calibration[head_key]) for head_key in head_keys}

    def _update_calibration(self, calibration: dict[str, CalibrationState], model_output: ModelOutput) -> dict[str, CalibrationState]:
        """
        Apply predictions while carrying states without corresponding heads unchanged.

        This permits fixed/reference sensor calibrations to remain in the
        calibration dictionary without requiring a learned output head.
        """

        next_calibration = dict(calibration)

        for calibration_key, prediction in model_output.predictions.items():
            if calibration_key not in calibration:
                raise KeyError(f"Model produced prediction for unknown calibration key {calibration_key!r}.")

            next_calibration[calibration_key] = self.calibration_updater.update(calibration[calibration_key], prediction)

        return next_calibration

    def __call__(self, window: WindowBatch, calibration: dict[str, CalibrationState] | None = None) -> WindowStepResult:
        """Process one temporal window from raw streams to updated calibration."""

        current_calibration = self._resolve_calibration(window, calibration)

        # Observability deliberately branches from the raw window before learned
        # geometry/token preprocessing. This keeps future NumPy/Numba estimators
        # independent from the neural computation graph.
        observability = self._compute_observability(window, current_calibration)

        # Apply current spatial and temporal calibration priors, then convert all
        # measurements to canonical vector representations.
        canonical_streams = self.geometry_processor(window.streams, window.metadata, current_calibration)

        # Build complete measurement tokens and chronologically sort them using
        # calibration-corrected timestamps.
        tokens = self.tokenizer(canonical_streams, observability)

        # Each output head receives the current calibration in the fixed
        # [phi, rho, tau] representation.
        calibration_context = self._build_calibration_context(current_calibration)

        model_output = self.model(tokens, calibration_context)
        next_calibration = self._update_calibration(current_calibration, model_output)

        return WindowStepResult(model_output=model_output, next_calibration=next_calibration, tokens=tokens, observability=observability)