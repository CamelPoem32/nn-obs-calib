"""Single-window orchestration from raw sensor batches to updated calibration states."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from obscalib.calibration.context import calibration_state_to_context
from obscalib.calibration.state import CalibrationState
from obscalib.calibration.update import CalibrationUpdater
from obscalib.data.structures import TokenBatch, WindowBatch
from obscalib.geometry.processing import GeometryProcessor
from obscalib.models.obs_calib_model import ObsCalibModel
from obscalib.models.structures import ModelOutput
from obscalib.observability.estimators import ObservabilityEstimator
from obscalib.observability.mappings import ObservabilityMapper
from obscalib.observability.structures import BatchedObservabilityMatrix, ObservabilityResult, WindowObservabilityMatrix
from obscalib.tokenization.tokenizer import Tokenizer


@dataclass
class WindowStepResult:
    """
    Complete output of one model-processing step.

    ``next_calibration`` remains differentiable with respect to the current model
    prediction when the configured ``CalibrationUpdater`` is differentiable.
    """

    model_output: ModelOutput
    next_calibration: dict[str, CalibrationState]
    tokens: TokenBatch
    observability: ObservabilityResult | None


def _validate_window(window: WindowBatch) -> int:
    """
    Validate one padded minibatch and return its batch size.
    """

    if not isinstance(window, WindowBatch):
        raise TypeError("window must be a WindowBatch.")
    if not window.streams:
        raise ValueError("window.streams must not be empty.")
    if set(window.streams) != set(window.metadata):
        raise ValueError("window.streams and window.metadata must contain identical stream keys.")
    if not window.current_calibration:
        raise ValueError("window.current_calibration must not be empty.")

    batch_sizes: list[int] = []

    for stream in window.streams.values():
        stream.validate()
        batch_sizes.append(stream.values.shape[0])

    for state in window.current_calibration.values():
        state.validate()
        batch_sizes.append(state.transform.shape[0])

    if any(batch_size != batch_sizes[0] for batch_size in batch_sizes[1:]):
        raise ValueError("All window streams and calibration states must share batch size.")

    return batch_sizes[0]


def _validate_calibration_for_window(window: WindowBatch, calibration: Mapping[str, CalibrationState], batch_size: int) -> None:
    """
    Validate a teacher-forced or rollout-supplied calibration mapping.
    """

    required_keys = {stream_metadata.calibration_key for stream_metadata in window.metadata.values()}
    missing = sorted(required_keys - set(calibration))

    if missing:
        raise KeyError(f"Calibration mapping is missing keys required by sensor streams: {missing}.")

    for calibration_key, state in calibration.items():
        state.validate()

        if state.transform.shape[0] != batch_size:
            raise ValueError(f"Calibration state {calibration_key!r} must use batch size {batch_size}.")


def _map_batched_observability(raw_result: ObservabilityResult, mapper: ObservabilityMapper) -> ObservabilityResult:
    """
    Apply a single-window mapper to each element of a batched Fisher result.

    ``BatchedObservabilityMatrix`` stores the canonical Fisher matrices. The
    projected Jacobians are not retained in that batched container, so mappers
    reconstruct spectral diagnostics from each Fisher matrix when needed.
    """

    if not isinstance(raw_result.raw, BatchedObservabilityMatrix):
        return mapper(raw_result)

    batched = raw_result.raw
    feature_rows: list[torch.Tensor] = []

    for batch_index, reference_timebase in enumerate(batched.reference_timebases):
        single_raw = WindowObservabilityMatrix(
            fisher_information_matrix=batched.fisher_information_matrix[batch_index],
            layout=batched.layout,
            reference_timebase=reference_timebase,
            projected_calibration_jacobian=None,
        )
        mapped = mapper(ObservabilityResult(raw=single_raw, features=None))

        if mapped.features is None:
            raise ValueError("Observability mapper must populate features.")
        if mapped.features.ndim != 2 or mapped.features.shape[0] != 1:
            raise ValueError("Single-window observability mapper must return features with shape [1, d_observability].")

        feature_rows.append(mapped.features)

    return ObservabilityResult(raw=batched, features=torch.cat(feature_rows, dim=0))


class WindowStep:
    """
    Coordinate one complete temporal-window model step.

    Processing order is fixed:

        1. resolve teacher-forced or rollout calibration state,
        2. optionally compute observability from raw sensor streams,
        3. apply calibration priors and geometry preprocessing,
        4. tokenize and chronologically sort measurements,
        5. build per-head calibration contexts,
        6. run the learned model,
        7. update predicted calibration states.

    Observability is computed from raw streams before geometry preprocessing.
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

        self.geometry_processor = geometry_processor
        self.tokenizer = tokenizer
        self.model = model
        self.calibration_updater = calibration_updater
        self.observability_estimator = observability_estimator
        self.observability_mapper = observability_mapper

    def _compute_observability(self, window: WindowBatch, calibration: Mapping[str, CalibrationState]) -> ObservabilityResult | None:
        """
        Compute and map optional observability features for the complete batch.
        """

        if self.observability_estimator is None:
            return None

        raw_result = self.observability_estimator(window.streams, calibration, window.metadata)

        if not isinstance(raw_result, ObservabilityResult):
            raise TypeError("observability_estimator must return an ObservabilityResult.")

        return _map_batched_observability(raw_result, self.observability_mapper)

    def _build_calibration_context(self, calibration: Mapping[str, CalibrationState]) -> dict[str, torch.Tensor]:
        """
        Build ``[phi, rho, tau]`` contexts for every configured model head.
        """

        head_keys = tuple(self.model.heads.keys())
        missing = sorted(set(head_keys) - set(calibration))

        if missing:
            raise KeyError(f"Calibration mapping is missing states required by model heads: {missing}.")

        return {
            head_key: calibration_state_to_context(calibration[head_key])
            for head_key in head_keys
        }

    def _update_calibration(self, calibration: Mapping[str, CalibrationState], model_output: ModelOutput) -> dict[str, CalibrationState]:
        """
        Update predicted calibration keys and preserve all fixed extra states.
        """

        next_calibration = dict(calibration)

        for calibration_key, prediction in model_output.predictions.items():
            if calibration_key not in calibration:
                raise KeyError(f"Model predicted unknown calibration key {calibration_key!r}.")

            next_calibration[calibration_key] = self.calibration_updater.update(calibration[calibration_key], prediction)

        return next_calibration

    def __call__(self, window: WindowBatch, calibration: Mapping[str, CalibrationState] | None = None) -> WindowStepResult:
        """
        Process one temporal window.

        ``calibration=None`` uses ``window.current_calibration`` for teacher-forced
        training. Passing a calibration mapping explicitly enables rollout or
        inference with externally carried state.
        """

        batch_size = _validate_window(window)
        current_calibration = window.current_calibration if calibration is None else calibration
        _validate_calibration_for_window(window, current_calibration, batch_size)

        observability = self._compute_observability(window, current_calibration)

        canonical_streams = self.geometry_processor(window.streams, window.metadata, current_calibration)
        tokens = self.tokenizer(canonical_streams, observability)
        tokens.validate()

        calibration_context = self._build_calibration_context(current_calibration)
        model_output = self.model(tokens, calibration_context)
        next_calibration = self._update_calibration(current_calibration, model_output)

        return WindowStepResult(
            model_output=model_output,
            next_calibration=next_calibration,
            tokens=tokens,
            observability=observability,
        )


__all__ = [
    "WindowStep",
    "WindowStepResult",
]