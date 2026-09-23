"""Integration tests for one-window model orchestration."""

from __future__ import annotations

import torch
from torch import nn

from obscalib.calibration.state import CalibrationState
from obscalib.data.structures import CanonicalSensorStreamBatch, GeometryType, MeasurementType, SensorMetadata, SensorStreamBatch, WindowBatch
from obscalib.models.structures import CalibrationPrediction, ModelOutput
from obscalib.observability.estimators import ObservabilityEstimator
from obscalib.observability.layout import CalibrationParameterLayout
from obscalib.observability.mappings import CRLBTanhObservabilityMapper
from obscalib.observability.numerics import ObservabilityNumericsConfig
from obscalib.observability.structures import BatchedObservabilityMatrix, ObservabilityResult
from obscalib.observability.timebase import ReferenceTimebase
from obscalib.pipeline.window_step import WindowStep
from obscalib.tokenization.tokenizer import Tokenizer


DTYPE = torch.float64


def _state(batch_size: int, tau: float) -> CalibrationState:
    """Create one identity transform with a constant time offset."""

    return CalibrationState(
        transform=torch.eye(4, dtype=DTYPE).repeat(batch_size, 1, 1),
        time_offset=torch.full((batch_size, 1), tau, dtype=DTYPE),
    )


def _window(batch_size: int = 2) -> WindowBatch:
    """Build a simple gyroscope-only padded minibatch."""

    values = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ],
        dtype=DTYPE,
    )[:batch_size]
    timestamps = torch.tensor([[0.10, 0.20], [0.15, 0.25]], dtype=DTYPE)[:batch_size]
    sample_mask = torch.ones((batch_size, 2), dtype=torch.bool)

    return WindowBatch(
        streams={
            "gyro": SensorStreamBatch(values=values, timestamps=timestamps, sample_mask=sample_mask),
        },
        current_calibration={
            "imu": _state(batch_size, 0.01),
            "fixed": _state(batch_size, -0.02),
        },
        metadata={
            "gyro": SensorMetadata(measurement_type=MeasurementType.IMU_GYROSCOPE, geometry_type=GeometryType.VECTOR, calibration_key="imu"),
        },
        targets=None,
    )


class _GeometryProcessor(nn.Module):
    """Minimal geometry stage that exposes calibration-time usage."""

    def forward(self, streams, metadata, calibration):
        result = {}

        for stream_key, stream in streams.items():
            calibration_state = calibration[metadata[stream_key].calibration_key]
            result[stream_key] = CanonicalSensorStreamBatch(
                values=stream.values,
                timestamps=stream.timestamps + calibration_state.time_offset,
                sample_mask=stream.sample_mask,
                measurement_type=metadata[stream_key].measurement_type,
            )

        return result


class _Model(nn.Module):
    """Record calibration context and emit a deterministic time-offset update."""

    def __init__(self) -> None:
        super().__init__()
        self.heads = nn.ModuleDict({"imu": nn.Identity()})
        self.last_tokens = None
        self.last_context = None

    def forward(self, tokens, calibration_context):
        self.last_tokens = tokens
        self.last_context = calibration_context

        batch_size = tokens.x.shape[0]
        prediction = CalibrationPrediction(
            change_event_logit=torch.zeros((batch_size, 1), dtype=tokens.x.dtype),
            change_time=torch.zeros((batch_size, 1), dtype=tokens.x.dtype),
            delta_xi=torch.zeros((batch_size, 6), dtype=tokens.x.dtype),
            delta_tau=torch.full((batch_size, 1), 0.05, dtype=tokens.x.dtype),
        )

        return ModelOutput(predictions={"imu": prediction}, shared_features=torch.zeros((batch_size, 1), dtype=tokens.x.dtype))


class _Updater:
    """Apply only the deterministic delta_tau used by these orchestration tests."""

    def update(self, state, prediction):
        return CalibrationState(transform=state.transform, time_offset=state.time_offset + prediction.delta_tau)


class _Estimator(ObservabilityEstimator):
    """Return a deterministic full-rank Fisher matrix for every batch element."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = []
        self.layout = CalibrationParameterLayout.from_calibration_keys(("imu",))

    def forward(self, measurements, calibration, metadata):
        self.calls.append((measurements, calibration, metadata))
        batch_size = next(iter(measurements.values())).values.shape[0]
        diagonal = torch.tensor([25.0, 16.0, 9.0, 4.0, 1.0, 0.25, 0.0625], dtype=DTYPE)
        fisher = torch.diag(diagonal).unsqueeze(0).repeat(batch_size, 1, 1)
        timebases = tuple(
            ReferenceTimebase(stream_key="gyro", timestamps=measurements["gyro"].timestamps[index, measurements["gyro"].sample_mask[index]].detach().cpu())
            for index in range(batch_size)
        )

        return ObservabilityResult(
            raw=BatchedObservabilityMatrix(fisher_information_matrix=fisher, layout=self.layout, reference_timebases=timebases),
            features=None,
        )


def test_window_step_teacher_forcing_updates_predicted_key_and_preserves_fixed_state() -> None:
    """Use window.current_calibration when no rollout state is supplied."""

    window = _window()
    model = _Model()
    step = WindowStep(
        geometry_processor=_GeometryProcessor(),
        tokenizer=Tokenizer(measurement_dim=6),
        model=model,
        calibration_updater=_Updater(),
        observability_estimator=None,
        observability_mapper=None,
    )

    result = step(window)

    assert result.observability is None
    torch.testing.assert_close(result.next_calibration["imu"].time_offset, torch.full((2, 1), 0.06, dtype=DTYPE))
    assert result.next_calibration["fixed"] is window.current_calibration["fixed"]

    # Geometry processing and head context both used the teacher-forced tau=0.01.
    torch.testing.assert_close(result.tokens.x[:, :, 6], window.streams["gyro"].timestamps + 0.01)
    torch.testing.assert_close(model.last_context["imu"][:, -1:], torch.full((2, 1), 0.01, dtype=DTYPE))


def test_window_step_explicit_rollout_calibration_overrides_window_state() -> None:
    """An externally carried calibration must control both tokens and head context."""

    window = _window()
    rollout_calibration = {
        "imu": _state(2, 0.20),
        "fixed": _state(2, -0.30),
    }
    model = _Model()
    step = WindowStep(
        geometry_processor=_GeometryProcessor(),
        tokenizer=Tokenizer(measurement_dim=6),
        model=model,
        calibration_updater=_Updater(),
    )

    result = step(window, calibration=rollout_calibration)

    torch.testing.assert_close(result.tokens.x[:, :, 6], window.streams["gyro"].timestamps + 0.20)
    torch.testing.assert_close(model.last_context["imu"][:, -1:], torch.full((2, 1), 0.20, dtype=DTYPE))
    torch.testing.assert_close(result.next_calibration["imu"].time_offset, torch.full((2, 1), 0.25, dtype=DTYPE))
    assert result.next_calibration["fixed"] is rollout_calibration["fixed"]


def test_window_step_observability_is_computed_from_raw_streams_and_repeated_in_tokens() -> None:
    """Verify estimator -> mapper -> tokenizer integration for B > 1."""

    window = _window()
    estimator = _Estimator()
    mapper = CRLBTanhObservabilityMapper(alpha=1.0, numerics=ObservabilityNumericsConfig(relative_tolerance=1e-12))
    model = _Model()
    step = WindowStep(
        geometry_processor=_GeometryProcessor(),
        tokenizer=Tokenizer(measurement_dim=6),
        model=model,
        calibration_updater=_Updater(),
        observability_estimator=estimator,
        observability_mapper=mapper,
    )

    result = step(window)

    assert len(estimator.calls) == 1
    assert estimator.calls[0][0] is window.streams
    assert estimator.calls[0][1] is window.current_calibration
    assert estimator.calls[0][2] is window.metadata

    assert result.observability is not None
    assert result.observability.features.shape == (2, 7)

    # Token width = 6 measurement + timestamp + type + 7 observability.
    assert result.tokens.x.shape == (2, 2, 15)
    expected_observability = result.observability.features[:, None, :].expand(-1, 2, -1)
    torch.testing.assert_close(result.tokens.x[:, :, 8:], expected_observability)


def test_window_step_requires_estimator_and_mapper_as_a_pair() -> None:
    """Observability cannot be half-enabled."""

    estimator = _Estimator()

    try:
        WindowStep(
            geometry_processor=_GeometryProcessor(),
            tokenizer=Tokenizer(measurement_dim=6),
            model=_Model(),
            calibration_updater=_Updater(),
            observability_estimator=estimator,
            observability_mapper=None,
        )
    except ValueError as error:
        assert "either both be provided or both be None" in str(error)
    else:
        raise AssertionError("WindowStep accepted an estimator without a mapper.")
