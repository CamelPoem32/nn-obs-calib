"""Integration tests for sequential calibration rollout across temporal windows."""

import math
from types import SimpleNamespace

import torch
from torch import nn

from obscalib.calibration import CalibrationState, CalibrationUpdater
from obscalib.config import CalibrationHeadConfig, MLPConfig, ObsCalibModelConfig, TransformerConfig
from obscalib.data.structures import GeometryType, MeasurementType, SensorMetadata, SensorStreamBatch, WindowBatch
from obscalib.geometry.lie import se3_exp
from obscalib.geometry.processing import GeometryProcessor
from obscalib.models.obs_calib_model import ObsCalibModel
from obscalib.models.structures import CalibrationPrediction, ModelOutput
from obscalib.pipeline import Rollout, WindowStep
from obscalib.tokenization import Tokenizer


def _make_calibration_state(angle_z: float = 0.0, time_offset: float = 0.0) -> CalibrationState:
    """Create one singleton-batch calibration state with a pure z rotation."""

    xi = torch.tensor([[0.0, 0.0, angle_z, 0.0, 0.0, 0.0]], dtype=torch.float64)

    return CalibrationState(transform=se3_exp(xi), time_offset=torch.tensor([[time_offset]], dtype=torch.float64))


def _make_window(current_calibration: dict[str, CalibrationState]) -> WindowBatch:
    """Create one minimal raw IMU window with one x-axis angular-velocity sample."""

    streams = {
        "imu_gyro": SensorStreamBatch(
            values=torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float64),
            timestamps=torch.tensor([[0.5]], dtype=torch.float64),
            sample_mask=torch.tensor([[True]]),
        ),
    }

    metadata = {
        "imu_gyro": SensorMetadata(measurement_type=MeasurementType.IMU_GYROSCOPE, geometry_type=GeometryType.VECTOR, calibration_key="imu"),
    }

    return WindowBatch(streams=streams, current_calibration=current_calibration, metadata=metadata, targets=None)


class RecordingConstantModel(nn.Module):
    """
    Deterministic stand-in for ObsCalibModel used to test pipeline orchestration.

    Every call predicts

        delta_phi_z = pi / 2
        delta_tau = 0.1.

    The model also records the [phi, rho, tau] calibration context received at
    every window, allowing the test to verify that rollout state is propagated
    into subsequent model calls.
    """

    def __init__(self) -> None:
        super().__init__()

        # WindowStep requires the same public configuration/head interfaces as
        # ObsCalibModel, but the deterministic test does not need a Transformer.
        self.config = SimpleNamespace(calibration_head=SimpleNamespace(calibration_context_dim=7))
        self.heads = nn.ModuleDict({"imu": nn.Identity()})

        # Parameters rather than constants let the test additionally verify that
        # gradients are not detached between rollout windows.
        self.delta_xi_parameter = nn.Parameter(torch.tensor([0.0, 0.0, math.pi / 2.0, 0.0, 0.0, 0.0], dtype=torch.float64))
        self.delta_tau_parameter = nn.Parameter(torch.tensor([0.1], dtype=torch.float64))

        self.recorded_contexts: list[torch.Tensor] = []

    def forward(self, tokens, calibration_context) -> ModelOutput:
        """Return one deterministic correction while recording the supplied prior."""

        context = calibration_context["imu"]
        self.recorded_contexts.append(context.detach().clone())

        batch_size = context.shape[0]

        delta_xi = self.delta_xi_parameter.unsqueeze(0).expand(batch_size, -1)
        delta_tau = self.delta_tau_parameter.unsqueeze(0).expand(batch_size, -1)
        auxiliary_zero = torch.zeros(batch_size, 1, dtype=context.dtype, device=context.device)

        prediction = CalibrationPrediction(change_event_logit=auxiliary_zero, change_time=auxiliary_zero, delta_xi=delta_xi, delta_tau=delta_tau)

        return ModelOutput(predictions={"imu": prediction}, shared_features=None)


def test_rollout_carries_predicted_calibration_into_next_window() -> None:
    """
    Verify the complete sequential state-carrying behavior.

    The test checks that the first prediction changes all three downstream uses
    of calibration in the second window:

        1. spatial measurement transformation,
        2. corrected timestamps,
        3. [phi, rho, tau] calibration-head context.
    """

    initial_calibration = {
        "imu": _make_calibration_state(angle_z=0.0, time_offset=0.1),
        "fixed_reference": _make_calibration_state(angle_z=0.3, time_offset=-0.2),
    }

    # Deliberately place nonsense calibration inside the WindowBatch. Rollout
    # must override it with the explicitly carried calibration dictionary.
    decoy_window_calibration = {
        "imu": _make_calibration_state(angle_z=-math.pi / 2.0, time_offset=9.0),
        "fixed_reference": _make_calibration_state(angle_z=0.3, time_offset=-0.2),
    }

    windows = [
        _make_window(decoy_window_calibration),
        _make_window(decoy_window_calibration),
    ]

    model = RecordingConstantModel()

    window_step = WindowStep(
        geometry_processor=GeometryProcessor(),
        tokenizer=Tokenizer(measurement_dim=6),
        model=model,
        calibration_updater=CalibrationUpdater(),
        observability_estimator=None,
        observability_mapper=None,
    )

    rollout = Rollout(window_step)
    result = rollout(windows, initial_calibration)

    assert len(result.steps) == 2
    assert result.steps[0].observability is None
    assert result.steps[1].observability is None

    # -------------------------------------------------------------------------
    # Window 0 receives the explicit rollout initial state:
    #
    #     R = I
    #     tau = 0.1.
    #
    # Therefore raw [1,0,0] remains [1,0,0], and timestamp 0.5 becomes 0.6.
    # -------------------------------------------------------------------------

    first_tokens = result.steps[0].tokens

    torch.testing.assert_close(first_tokens.x[0, 0, :6], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(first_tokens.x[0, 0, 6], torch.tensor(0.6, dtype=torch.float64))
    torch.testing.assert_close(first_tokens.x[0, 0, 7], torch.tensor(float(MeasurementType.IMU_GYROSCOPE), dtype=torch.float64))

    # First head context must be
    #
    #     [phi, rho, tau] = [0,0,0, 0,0,0, 0.1].
    expected_first_context = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]], dtype=torch.float64)
    torch.testing.assert_close(model.recorded_contexts[0], expected_first_context, atol=1e-12, rtol=1e-12)

    # First prediction applies Rz(pi/2) and +0.1 seconds.
    expected_after_first = _make_calibration_state(angle_z=math.pi / 2.0, time_offset=0.2)

    torch.testing.assert_close(result.steps[0].next_calibration["imu"].transform, expected_after_first.transform, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(result.steps[0].next_calibration["imu"].time_offset, expected_after_first.time_offset, atol=1e-12, rtol=1e-12)

    # -------------------------------------------------------------------------
    # Window 1 must use the prediction from window 0 rather than the bogus
    # WindowBatch.current_calibration.
    #
    # Rz(pi/2) maps sensor x -> world y.
    # tau=0.2 maps timestamp 0.5 -> 0.7.
    # -------------------------------------------------------------------------

    second_tokens = result.steps[1].tokens

    torch.testing.assert_close(second_tokens.x[0, 0, :6], torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64), atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(second_tokens.x[0, 0, 6], torch.tensor(0.7, dtype=torch.float64))

    expected_second_context = torch.tensor([[0.0, 0.0, math.pi / 2.0, 0.0, 0.0, 0.0, 0.2]], dtype=torch.float64)
    torch.testing.assert_close(model.recorded_contexts[1], expected_second_context, atol=1e-10, rtol=1e-10)

    # Two left-multiplicative pi/2 corrections produce Rz(pi), while the time
    # offset progresses 0.1 -> 0.2 -> 0.3.
    expected_final = _make_calibration_state(angle_z=math.pi, time_offset=0.3)

    torch.testing.assert_close(result.final_calibration["imu"].transform, expected_final.transform, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(result.final_calibration["imu"].time_offset, expected_final.time_offset, atol=1e-12, rtol=1e-12)

    # A carried calibration without a learned head remains unchanged.
    torch.testing.assert_close(result.final_calibration["fixed_reference"].transform, initial_calibration["fixed_reference"].transform)
    torch.testing.assert_close(result.final_calibration["fixed_reference"].time_offset, initial_calibration["fixed_reference"].time_offset)

    # Nothing in Rollout or WindowStep should detach the sequential state graph.
    # R[0,1] has nonzero derivative at the final pi rotation and tau obviously
    # depends on both predicted temporal increments.
    loss = result.final_calibration["imu"].transform[0, 0, 1] + result.final_calibration["imu"].time_offset.sum()
    loss.backward()

    assert model.delta_xi_parameter.grad is not None
    assert model.delta_tau_parameter.grad is not None
    assert torch.isfinite(model.delta_xi_parameter.grad).all()
    assert torch.isfinite(model.delta_tau_parameter.grad).all()
    assert torch.abs(model.delta_xi_parameter.grad[2]) > 0.0
    assert torch.abs(model.delta_tau_parameter.grad[0]) > 0.0


def test_window_step_uses_window_calibration_when_no_override_is_given() -> None:
    """Verify the teacher-forced WindowStep path independently from Rollout."""

    teacher_forced_calibration = {
        "imu": _make_calibration_state(angle_z=0.0, time_offset=0.4),
    }

    window = _make_window(teacher_forced_calibration)
    model = RecordingConstantModel()

    window_step = WindowStep(
        geometry_processor=GeometryProcessor(),
        tokenizer=Tokenizer(measurement_dim=6),
        model=model,
        calibration_updater=CalibrationUpdater(),
    )

    result = window_step(window)

    # The raw timestamp 0.5 must use the teacher-forced tau=0.4.
    torch.testing.assert_close(result.tokens.x[0, 0, 6], torch.tensor(0.9, dtype=torch.float64))

    # The same state is supplied to the calibration head.
    expected_context = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4]], dtype=torch.float64)
    torch.testing.assert_close(model.recorded_contexts[0], expected_context, atol=1e-12, rtol=1e-12)


def test_real_model_rollout_runs_end_to_end_and_backpropagates() -> None:
    """
    Smoke-test the real neural architecture inside WindowStep and Rollout.

    The deterministic test above verifies exact orchestration semantics. This
    test verifies that the real Transformer -> MLP -> calibration-head path is
    compatible with the same interfaces and remains differentiable across
    multiple windows.
    """

    torch.manual_seed(0)

    config = ObsCalibModelConfig(
        transformer=TransformerConfig(
            input_dim=8,
            d_model=8,
            n_heads=2,
            num_layers=1,
            dim_feedforward=16,
            dropout=0.0,
            num_summary_tokens=2,
        ),
        mlp=MLPConfig(
            hidden_dims=(12,),
            output_dim=10,
            activation="gelu",
            dropout=0.0,
        ),
        calibration_head=CalibrationHeadConfig(
            calibration_context_dim=7,
            hidden_dims=(8,),
            activation="gelu",
            dropout=0.0,
        ),
        head_keys=("imu",),
    )

    model = ObsCalibModel(config).to(dtype=torch.float64)

    window_step = WindowStep(
        geometry_processor=GeometryProcessor(),
        tokenizer=Tokenizer(measurement_dim=6),
        model=model,
        calibration_updater=CalibrationUpdater(),
    )

    rollout = Rollout(window_step)

    initial_calibration = {
        "imu": _make_calibration_state(),
    }

    windows = [
        _make_window(initial_calibration),
        _make_window(initial_calibration),
    ]

    result = rollout(windows, initial_calibration)

    assert len(result.steps) == 2
    assert result.steps[0].tokens.x.shape == (1, 1, 8)
    assert result.steps[1].tokens.x.shape == (1, 1, 8)
    assert result.final_calibration["imu"].transform.shape == (1, 4, 4)
    assert result.final_calibration["imu"].time_offset.shape == (1, 1)

    assert torch.isfinite(result.final_calibration["imu"].transform).all()
    assert torch.isfinite(result.final_calibration["imu"].time_offset).all()

    # Use the final carried state so gradients must pass through the complete
    # learned path and sequential calibration updates.
    loss = result.final_calibration["imu"].transform.sum() + result.final_calibration["imu"].time_offset.sum()
    loss.backward()

    assert model.heads["imu"].delta_xi_head.weight.grad is not None
    assert model.heads["imu"].delta_tau_head.weight.grad is not None
    assert model.transformer_encoder.learned_summary_tokens.grad is not None

    assert torch.isfinite(model.heads["imu"].delta_xi_head.weight.grad).all()
    assert torch.isfinite(model.heads["imu"].delta_tau_head.weight.grad).all()
    assert torch.isfinite(model.transformer_encoder.learned_summary_tokens.grad).all()