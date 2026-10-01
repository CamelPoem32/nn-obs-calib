"""Tests for calibration-wise observability prediction, target caching, and dataset identity propagation."""

from __future__ import annotations

import math

import pytest
import torch

from obscalib.calibration.observability_targets import ObservabilityTargetCache
from obscalib.calibration.state import CalibrationState
from obscalib.config import ObservabilityPredictionHeadConfig, ObservabilityPredictionModelConfig, TransformerConfig
from obscalib.data.collation import collate_windows
from obscalib.data.sorting import sort_measurement_sequence
from obscalib.data.structures import CanonicalSensorStreamBatch, GeometryType, MeasurementSequenceBatch, MeasurementType, SensorMetadata, SensorStream, TokenBatch, WindowSample
from obscalib.losses.observability_prediction_loss import ObservabilityPredictionLoss
from obscalib.models.observability_prediction_model import ObservabilityPredictionModel
from obscalib.models.structures import ObservabilityPredictionOutput
from obscalib.observability.layout import CalibrationParameterLayout
from obscalib.observability.mappings import dimensionless_soft_projector_observability
from obscalib.tokenization.tokenizer import Tokenizer


CALIBRATION_KEYS = ("imu", "lidar")


def make_calibration_state(batch_size: int = 1, dtype: torch.dtype = torch.float32) -> CalibrationState:
    transform = torch.eye(4, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    time_offset = torch.zeros((batch_size, 1), dtype=dtype)
    return CalibrationState(transform=transform, time_offset=time_offset)


def make_window_sample(dataset_index: int) -> WindowSample:
    stream = SensorStream(
        values=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),
        timestamps=torch.tensor([0.0, 0.1], dtype=torch.float32),
    )

    return WindowSample(
        streams={"gyro": stream},
        current_calibration={"imu": make_calibration_state()},
        metadata={"gyro": SensorMetadata(measurement_type=MeasurementType.IMU_GYROSCOPE, geometry_type=GeometryType.VECTOR, calibration_key="imu")},
        targets=None,
        dataset_index=dataset_index,
    )


def make_canonical_streams(batch_size: int = 2) -> dict[str, CanonicalSensorStreamBatch]:
    gyro_values = torch.randn((batch_size, 3, 3), dtype=torch.float32)
    lidar_values = torch.randn((batch_size, 2, 6), dtype=torch.float32)

    gyro_timestamps = torch.tensor([[0.0, 0.2, 0.4]], dtype=torch.float32).repeat(batch_size, 1)
    lidar_timestamps = torch.tensor([[0.1, 0.3]], dtype=torch.float32).repeat(batch_size, 1)

    gyro_mask = torch.ones((batch_size, 3), dtype=torch.bool)
    lidar_mask = torch.ones((batch_size, 2), dtype=torch.bool)

    return {
        "gyro": CanonicalSensorStreamBatch(values=gyro_values, timestamps=gyro_timestamps, sample_mask=gyro_mask, measurement_type=MeasurementType.IMU_GYROSCOPE, calibration_key="imu"),
        "lidar": CanonicalSensorStreamBatch(values=lidar_values, timestamps=lidar_timestamps, sample_mask=lidar_mask, measurement_type=MeasurementType.LIDAR_POSE, calibration_key="lidar"),
    }


def make_model(head_mode: str) -> ObservabilityPredictionModel:
    config = ObservabilityPredictionModelConfig(
        transformer=TransformerConfig(input_dim=8, d_model=16, n_heads=4, num_layers=1, dim_feedforward=32, dropout=0.0, num_summary_tokens=1, activation="gelu"),
        calibration_keys=CALIBRATION_KEYS,
        head=ObservabilityPredictionHeadConfig(hidden_dims=(12,), activation="gelu", dropout=0.0),
        head_mode=head_mode,
    )

    return ObservabilityPredictionModel(config)


def make_model_tokens(batch_size: int = 2) -> TokenBatch:
    x = torch.randn((batch_size, 6, 8), dtype=torch.float32)
    token_mask = torch.ones((batch_size, 6), dtype=torch.bool)
    calibration_ids = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long).repeat(batch_size, 1)
    return TokenBatch(x=x, token_mask=token_mask, calibration_ids=calibration_ids)


def test_collate_windows_preserves_dataset_indices() -> None:
    batch = collate_windows([make_window_sample(7), make_window_sample(19)])

    assert batch.dataset_indices is not None
    assert batch.dataset_indices.dtype == torch.long
    assert batch.dataset_indices.device.type == "cpu"
    assert torch.equal(batch.dataset_indices, torch.tensor([7, 19], dtype=torch.long))


def test_sort_measurement_sequence_preserves_calibration_id_alignment() -> None:
    measurements = MeasurementSequenceBatch(
        x=torch.tensor([[[20.0], [0.0], [10.0]]], dtype=torch.float32),
        timestamps=torch.tensor([[2.0, 0.0, 1.0]], dtype=torch.float32),
        token_mask=torch.tensor([[True, True, True]]),
        calibration_ids=torch.tensor([[0, 1, 0]], dtype=torch.long),
    )

    sorted_measurements = sort_measurement_sequence(measurements)

    assert torch.equal(sorted_measurements.timestamps, torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32))
    assert torch.equal(sorted_measurements.x[..., 0], torch.tensor([[0.0, 10.0, 20.0]], dtype=torch.float32))
    assert torch.equal(sorted_measurements.calibration_ids, torch.tensor([[1, 0, 0]], dtype=torch.long))


def test_tokenizer_legacy_mode_does_not_create_calibration_ids() -> None:
    streams = make_canonical_streams(batch_size=2)
    tokenizer = Tokenizer(measurement_dim=6)

    tokens = tokenizer(streams)

    assert tokens.calibration_ids is None
    assert tokens.x.shape == (2, 5, 8)
    assert tokens.token_mask.shape == (2, 5)


def test_tokenizer_calibration_aware_mode_creates_sorted_calibration_ids() -> None:
    streams = make_canonical_streams(batch_size=2)
    tokenizer = Tokenizer(measurement_dim=6, calibration_keys=CALIBRATION_KEYS)

    tokens = tokenizer(streams)

    assert tokens.calibration_ids is not None
    assert tokens.calibration_ids.shape == (2, 5)

    expected_ids = torch.tensor([[0, 1, 0, 1, 0]], dtype=torch.long).repeat(2, 1)

    assert torch.equal(tokens.calibration_ids, expected_ids)


@pytest.mark.parametrize("head_mode", ("individual", "shared"))
def test_observability_prediction_model_output_contract(head_mode: str) -> None:
    model = make_model(head_mode)
    tokens = make_model_tokens(batch_size=3)

    output = model(tokens)

    assert tuple(output.predictions) == CALIBRATION_KEYS
    assert tuple(output.pooled_features) == CALIBRATION_KEYS

    for calibration_key in CALIBRATION_KEYS:
        assert output.predictions[calibration_key].shape == (3, 7)
        assert output.pooled_features[calibration_key].shape == (3, 16)
        assert torch.isfinite(output.predictions[calibration_key]).all()
        assert torch.all(output.predictions[calibration_key] >= 0.0)
        assert torch.all(output.predictions[calibration_key] <= 1.0)


def test_individual_head_mode_uses_distinct_heads() -> None:
    model = make_model("individual")

    assert model.shared_head is None
    assert set(model.individual_heads) == set(CALIBRATION_KEYS)
    assert model.individual_heads["imu"] is not model.individual_heads["lidar"]


def test_shared_head_mode_uses_one_head() -> None:
    model = make_model("shared")

    assert model.shared_head is not None
    assert len(model.individual_heads) == 0


def test_observability_prediction_model_requires_calibration_ids() -> None:
    model = make_model("individual")
    tokens = TokenBatch(x=torch.randn((2, 4, 8)), token_mask=torch.ones((2, 4), dtype=torch.bool), calibration_ids=None)

    with pytest.raises(ValueError, match="calibration_ids"):
        model(tokens)


def test_observability_prediction_model_rejects_missing_calibration_tokens() -> None:
    model = make_model("individual")

    tokens = TokenBatch(
        x=torch.randn((2, 4, 8)),
        token_mask=torch.ones((2, 4), dtype=torch.bool),
        calibration_ids=torch.zeros((2, 4), dtype=torch.long),
    )

    with pytest.raises(ValueError, match="lidar"):
        model(tokens)


def test_observability_target_cache_preserves_requested_index_order() -> None:
    values = torch.linspace(0.0, 1.0, 3 * 2 * 7, dtype=torch.float32).reshape(3, 2, 7)
    cache = ObservabilityTargetCache(values=values, calibration_keys=CALIBRATION_KEYS)

    targets = cache.targets_for_indices(torch.tensor([2, 0], dtype=torch.long), device="cpu", dtype=torch.float32)

    assert torch.equal(targets["imu"], values[[2, 0], 0])
    assert torch.equal(targets["lidar"], values[[2, 0], 1])


def test_dimensionless_soft_projector_identity_case() -> None:
    layout = CalibrationParameterLayout.from_calibration_keys(("imu",))
    calibration_scale = (math.radians(5.0), math.radians(5.0), math.radians(5.0), 0.05, 0.05, 0.05, 0.015)
    scale = torch.tensor(calibration_scale, dtype=torch.float64)

    fisher_information_matrix = torch.diag(1.0 / scale.square())

    coordinate_observability, soft_projector, dimensionless_fisher, eigenvalues = dimensionless_soft_projector_observability(
        fisher_information_matrix,
        layout,
        calibration_scale=calibration_scale,
        lambda_value_setting=1.0,
    )

    assert torch.allclose(dimensionless_fisher, torch.eye(7, dtype=torch.float64), atol=1e-12, rtol=0.0)
    assert torch.allclose(eigenvalues, torch.ones(7, dtype=torch.float64), atol=1e-12, rtol=0.0)
    assert torch.allclose(soft_projector, 0.5 * torch.eye(7, dtype=torch.float64), atol=1e-12, rtol=0.0)
    assert torch.allclose(coordinate_observability, torch.full((7,), 0.5, dtype=torch.float64), atol=1e-12, rtol=0.0)


def test_dimensionless_soft_projector_zero_information_maps_to_zero() -> None:
    layout = CalibrationParameterLayout.from_calibration_keys(("imu",))
    calibration_scale = (math.radians(5.0), math.radians(5.0), math.radians(5.0), 0.05, 0.05, 0.05, 0.015)

    coordinate_observability, soft_projector, dimensionless_fisher, eigenvalues = dimensionless_soft_projector_observability(
        torch.zeros((7, 7), dtype=torch.float64),
        layout,
        calibration_scale=calibration_scale,
        lambda_value_setting=0.25,
    )

    assert torch.equal(dimensionless_fisher, torch.zeros((7, 7), dtype=torch.float64))
    assert torch.equal(eigenvalues, torch.zeros(7, dtype=torch.float64))
    assert torch.equal(soft_projector, torch.zeros((7, 7), dtype=torch.float64))
    assert torch.equal(coordinate_observability, torch.zeros(7, dtype=torch.float64))


def test_observability_prediction_loss_is_zero_for_exact_prediction() -> None:
    prediction = {
        "imu": torch.full((2, 7), 0.25, dtype=torch.float32),
        "lidar": torch.full((2, 7), 0.75, dtype=torch.float32),
    }

    output = ObservabilityPredictionOutput(predictions=prediction, pooled_features={})
    targets = {key: value.clone() for key, value in prediction.items()}
    loss_result = ObservabilityPredictionLoss()(output, targets)

    assert torch.allclose(loss_result.total, torch.zeros_like(loss_result.total))

    for value in loss_result.per_calibration.values():
        assert torch.allclose(value, torch.zeros_like(value))


@pytest.mark.parametrize("head_mode", ("individual", "shared"))
def test_full_observability_prediction_path_backpropagates(head_mode: str) -> None:
    torch.manual_seed(123)

    streams = make_canonical_streams(batch_size=2)
    tokenizer = Tokenizer(measurement_dim=6, calibration_keys=CALIBRATION_KEYS)
    tokens = tokenizer(streams)
    model = make_model(head_mode)
    output = model(tokens)

    targets = {
        "imu": torch.full((2, 7), 0.3, dtype=torch.float32),
        "lidar": torch.full((2, 7), 0.7, dtype=torch.float32),
    }

    loss_result = ObservabilityPredictionLoss()(output, targets)
    loss_result.total.backward()

    assert torch.isfinite(loss_result.total)

    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]

    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)