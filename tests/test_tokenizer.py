"""Tests for zero-padded multi-sensor Transformer token construction."""

import torch

from obscalib.data.structures import CanonicalSensorStreamBatch, MeasurementType
from obscalib.observability.structures import ObservabilityResult
from obscalib.tokenization import Tokenizer


def test_tokenizer_zero_pads_measurements_appends_metadata_and_sorts_by_time() -> None:
    streams = {
        "gyroscope": CanonicalSensorStreamBatch(
            values=torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=torch.float64),
            timestamps=torch.tensor([[3.0, 1.0]], dtype=torch.float64),
            sample_mask=torch.tensor([[True, True]]),
            measurement_type=MeasurementType.IMU_GYROSCOPE,
        ),
        "lidar": CanonicalSensorStreamBatch(
            values=torch.tensor([[[10.0, 11.0, 12.0, 13.0, 14.0, 15.0]]], dtype=torch.float64),
            timestamps=torch.tensor([[2.0]], dtype=torch.float64),
            sample_mask=torch.tensor([[True]]),
            measurement_type=MeasurementType.LIDAR_POSE,
        ),
    }

    tokens = Tokenizer(measurement_dim=6)(streams)

    # Final chronological order is:
    # gyro(t=1), lidar(t=2), gyro(t=3).
    assert tokens.x.shape == (1, 3, 8)
    assert torch.equal(tokens.token_mask, torch.tensor([[True, True, True]]))

    # First six entries contain zero-padded canonical measurement vectors.
    torch.testing.assert_close(tokens.x[0, 0, :6], torch.tensor([4.0, 5.0, 6.0, 0.0, 0.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(tokens.x[0, 1, :6], torch.tensor([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=torch.float64))
    torch.testing.assert_close(tokens.x[0, 2, :6], torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=torch.float64))

    # Timestamp is the seventh token component.
    torch.testing.assert_close(tokens.x[0, :, 6], torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))

    # Measurement type is the eighth component and remains aligned after sorting.
    torch.testing.assert_close(tokens.x[0, :, 7], torch.tensor([float(MeasurementType.IMU_GYROSCOPE), float(MeasurementType.LIDAR_POSE), float(MeasurementType.IMU_GYROSCOPE)], dtype=torch.float64))


def test_tokenizer_repeats_window_observability_for_every_measurement() -> None:
    streams = {
        "gyroscope": CanonicalSensorStreamBatch(
            values=torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=torch.float64),
            timestamps=torch.tensor([[0.5, 1.5]], dtype=torch.float64),
            sample_mask=torch.tensor([[True, True]]),
            measurement_type=MeasurementType.IMU_GYROSCOPE,
        ),
    }

    observability = ObservabilityResult(features=torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float64))

    tokens = Tokenizer(measurement_dim=6)(streams, observability)

    # 6 measurement + 1 timestamp + 1 type + 3 observability.
    assert tokens.x.shape == (1, 2, 11)

    torch.testing.assert_close(tokens.x[0, :, 8:], torch.tensor([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]], dtype=torch.float64))


def test_tokenizer_without_observability_adds_no_dummy_dimensions() -> None:
    streams = {
        "gyroscope": CanonicalSensorStreamBatch(
            values=torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float64),
            timestamps=torch.tensor([[0.5]], dtype=torch.float64),
            sample_mask=torch.tensor([[True]]),
            measurement_type=MeasurementType.IMU_GYROSCOPE,
        ),
    }

    tokens = Tokenizer(measurement_dim=6)(streams, observability=None)

    assert tokens.x.shape == (1, 1, 8)


def test_tokenizer_moves_padding_to_the_end_without_losing_alignment() -> None:
    streams = {
        "gyroscope": CanonicalSensorStreamBatch(
            values=torch.tensor([[[1.0, 0.0, 0.0], [999.0, 999.0, 999.0], [2.0, 0.0, 0.0]]], dtype=torch.float64),
            timestamps=torch.tensor([[2.0, 0.0, 1.0]], dtype=torch.float64),
            sample_mask=torch.tensor([[True, False, True]]),
            measurement_type=MeasurementType.IMU_GYROSCOPE,
        ),
    }

    tokens = Tokenizer(measurement_dim=6)(streams)

    assert torch.equal(tokens.token_mask, torch.tensor([[True, True, False]]))

    # Real measurements are sorted first: value 2 at t=1, then value 1 at t=2.
    torch.testing.assert_close(tokens.x[0, 0, :3], torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(tokens.x[0, 1, :3], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(tokens.x[0, :2, 6], torch.tensor([1.0, 2.0], dtype=torch.float64))


def test_tokenizer_rejects_measurements_wider_than_configured_dimension() -> None:
    streams = {
        "measurement": CanonicalSensorStreamBatch(
            values=torch.zeros(1, 2, 7),
            timestamps=torch.tensor([[0.0, 1.0]]),
            sample_mask=torch.tensor([[True, True]]),
            measurement_type=MeasurementType.LIDAR_POSE,
        ),
    }

    tokenizer = Tokenizer(measurement_dim=6)

    try:
        tokenizer(streams)
    except ValueError as error:
        assert "larger than measurement_dim=6" in str(error)
    else:
        raise AssertionError("Expected Tokenizer to reject a canonical dimension larger than measurement_dim.")