"""Tokenizer tests covering observability repetition and chronological sorting."""

from __future__ import annotations

import pytest
import torch

from obscalib.data.structures import CanonicalSensorStreamBatch, MeasurementType
from obscalib.observability.structures import ObservabilityResult
from obscalib.tokenization.tokenizer import Tokenizer


DTYPE = torch.float64


def _make_streams() -> dict[str, CanonicalSensorStreamBatch]:
    """Build two canonical streams with mixed dimensions and one padded sample."""

    gyro = CanonicalSensorStreamBatch(
        values=torch.tensor(
            [[[3.0, 30.0, 300.0], [1.0, 10.0, 100.0], [999.0, 999.0, 999.0]]],
            dtype=DTYPE,
        ),
        timestamps=torch.tensor([[0.30, 0.10, 0.0]], dtype=DTYPE),
        sample_mask=torch.tensor([[True, True, False]]),
        measurement_type=MeasurementType.IMU_GYROSCOPE,
    )
    lidar = CanonicalSensorStreamBatch(
        values=torch.tensor(
            [[[2.0, 20.0, 200.0, 2000.0, 20000.0, 200000.0], [4.0, 40.0, 400.0, 4000.0, 40000.0, 400000.0]]],
            dtype=DTYPE,
        ),
        timestamps=torch.tensor([[0.20, 0.40]], dtype=DTYPE),
        sample_mask=torch.tensor([[True, True]]),
        measurement_type=MeasurementType.LIDAR_POSE,
    )

    return {"gyro": gyro, "lidar": lidar}


def test_tokenizer_zero_pads_repeats_observability_and_sorts_once() -> None:
    """Verify the complete token-row contract with observability enabled."""

    tokenizer = Tokenizer(measurement_dim=6)
    observability = ObservabilityResult(raw=None, features=torch.tensor([[0.7, 0.8]], dtype=DTYPE))

    tokens = tokenizer(_make_streams(), observability)

    # 6 measurement + 1 timestamp + 1 measurement type + 2 observability.
    assert tokens.x.shape == (1, 5, 10)
    assert tokens.token_mask.tolist() == [[True, True, True, True, False]]

    # Real-token chronological order is gyro@0.10, lidar@0.20, gyro@0.30, lidar@0.40.
    torch.testing.assert_close(tokens.x[0, :4, 6], torch.tensor([0.10, 0.20, 0.30, 0.40], dtype=DTYPE))

    # Gyroscope vectors are zero-padded from 3D to measurement_dim=6.
    torch.testing.assert_close(tokens.x[0, 0, :6], torch.tensor([1.0, 10.0, 100.0, 0.0, 0.0, 0.0], dtype=DTYPE))
    torch.testing.assert_close(tokens.x[0, 2, :6], torch.tensor([3.0, 30.0, 300.0, 0.0, 0.0, 0.0], dtype=DTYPE))

    # LiDAR vectors remain six-dimensional.
    torch.testing.assert_close(tokens.x[0, 1, :6], torch.tensor([2.0, 20.0, 200.0, 2000.0, 20000.0, 200000.0], dtype=DTYPE))

    # Measurement type is part of the complete row and moves with sorting.
    torch.testing.assert_close(
        tokens.x[0, :4, 7],
        torch.tensor(
            [
                float(int(MeasurementType.IMU_GYROSCOPE)),
                float(int(MeasurementType.LIDAR_POSE)),
                float(int(MeasurementType.IMU_GYROSCOPE)),
                float(int(MeasurementType.LIDAR_POSE)),
            ],
            dtype=DTYPE,
        ),
    )

    # One window-level observability vector is repeated across every token row.
    torch.testing.assert_close(tokens.x[0, :, 8:], torch.tensor([[0.7, 0.8]], dtype=DTYPE).expand(5, -1))


def test_tokenizer_without_observability_has_expected_width() -> None:
    """Observability-disabled experiments must not receive fake zero features."""

    tokens = Tokenizer(measurement_dim=6)(_make_streams(), observability=None)

    assert tokens.x.shape == (1, 5, 8)


def test_tokenizer_rejects_native_measurement_dimension_above_configured_width() -> None:
    """A canonical measurement must fit inside the configured zero-padding width."""

    streams = _make_streams()

    with pytest.raises(ValueError, match="larger than measurement_dim"):
        Tokenizer(measurement_dim=5)(streams)


def test_tokenizer_rejects_wrong_observability_batch_size() -> None:
    """One observability feature row is required for every minibatch element."""

    observability = ObservabilityResult(raw=None, features=torch.ones((2, 3), dtype=DTYPE))

    with pytest.raises(ValueError, match=r"shape \[B, d_observability\]"):
        Tokenizer(measurement_dim=6)(_make_streams(), observability)
