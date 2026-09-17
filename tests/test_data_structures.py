import pytest
import torch

from obscalib.calibration import CalibrationState as CanonicalCalibrationState
from obscalib.data import CalibrationState, SensorStreamBatch


def test_sensor_stream_batch_validation_accepts_heterogeneous_feature_width() -> None:
    stream = SensorStreamBatch(
        values=torch.randn(2, 5, 7),
        timestamps=torch.randn(2, 5),
        sample_mask=torch.ones(2, 5, dtype=torch.bool),
    )

    stream.validate()


def test_sensor_stream_batch_validation_rejects_misaligned_sequence() -> None:
    stream = SensorStreamBatch(
        values=torch.randn(2, 5, 3),
        timestamps=torch.randn(2, 4),
        sample_mask=torch.ones(2, 5, dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="share B and N"):
        stream.validate()


def test_calibration_state_validation_keeps_state_representation_explicit() -> None:
    state = CalibrationState(
        transform=torch.eye(4).repeat(3, 1, 1),
        time_offset=torch.zeros(3, 1),
    )

    state.validate()
