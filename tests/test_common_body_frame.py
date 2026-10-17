"""Tests for obscalib.augmentations.common_body_frame (place under tests/)."""

from types import SimpleNamespace

import pytest
import torch

from obscalib.augmentations.common_body_frame import CommonBodyFrameRandomizer
from obscalib.augmentations.config import CommonBodyFrameRandomizationConfig
from obscalib.calibration.state import CalibrationState
from obscalib.data.structures import CalibrationTargetBatch, SensorStreamBatch, WindowBatch


def sample_window(batch_size=2):
    def extrinsic(translation):
        T = torch.eye(4, dtype=torch.float64).expand(batch_size, 4, 4).clone()
        T[:, :3, 3] = torch.as_tensor(translation, dtype=torch.float64)
        return T

    a = extrinsic([0.2, 0.1, 0.3])
    b = extrinsic([-0.1, 0.0, 0.5])
    ts = torch.linspace(0.0, 0.4, 5, dtype=torch.float64).expand(batch_size, 5).clone()
    mask = torch.ones(batch_size, 5, dtype=torch.bool)

    streams = {
        "gyro": SensorStreamBatch(values=torch.randn(batch_size, 5, 3, dtype=torch.float64), timestamps=ts, sample_mask=mask, interval_start_timestamps=None),
        "accel": SensorStreamBatch(values=torch.randn(batch_size, 5, 3, dtype=torch.float64), timestamps=ts, sample_mask=mask, interval_start_timestamps=None),
        "lidar": SensorStreamBatch(values=torch.eye(4, dtype=torch.float64).expand(batch_size, 5, 4, 4).clone(), timestamps=ts, sample_mask=mask, interval_start_timestamps=ts - 0.1),
    }
    calibration = {
        "imu": CalibrationState(transform=a, time_offset=torch.full((batch_size, 1), 0.05, dtype=torch.float64)),
        "lidar": CalibrationState(transform=b, time_offset=torch.zeros(batch_size, 1, dtype=torch.float64)),
    }
    target = CalibrationTargetBatch(next_transform=a.clone(), next_time_offset=calibration["imu"].time_offset.clone(), change_label=None, change_time=None)
    return WindowBatch(streams=streams, metadata={"gyro": SimpleNamespace(calibration_key="imu"), "accel": SimpleNamespace(calibration_key="imu"), "lidar": SimpleNamespace(calibration_key="lidar")}, current_calibration=calibration, targets={"imu": target}, dataset_indices=None)


def test_shared_common_frame_preserves_native_measurements_and_relative_extrinsics():
    original = sample_window()
    out, calibration, H = CommonBodyFrameRandomizer(CommonBodyFrameRandomizationConfig(enabled=True))(original, generator=torch.Generator().manual_seed(5))
    assert H.shape == (2, 4, 4)
    assert not torch.allclose(H[:, :3, :3], torch.eye(3, dtype=H.dtype).expand(2, 3, 3))

    for key in original.current_calibration:
        torch.testing.assert_close(calibration[key].transform, H @ original.current_calibration[key].transform)
        torch.testing.assert_close(calibration[key].time_offset, original.current_calibration[key].time_offset)

    for key in original.streams:
        assert out.streams[key] is original.streams[key]  # No raw-stream transformation.

    relative_before = torch.linalg.inv(original.current_calibration["lidar"].transform) @ original.current_calibration["imu"].transform
    relative_after = torch.linalg.inv(calibration["lidar"].transform) @ calibration["imu"].transform
    torch.testing.assert_close(relative_after, relative_before, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(out.targets["imu"].next_transform, H @ original.targets["imu"].next_transform)
    torch.testing.assert_close(out.targets["imu"].next_time_offset, original.targets["imu"].next_time_offset)

    # Canonical IMU body vector rotates, unlike a sensor-frame re-expression.
    v = original.streams["gyro"].values[:, 0, :].unsqueeze(-1)
    R_old = original.current_calibration["imu"].transform[:, :3, :3]
    R_new = calibration["imu"].transform[:, :3, :3]
    torch.testing.assert_close(R_new @ v, H[:, :3, :3] @ (R_old @ v))


def test_disabled_and_zero_probability_are_identity():
    original = sample_window()
    for cfg in (CommonBodyFrameRandomizationConfig(enabled=False), CommonBodyFrameRandomizationConfig(enabled=True, probability=0.0)):
        out, calibration, H = CommonBodyFrameRandomizer(cfg)(original, generator=torch.Generator().manual_seed(5))
        torch.testing.assert_close(H, torch.eye(4, dtype=H.dtype).expand_as(H))
        for key in calibration:
            torch.testing.assert_close(calibration[key].transform, original.current_calibration[key].transform)
        assert out.streams["gyro"] is original.streams["gyro"]


def test_seed_reproducibility_and_probability_validation():
    window = sample_window()
    augmenter = CommonBodyFrameRandomizer(CommonBodyFrameRandomizationConfig(enabled=True, probability=0.5))
    _, _, first = augmenter(window, generator=torch.Generator().manual_seed(14))
    _, _, second = augmenter(window, generator=torch.Generator().manual_seed(14))
    torch.testing.assert_close(first, second)
    with pytest.raises(ValueError):
        CommonBodyFrameRandomizationConfig(enabled=True, probability=1.5)


def test_left_multiplicative_correction_target_transforms_by_conjugation():
    window = sample_window(batch_size=2)
    true_transform = window.targets["imu"].next_transform.clone()
    wrong_transform = window.current_calibration["imu"].transform.clone()
    # Nontrivial correction in the original common frame.
    delta = torch.eye(4, dtype=wrong_transform.dtype).expand(2, 4, 4).clone()
    angle = torch.tensor([0.15, -0.25], dtype=wrong_transform.dtype)
    delta[:, 0, 0] = angle.cos()
    delta[:, 0, 1] = -angle.sin()
    delta[:, 1, 0] = angle.sin()
    delta[:, 1, 1] = angle.cos()
    wrong_transform = torch.linalg.inv(delta) @ true_transform

    changed_window, _, H = CommonBodyFrameRandomizer(CommonBodyFrameRandomizationConfig(enabled=True))(window, generator=torch.Generator().manual_seed(17))
    old_correction = true_transform @ torch.linalg.inv(wrong_transform)
    new_correction = changed_window.targets["imu"].next_transform @ torch.linalg.inv(H @ wrong_transform)
    torch.testing.assert_close(new_correction, H @ old_correction @ torch.linalg.inv(H), rtol=1e-11, atol=1e-11)
    assert not torch.allclose(H[0], H[1]), "Different independent batch items should sample different common frames."
