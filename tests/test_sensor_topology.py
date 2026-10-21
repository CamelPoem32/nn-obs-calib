
"""Small geometry and probability checks for the topology augmentation."""

import torch

from obscalib.augmentations.config import SensorTopologyAugmentationConfig
from obscalib.augmentations.sensor_topology import SensorTopologyAugmenter
from obscalib.calibration.state import CalibrationState
from obscalib.data.structures import GeometryType, MeasurementType, SensorMetadata, SensorStreamBatch, WindowBatch


def _base_window(batch_size=2):
    T = torch.eye(4).repeat(batch_size, 1, 1)
    tau = torch.zeros(batch_size, 1)
    states = {key: CalibrationState(transform=T.clone(), time_offset=tau.clone()) for key in ("imu", "lidar", "lidar_right")}

    gyro = torch.randn(batch_size, 16, 3)
    accel = torch.randn(batch_size, 16, 3)
    lidar = torch.eye(4).repeat(batch_size, 4, 1, 1)
    t_imu = torch.linspace(0, 1.5, 16).repeat(batch_size, 1)
    t_lidar = torch.linspace(0.3, 1.5, 4).repeat(batch_size, 1)
    start_lidar = t_lidar - 0.3

    streams = {
        "gyro": SensorStreamBatch(values=gyro, timestamps=t_imu, sample_mask=torch.ones(batch_size, 16, dtype=torch.bool)),
        "accel": SensorStreamBatch(values=accel, timestamps=t_imu, sample_mask=torch.ones(batch_size, 16, dtype=torch.bool)),
        "lidar": SensorStreamBatch(values=lidar, timestamps=t_lidar, sample_mask=torch.ones(batch_size, 4, dtype=torch.bool), interval_start_timestamps=start_lidar),
        "lidar_right": SensorStreamBatch(values=lidar.clone(), timestamps=t_lidar, sample_mask=torch.ones(batch_size, 4, dtype=torch.bool), interval_start_timestamps=start_lidar),
    }
    metadata = {
        "gyro": SensorMetadata(measurement_type=MeasurementType.IMU_GYROSCOPE, geometry_type=GeometryType.VECTOR, calibration_key="imu"),
        "accel": SensorMetadata(measurement_type=MeasurementType.IMU_ACCELEROMETER, geometry_type=GeometryType.VECTOR, calibration_key="imu"),
        "lidar": SensorMetadata(measurement_type=MeasurementType.LIDAR_POSE, geometry_type=GeometryType.SE3, calibration_key="lidar"),
        "lidar_right": SensorMetadata(measurement_type=MeasurementType.LIDAR_POSE, geometry_type=GeometryType.SE3, calibration_key="lidar_right"),
    }
    return WindowBatch(streams=streams, current_calibration=states, metadata=metadata, targets=None, dataset_indices=None)


def test_both_lidar_replacement_choices_preserve_ideal_body_motion():
    original = _base_window()
    for first_probability, removed_key in ((1.0, "lidar"), (0.0, "lidar_right")):
        config = SensorTopologyAugmentationConfig(enabled=True, replacement_probability=1.0, replace_first_lidar_probability=first_probability, maximum_mount_rotation_deg=180.0)
        modified, mode = SensorTopologyAugmenter(config)(original, generator=torch.Generator().manual_seed(42))

        assert mode == f"replace_{removed_key}"
        assert removed_key not in modified.current_calibration
        assert removed_key not in modified.streams
        assert set(modified.current_calibration) == {"imu", "imu_synth", ({"lidar", "lidar_right"} - {removed_key}).pop()}
        assert len(modified.streams) == 5

        for source, synthetic in (("gyro", "gyro_synth"), ("accel", "accel_synth")):
            R_source = original.current_calibration["imu"].transform[:, :3, :3]
            R_synth = modified.current_calibration["imu_synth"].transform[:, :3, :3]
            body_source = original.streams[source].values @ R_source.transpose(-1, -2)
            body_synth = modified.streams[synthetic].values @ R_synth.transpose(-1, -2)

            torch.testing.assert_close(body_source, body_synth, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(original.streams[source].timestamps, modified.streams[synthetic].timestamps)


def test_topology_probabilities_are_approximately_balanced():
    original = _base_window()
    config = SensorTopologyAugmentationConfig(enabled=True, replacement_probability=0.5, replace_first_lidar_probability=0.5)
    augmenter = SensorTopologyAugmenter(config)
    generator = torch.Generator().manual_seed(1234)
    counts = {"original": 0, "replace_lidar": 0, "replace_lidar_right": 0}

    for _ in range(200):
        _, mode = augmenter(original, generator=generator)
        counts[mode] += 1

    assert 70 <= counts["original"] <= 130, counts
    assert 30 <= counts["replace_lidar"] <= 70, counts
    assert 30 <= counts["replace_lidar_right"] <= 70, counts
