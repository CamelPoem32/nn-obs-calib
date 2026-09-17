"""Integration tests for raw measurement preprocessing and final token construction."""

import math

import torch

from obscalib.calibration import CalibrationState
from obscalib.data.structures import GeometryType, MeasurementType, SensorMetadata, SensorStreamBatch
from obscalib.geometry.lie import se3_exp
from obscalib.geometry.processing import GeometryProcessor
from obscalib.observability.structures import ObservabilityResult
from obscalib.tokenization import Tokenizer


def _sensor_in_world_transform() -> torch.Tensor:
    """Create T_sensor_in_world with Rz(90 deg) and translation [1, 0, 0]."""

    angle = math.pi / 2.0

    transform = torch.eye(4, dtype=torch.float64).unsqueeze(0)
    transform[0, :3, :3] = torch.tensor([[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    transform[0, :3, 3] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)

    return transform


def test_measurement_pipeline_applies_calibration_before_log_adds_time_offset_and_sorts_complete_tokens() -> None:
    transform_sensor_in_world = _sensor_in_world_transform()

    calibration = {
        "imu": CalibrationState(transform=transform_sensor_in_world.clone(), time_offset=torch.tensor([[0.2]], dtype=torch.float64)),
        "lidar": CalibrationState(transform=transform_sensor_in_world.clone(), time_offset=torch.tensor([[-0.1]], dtype=torch.float64)),
    }

    # Raw gyro measurements are deliberately supplied in nonchronological order.
    gyro_stream = SensorStreamBatch(
        values=torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float64),
        timestamps=torch.tensor([[3.0, 1.0]], dtype=torch.float64),
        sample_mask=torch.tensor([[True, True]]),
    )

    # Raw relative LiDAR odometry remains an SE(3) matrix when entering the pipeline.
    lidar_xi_sensor = torch.tensor([[[0.0, 0.0, 0.1, 0.0, 0.0, 0.0]]], dtype=torch.float64)

    lidar_stream = SensorStreamBatch(
        values=se3_exp(lidar_xi_sensor),
        timestamps=torch.tensor([[2.0]], dtype=torch.float64),
        sample_mask=torch.tensor([[True]]),
    )

    streams = {
        "imu_gyro": gyro_stream,
        "lidar_odometry": lidar_stream,
    }

    metadata = {
        "imu_gyro": SensorMetadata(measurement_type=MeasurementType.IMU_GYROSCOPE, geometry_type=GeometryType.VECTOR, calibration_key="imu"),
        "lidar_odometry": SensorMetadata(measurement_type=MeasurementType.LIDAR_POSE, geometry_type=GeometryType.SE3, calibration_key="lidar"),
    }

    # Stage 1:
    # raw measurement -> calibration-prior frame transformation -> Log().vee().
    canonical_streams = GeometryProcessor()(streams, metadata, calibration)

    # Rz(90 deg):
    # sensor x -> world y
    # sensor y -> world -x.
    torch.testing.assert_close(canonical_streams["imu_gyro"].values[0, 0], torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(canonical_streams["imu_gyro"].values[0, 1], torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64), atol=1e-12, rtol=1e-12)

    # SE(3) conjugation happens before Log().vee().
    #
    # For the world-frame lever arm t=[1,0,0] and phi=[0,0,0.1],
    #
    #     rho_world = t x phi_world = [0,-0.1,0].
    expected_lidar_vector = torch.tensor([0.0, 0.0, 0.1, 0.0, -0.1, 0.0], dtype=torch.float64)
    torch.testing.assert_close(canonical_streams["lidar_odometry"].values[0, 0], expected_lidar_vector, atol=1e-10, rtol=1e-10)

    # Temporal calibration convention is additive.
    #
    # gyro:  [3,1] + 0.2  -> [3.2,1.2]
    # lidar: [2]   - 0.1  -> [1.9]
    torch.testing.assert_close(canonical_streams["imu_gyro"].timestamps, torch.tensor([[3.2, 1.2]], dtype=torch.float64))
    torch.testing.assert_close(canonical_streams["lidar_odometry"].timestamps, torch.tensor([[1.9]], dtype=torch.float64))

    # Stage 2:
    # zero-pad canonical vectors -> append time/type/observability -> concatenate -> sort.
    observability = ObservabilityResult(features=torch.tensor([[10.0, 20.0]], dtype=torch.float64))
    tokens = Tokenizer(measurement_dim=6)(canonical_streams, observability)

    # 6 canonical coordinates + time + type + 2 observability values.
    assert tokens.x.shape == (1, 3, 10)
    assert torch.equal(tokens.token_mask, torch.tensor([[True, True, True]]))

    # Corrected chronological order:
    #
    # gyro second sample  t=1.2
    # lidar               t=1.9
    # gyro first sample   t=3.2
    torch.testing.assert_close(tokens.x[0, :, 6], torch.tensor([1.2, 1.9, 3.2], dtype=torch.float64))

    torch.testing.assert_close(tokens.x[0, :, 7], torch.tensor([float(MeasurementType.IMU_GYROSCOPE), float(MeasurementType.LIDAR_POSE), float(MeasurementType.IMU_GYROSCOPE)], dtype=torch.float64))

    # Corresponding measurement vectors must move together with their corrected timestamps.
    torch.testing.assert_close(tokens.x[0, 0, :6], torch.tensor([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(tokens.x[0, 1, :6], expected_lidar_vector, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(tokens.x[0, 2, :6], torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64), atol=1e-12, rtol=1e-12)

    # Window-level observability is repeated identically for every token and
    # remains attached after the final chronological sorting.
    torch.testing.assert_close(tokens.x[0, :, 8:], torch.tensor([[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]], dtype=torch.float64))


def test_measurement_pipeline_without_observability_uses_only_measurement_time_and_type() -> None:
    calibration = {
        "imu": CalibrationState(transform=torch.eye(4, dtype=torch.float64).unsqueeze(0), time_offset=torch.tensor([[0.25]], dtype=torch.float64)),
    }

    streams = {
        "imu_gyro": SensorStreamBatch(values=torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float64), timestamps=torch.tensor([[0.5]], dtype=torch.float64), sample_mask=torch.tensor([[True]])),
    }

    metadata = {
        "imu_gyro": SensorMetadata(measurement_type=MeasurementType.IMU_GYROSCOPE, geometry_type=GeometryType.VECTOR, calibration_key="imu"),
    }

    canonical_streams = GeometryProcessor()(streams, metadata, calibration)

    torch.testing.assert_close(canonical_streams["imu_gyro"].timestamps, torch.tensor([[0.75]], dtype=torch.float64))

    tokens = Tokenizer(measurement_dim=6)(canonical_streams, observability=None)

    # 6 padded measurement coordinates + timestamp + measurement type.
    assert tokens.x.shape == (1, 1, 8)

    torch.testing.assert_close(tokens.x[0, 0, :6], torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(tokens.x[0, 0, 6], torch.tensor(0.75, dtype=torch.float64))
    torch.testing.assert_close(tokens.x[0, 0, 7], torch.tensor(float(MeasurementType.IMU_GYROSCOPE), dtype=torch.float64))