"""Boreas-RT containers using the existing calibration pipeline array conventions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class IMUData:
    """
    Synchronized accelerometer and gyroscope measurements for one IMU.

    timestamps_s:
        Strictly increasing timestamps in seconds with shape [N].

    accel_mps2:
        Accelerometer measurements with shape [N, 3].

    gyro_radps:
        Gyroscope measurements with shape [N, 3].
    """

    timestamps_s: np.ndarray
    accel_mps2: np.ndarray
    gyro_radps: np.ndarray
    name: str = "imu"
    frame_id: str | None = None
    frequency_hz: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamps_ns: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamps_s", np.asarray(self.timestamps_s, dtype=float))
        object.__setattr__(self, "accel_mps2", np.asarray(self.accel_mps2, dtype=float))
        object.__setattr__(self, "gyro_radps", np.asarray(self.gyro_radps, dtype=float))

        _validate_sensor_triplet(self.timestamps_s, self.accel_mps2, self.gyro_radps, self.name)
        if self.timestamps_ns is not None:
            from .timing import validate_timestamps
            ns = np.asarray(self.timestamps_ns, dtype=np.int64)
            validate_timestamps(ns)
            if ns.shape != self.timestamps_s.shape:
                raise ValueError("timestamps_ns must have shape [N]")
            object.__setattr__(self, "timestamps_ns", ns)

    @property
    def t_utc(self):
        return self.timestamps_s

    @property
    def data_gyr(self):
        return self.gyro_radps

    @property
    def data_acc(self):
        return self.accel_mps2

    def select(self, time_start=None, time_end=None) -> IMUData:
        from .timing import interval_mask, timestamps_to_ns
        ns = self.timestamps_ns if self.timestamps_ns is not None else timestamps_to_ns(self.timestamps_s, "s")
        mask = interval_mask(ns, time_start, time_end)
        return IMUData(self.timestamps_s[mask], self.accel_mps2[mask], self.gyro_radps[mask],
                       self.name, self.frame_id, self.frequency_hz, dict(self.metadata), ns[mask])


@dataclass(frozen=True)
class LidarData:
    """
    Relative LiDAR motion; metadata identifies reference or independent scan motion.

    timestamps_s:
        End timestamp of every relative-pose interval with shape [N].

        Relative pose i spans

            scan_timestamps_s[i] -> scan_timestamps_s[i + 1]

        and is associated with

            timestamps_s[i] = scan_timestamps_s[i + 1].

        This end-time convention matches the calibration-event renderer.

    relative_poses_se3:
        Relative transforms with shape [N, 4, 4].

        The convention is

            T_L_previous_L_current = inverse(T_W_L_previous) @ T_W_L_current.

        Each matrix maps coordinates from the current LiDAR frame into the immediately preceding retained LiDAR frame.

    scan_timestamps_s:
        Absolute scan timestamps with shape [N + 1].

    fitness / inlier_rmse:
        Actual registration diagnostics for independent odometry; None for ground truth.
    """

    timestamps_s: np.ndarray
    relative_poses_se3: np.ndarray
    scan_timestamps_s: np.ndarray
    source_scan_paths: list[str] = field(default_factory=list)
    fitness: np.ndarray | None = None
    inlier_rmse: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamps_s", np.asarray(self.timestamps_s, dtype=float))
        object.__setattr__(self, "relative_poses_se3", np.asarray(self.relative_poses_se3, dtype=float))
        object.__setattr__(self, "scan_timestamps_s", np.asarray(self.scan_timestamps_s, dtype=float))

        if self.fitness is not None:
            object.__setattr__(self, "fitness", np.asarray(self.fitness, dtype=float))

        if self.inlier_rmse is not None:
            object.__setattr__(self, "inlier_rmse", np.asarray(self.inlier_rmse, dtype=float))

        _validate_lidar(self)


def _validate_sensor_triplet(timestamps_s: np.ndarray, accel_mps2: np.ndarray, gyro_radps: np.ndarray, name: str) -> None:
    """Validate one IMU timestamp/accelerometer/gyroscope triplet."""

    if timestamps_s.ndim != 1:
        raise ValueError(f"{name}: timestamps_s must have shape [N].")

    if accel_mps2.shape != (timestamps_s.size, 3):
        raise ValueError(f"{name}: accel_mps2 must have shape [N, 3].")

    if gyro_radps.shape != (timestamps_s.size, 3):
        raise ValueError(f"{name}: gyro_radps must have shape [N, 3].")

    if not np.all(np.isfinite(timestamps_s)):
        raise ValueError(f"{name}: timestamps_s contains non-finite values.")

    if not np.all(np.isfinite(accel_mps2)):
        raise ValueError(f"{name}: accel_mps2 contains non-finite values.")

    if not np.all(np.isfinite(gyro_radps)):
        raise ValueError(f"{name}: gyro_radps contains non-finite values.")

    if timestamps_s.size > 1 and np.any(np.diff(timestamps_s) <= 0.0):
        raise ValueError(f"{name}: timestamps_s must be strictly increasing.")


def _validate_lidar(lidar_data: LidarData) -> None:
    """Validate one relative LiDAR-pose sequence."""

    poses = lidar_data.relative_poses_se3

    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("relative_poses_se3 must have shape [N, 4, 4].")

    number_relative_poses = poses.shape[0]

    if lidar_data.timestamps_s.shape != (number_relative_poses,):
        raise ValueError("timestamps_s must have shape [N].")

    if lidar_data.scan_timestamps_s.shape != (number_relative_poses + 1,):
        raise ValueError("scan_timestamps_s must have shape [N + 1].")

    if number_relative_poses == 0:
        raise ValueError("At least one relative LiDAR pose is required.")

    if not np.all(np.isfinite(poses)):
        raise ValueError("relative_poses_se3 contains non-finite values.")

    if not np.all(np.isfinite(lidar_data.timestamps_s)):
        raise ValueError("timestamps_s contains non-finite values.")

    if not np.all(np.isfinite(lidar_data.scan_timestamps_s)):
        raise ValueError("scan_timestamps_s contains non-finite values.")

    if np.any(np.diff(lidar_data.scan_timestamps_s) <= 0.0):
        raise ValueError("scan_timestamps_s must be strictly increasing.")

    expected_end_timestamps = lidar_data.scan_timestamps_s[1:]

    if not np.allclose(lidar_data.timestamps_s, expected_end_timestamps, rtol=0.0, atol=1e-12):
        raise ValueError("timestamps_s must contain the end timestamp of every relative LiDAR-pose interval.")

    if lidar_data.fitness is not None:
        if lidar_data.fitness.shape != (number_relative_poses,):
            raise ValueError("fitness must have shape [N].")

        if not np.all(np.isfinite(lidar_data.fitness)):
            raise ValueError("fitness contains non-finite values.")

    if lidar_data.inlier_rmse is not None:
        if lidar_data.inlier_rmse.shape != (number_relative_poses,):
            raise ValueError("inlier_rmse must have shape [N].")

        if not np.all(np.isfinite(lidar_data.inlier_rmse)):
            raise ValueError("inlier_rmse contains non-finite values.")

