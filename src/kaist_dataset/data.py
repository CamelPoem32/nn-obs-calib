"""KAIST dataset containers and low-level trajectory/calibration importers."""

from __future__ import annotations

from dataclasses import dataclass, field
from html import unescape
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamps_s", np.asarray(self.timestamps_s, dtype=float))
        object.__setattr__(self, "accel_mps2", np.asarray(self.accel_mps2, dtype=float))
        object.__setattr__(self, "gyro_radps", np.asarray(self.gyro_radps, dtype=float))

        _validate_sensor_triplet(self.timestamps_s, self.accel_mps2, self.gyro_radps, self.name)


@dataclass(frozen=True)
class LidarData:
    """
    Consecutive relative LiDAR poses derived from absolute scan-to-map poses.

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
        Optional scan-to-map diagnostics associated with the endpoint absolute scan pose of each relative interval.
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

    if timestamps_s.size == 0:
        raise ValueError(f"{name}: at least one IMU sample is required.")

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


def timestamps_ns_to_s(timestamps_ns: np.ndarray) -> np.ndarray:
    """Convert integer nanosecond timestamps to floating-point seconds."""

    timestamps_ns = np.asarray(timestamps_ns)

    if timestamps_ns.ndim != 1:
        raise ValueError("timestamps_ns must have shape [N].")

    try:
        timestamps_ns = timestamps_ns.astype(np.int64, copy=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("timestamps_ns must contain valid int64 nanosecond timestamps.") from error

    seconds = timestamps_ns // 1_000_000_000
    nanoseconds = timestamps_ns % 1_000_000_000

    return seconds.astype(np.float64) + nanoseconds.astype(np.float64) * 1e-9


def import_true_trajectory(file_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Import a KAIST global_pose.csv trajectory.

    The expected row is:

        timestamp_ns,
        P00,P01,P02,P03,
        P10,P11,P12,P13,
        P20,P21,P22,P23

    The stored transform is returned without inversion, rebasing, recentering, or SO(3) projection.
    """

    source_path = Path(file_path).expanduser()

    if not source_path.is_file():
        raise FileNotFoundError(f"Trajectory file does not exist: {source_path}")

    try:
        trajectory_data = pd.read_csv(source_path, header=None, comment="#", skip_blank_lines=True, dtype={0: "int64"})
    except (TypeError, ValueError) as error:
        raise ValueError(f"Could not parse KAIST trajectory CSV: {source_path}") from error

    trajectory_data = trajectory_data.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if trajectory_data.empty:
        raise ValueError(f"Trajectory file is empty: {source_path}")

    if trajectory_data.shape[1] != 13:
        raise ValueError(f"Expected 13 columns: timestamp_ns followed by a 3x4 pose matrix; received {trajectory_data.shape[1]}.")

    timestamps_ns = trajectory_data.iloc[:, 0].to_numpy(dtype=np.int64)
    pose_values = trajectory_data.iloc[:, 1:13].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

    if not np.all(np.isfinite(pose_values)):
        raise ValueError("Trajectory file contains non-finite pose values.")

    trajectory_timestamps = timestamps_ns_to_s(timestamps_ns)

    if trajectory_timestamps.size > 1 and np.any(np.diff(trajectory_timestamps) <= 0.0):
        raise ValueError("Trajectory timestamps must be strictly increasing.")

    number_poses = trajectory_timestamps.size
    trajectory_poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], number_poses, axis=0)
    trajectory_poses[:, :3, :] = pose_values.reshape(-1, 3, 4)

    return trajectory_timestamps, trajectory_poses


_FLOAT_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def import_extrinsic_calibration(file_path: str | Path) -> np.ndarray:
    """
    Import one KAIST Vehicle2*.txt extrinsic as a 4x4 matrix.

    The supplied rotation and translation are returned directly. Frame-direction semantics remain the responsibility of the caller.
    """

    source_path = Path(file_path).expanduser()

    if not source_path.is_file():
        raise FileNotFoundError(f"Calibration file does not exist: {source_path}")

    text = unescape(source_path.read_text(encoding="utf-8", errors="replace"))

    rotation_values = _numbers_after_prefix(text, "R:", expected_count=9)
    translation_values = _numbers_after_prefix(text, "T:", expected_count=3)

    rotation = np.asarray(rotation_values, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation_values, dtype=np.float64)

    if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
        raise ValueError(f"Calibration file contains non-finite values: {source_path}")

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation
    T[:3, 3] = translation

    return T


def _numbers_after_prefix(text: str, prefix: str, *, expected_count: int) -> list[float]:
    """Extract numeric values from the first line beginning with prefix."""

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line.startswith(prefix):
            continue

        values = [float(token) for token in _FLOAT_PATTERN.findall(line[len(prefix):])]

        if len(values) != expected_count:
            raise ValueError(f"Expected {expected_count} numeric values after {prefix!r}; received {len(values)}.")

        return values

    raise ValueError(f"Could not find a {prefix!r} line in calibration file.")