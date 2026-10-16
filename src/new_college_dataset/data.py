"""Dataclasses that carry synchronized New College sensor streams."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Tuple

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class IMUData:
    """Synchronized accelerometer and gyroscope measurements for one IMU.

    Args:
        timestamps_s: Monotonic timestamps in seconds with shape ``(N,)``.
        accel_mps2: Accelerometer samples in meters per second squared, shape
            ``(N, 3)``.
        gyro_radps: Gyroscope samples in radians per second, shape ``(N, 3)``.
        name: Human-readable stream name.
        frame_id: Optional sensor frame name from the source data.
        frequency_hz: Nominal resampled frequency in hertz.
        metadata: Extra loader details such as source path and column names.
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
    """Pairwise LiDAR odometry estimates as relative SE(3) matrices.

    Args:
        timestamps_s: Timestamps in seconds for each relative-pose interval,
            usually midpoint times, shape ``(N,)``.
        relative_poses_se3: Relative transforms from scan ``i`` to scan
            ``i + 1`` with shape ``(N, 4, 4)``.
        scan_timestamps_s: Optional original scan timestamps with shape
            ``(N + 1,)``. Use these for exact velocity durations when available.
        source_scan_paths: Ordered source PCD paths used to compute the poses.
        fitness: ICP fitness values, shape ``(N,)``.
        inlier_rmse: ICP inlier RMSE values, shape ``(N,)``.
        metadata: Extra loader and ICP settings.
    """

    timestamps_s: np.ndarray
    relative_poses_se3: np.ndarray
    scan_timestamps_s: np.ndarray | None = None
    source_scan_paths: list[str] = field(default_factory=list)
    fitness: np.ndarray | None = None
    inlier_rmse: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamps_s", np.asarray(self.timestamps_s, dtype=float))
        object.__setattr__(
            self,
            "relative_poses_se3",
            np.asarray(self.relative_poses_se3, dtype=float),
        )
        if self.scan_timestamps_s is not None:
            object.__setattr__(
                self,
                "scan_timestamps_s",
                np.asarray(self.scan_timestamps_s, dtype=float),
            )
        if self.fitness is not None:
            object.__setattr__(self, "fitness", np.asarray(self.fitness, dtype=float))
        if self.inlier_rmse is not None:
            object.__setattr__(self, "inlier_rmse", np.asarray(self.inlier_rmse, dtype=float))
        _validate_lidar(self)


def _validate_sensor_triplet(
    timestamps_s: np.ndarray,
    accel_mps2: np.ndarray,
    gyro_radps: np.ndarray,
    name: str,
) -> None:
    if timestamps_s.ndim != 1:
        raise ValueError(f"{name}: timestamps_s must have shape (N,).")
    if accel_mps2.shape != (timestamps_s.size, 3):
        raise ValueError(f"{name}: accel_mps2 must have shape (N, 3).")
    if gyro_radps.shape != (timestamps_s.size, 3):
        raise ValueError(f"{name}: gyro_radps must have shape (N, 3).")
    if timestamps_s.size and not np.all(np.isfinite(timestamps_s)):
        raise ValueError(f"{name}: timestamps_s contains non-finite values.")
    if timestamps_s.size > 1 and np.any(np.diff(timestamps_s) <= 0.0):
        raise ValueError(f"{name}: timestamps_s must be strictly increasing.")


def _validate_lidar(lidar_data: LidarData) -> None:
    poses = lidar_data.relative_poses_se3
    n_poses = poses.shape[0] if poses.ndim == 3 else -1
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("relative_poses_se3 must have shape (N, 4, 4).")
    if lidar_data.timestamps_s.shape != (n_poses,):
        raise ValueError("timestamps_s must have shape (N,) for N relative poses.")
    if n_poses and not np.all(np.isfinite(lidar_data.timestamps_s)):
        raise ValueError("timestamps_s contains non-finite values.")
    if n_poses > 1 and np.any(np.diff(lidar_data.timestamps_s) <= 0.0):
        raise ValueError("timestamps_s must be strictly increasing.")
    if lidar_data.scan_timestamps_s is not None:
        if lidar_data.scan_timestamps_s.shape != (n_poses + 1,):
            raise ValueError("scan_timestamps_s must have shape (N + 1,).")
        if np.any(np.diff(lidar_data.scan_timestamps_s) <= 0.0):
            raise ValueError("scan_timestamps_s must be strictly increasing.")



def import_true_trajectory(file_path: Any) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Import a timestamped ground-truth SE(3) trajectory from a CSV file.

    The expected column order is:

        sec, nsec, x, y, z, qx, qy, qz, qw

    The returned poses are T_W_B transformations, assuming the position and
    quaternion describe the body or sensor pose in the world frame.

    Returns
    -------
    trajectory_timestamps
        Array of shape (N,) containing timestamps in seconds.

    trajectory_poses
        Array of shape (N, 4, 4) containing SE(3) pose matrices.
    '''

    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"Trajectory file does not exist: {file_path}")

    trajectory_data = np.genfromtxt(
        file_path,
        delimiter=",",
        comments="#",
        dtype=float,
    )

    if trajectory_data.size == 0:
        raise ValueError(f"Trajectory file is empty: {file_path}")

    if trajectory_data.ndim == 1:
        trajectory_data = trajectory_data[None, :]

    if trajectory_data.shape[1] != 9:
        raise ValueError(
            "Expected 9 columns: sec, nsec, x, y, z, qx, qy, qz, qw; "
            f"received {trajectory_data.shape[1]}"
        )

    if not np.all(np.isfinite(trajectory_data)):
        raise ValueError("Trajectory file contains non-finite values")

    seconds = trajectory_data[:, 0]
    nanoseconds = trajectory_data[:, 1]
    positions = trajectory_data[:, 2:5]
    quaternions = trajectory_data[:, 5:9]  # scipy convention: xyzw

    trajectory_timestamps = seconds + nanoseconds * 1e-9

    if np.any(np.diff(trajectory_timestamps) <= 0):
        raise ValueError("Trajectory timestamps must be strictly increasing")

    quaternion_norms = np.linalg.norm(quaternions, axis=1)

    if np.any(quaternion_norms < 1e-12):
        raise ValueError("Trajectory contains a zero-norm quaternion")

    # Normalize to suppress small numerical deviations in the source file.
    quaternions = quaternions / quaternion_norms[:, None]

    rotations = Rotation.from_quat(quaternions).as_matrix()

    number_poses = len(trajectory_timestamps)
    trajectory_poses = np.repeat(
        np.eye(4, dtype=float)[None, :, :],
        number_poses,
        axis=0,
    )

    trajectory_poses[:, :3, :3] = rotations
    trajectory_poses[:, :3, 3] = positions

    return trajectory_timestamps, trajectory_poses