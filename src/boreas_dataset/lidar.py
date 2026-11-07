"""Lazy Velodyne scans and Applanix-derived ground-truth LiDAR motion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .calibration import validate_rigid_transform
from .data import LidarData
from .timing import (interval_mask, nearest_timestamps, relative_seconds, timestamps_ns_to_s,
                     timestamps_to_ns, validate_timestamps)


POSE_COLUMNS = ("GPSTime", "easting", "northing", "altitude", "vel_east", "vel_north", "vel_up",
                "roll", "pitch", "heading", "angvel_z", "angvel_y", "angvel_x")
LIDAR_SOURCES = {"velodyne": ("lidar", 6, "midpoint"), "aeva": ("aeva", 10, "start")}


@dataclass(frozen=True)
class LidarScan:
    """Unloaded entry; UTC scan time is Velodyne midpoint or Aeva scan start."""

    path: Path
    timestamp_ns: int
    source: str = "velodyne"
    frame_id: str = "lidar"

    @property
    def timestamp_s(self) -> float:
        return float(timestamps_ns_to_s([self.timestamp_ns])[0])

    def load(self, *, max_points: int | None = None) -> np.ndarray:
        return load_lidar_scan(self.path, source=self.source, max_points=max_points)

    @property
    def timestamp_reference(self) -> str:
        return LIDAR_SOURCES[self.source][2]


@dataclass(frozen=True)
class PoseTrajectory:
    """Sensor-to-ENU_ref poses, with the source frame named in metadata.

    The existing poses_T_W_L attribute is retained for both LiDAR sources and
    the Applanix reference; its matrices always map that source sensor to world.
    """

    timestamps_ns: np.ndarray
    poses_T_W_L: np.ndarray
    rpy_rad: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        ns = np.asarray(self.timestamps_ns, dtype=np.int64)
        T = np.asarray(self.poses_T_W_L, dtype=float)
        validate_timestamps(ns)
        if T.shape != (ns.size, 4, 4):
            raise ValueError("poses_T_W_L must have shape (N, 4, 4)")
        validate_rigid_transform(T)
        object.__setattr__(self, "timestamps_ns", ns)
        object.__setattr__(self, "poses_T_W_L", T)
        if self.rpy_rad is not None:
            rpy = np.asarray(self.rpy_rad, dtype=float)
            if rpy.shape != (ns.size, 3) or not np.isfinite(rpy).all():
                raise ValueError("rpy_rad must have shape (N, 3) and finite values")
            object.__setattr__(self, "rpy_rad", rpy)

    @property
    def timestamps_s(self):
        return timestamps_ns_to_s(self.timestamps_ns)

    @property
    def R(self):
        return self.poses_T_W_L[:, :3, :3]

    def select(self, time_start=None, time_end=None) -> PoseTrajectory:
        mask = interval_mask(self.timestamps_ns, time_start, time_end)
        return PoseTrajectory(self.timestamps_ns[mask], self.poses_T_W_L[mask],
                              None if self.rpy_rad is None else self.rpy_rad[mask], dict(self.metadata))

    def relative(self, *, frame_stride: int = 1, time_start=None, time_end=None) -> LidarData:
        """Consecutive retained frames, subsampled by stride; intervals use end times.

        T_lidar_i_lidar_j = inverse(T_world_lidar_i) @ T_world_lidar_j maps j to i.
        """
        if not isinstance(frame_stride, (int, np.integer)) or frame_stride < 1:
            raise ValueError("frame_stride must be a positive integer")
        selected = self.select(time_start, time_end)
        ns = selected.timestamps_ns[::frame_stride]
        T = selected.poses_T_W_L[::frame_stride]
        if ns.size < 2:
            raise ValueError("At least two retained poses are required")
        relative = np.linalg.solve(T[:-1], T[1:])
        times = timestamps_ns_to_s(ns)
        return LidarData(times[1:], relative, times, metadata={**self.metadata,
                         "motion_source": "Applanix/GNSS/INS ground truth", "independent_odometry": False,
                         "frame_stride": frame_stride, "relative_pose_convention": "T_lidar_i_lidar_j",
                         "relative_timestamp_semantics": "end_scan_timestamp"})

    def match(self, query_ns, *, tolerance_s: float = 0.0, interpolate: bool = False,
              max_gap_s: float | None = None) -> np.ndarray:
        """Exact matches by default; nearest matching requires explicit tolerance.

        Optional interpolation uses SciPy SLERP and linear ENU translation, never
        extrapolation. A finite positive maximum bracket gap must be supplied.
        """
        query = np.asarray(query_ns, dtype=np.int64)
        indices, _ = nearest_timestamps(self.timestamps_ns, query, tolerance_s=tolerance_s)
        if not interpolate or not query.size:
            return self.poses_T_W_L[indices].copy()
        if max_gap_s is None or not np.isfinite(max_gap_s) or max_gap_s <= 0:
            raise ValueError("Interpolation requires a finite positive max_gap_s")
        ns = self.timestamps_ns
        if np.any(query < ns[0]) or np.any(query > ns[-1]):
            raise ValueError("Pose extrapolation is not allowed")
        exact = ns[indices] == query
        output = self.poses_T_W_L[indices].copy()
        if np.all(exact):
            return output
        right = np.searchsorted(ns, query[~exact])
        left = right - 1
        if np.any((ns[right] - ns[left]) * 1e-9 > max_gap_s):
            raise ValueError("Pose interpolation bracket exceeds max_gap_s")
        t = relative_seconds(ns)
        q = relative_seconds(query[~exact], ns[0])
        output[~exact, :3, :3] = Slerp(t, Rotation.from_matrix(self.R))(q).as_matrix()
        output[~exact, :3, 3] = np.column_stack([np.interp(q, t, self.poses_T_W_L[:, axis, 3]) for axis in range(3)])
        return output


def load_lidar_poses(filepath: str | Path, *, time_start=None, time_end=None,
                     source: str = "velodyne") -> PoseTrajectory:
    """Read sensor pose CSV in microseconds with radians, matching pyboreas get_transform.

    Rotation equals roll_passive(r) @ pitch_passive(p) @ yaw_passive(y).
    SciPy's extrinsic xyz rotation with [r,p,y], transposed, gives this convention.
    """
    if source not in LIDAR_SOURCES:
        raise ValueError(f"Unknown LiDAR source: {source}")
    return load_reference_trajectory(filepath, timestamp_unit="us", sensor_frame=LIDAR_SOURCES[source][0],
                                     time_start=time_start, time_end=time_end)


def load_reference_trajectory(filepath: str | Path, *, timestamp_unit: str, sensor_frame: str,
                               time_start=None, time_end=None) -> PoseTrajectory:
    """Read the common 13-column pose prefix, including Applanix GPS reference CSV.

    The GPS reference has additional acceleration/geographic fields and seconds;
    sensor pose CSVs have microseconds. Rotation follows pyboreas get_transform.
    """
    path = Path(filepath).expanduser()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = tuple(x.strip() for x in next(reader, []))
        if header[:13] != POSE_COLUMNS:
            raise ValueError(f"Unexpected pose CSV header in {path}: {header}")
        rows = [row for row in reader if row]
    if any(len(row) != len(header) for row in rows):
        raise ValueError("Pose row field count does not match header")
    ns = timestamps_to_ns([row[0] for row in rows], timestamp_unit)
    validate_timestamps(ns)
    values = np.asarray([row[1:13] for row in rows], dtype=float).reshape(-1, 12)
    if not np.isfinite(values).all():
        raise ValueError("Pose CSV contains non-finite values")
    mask = interval_mask(ns, time_start, time_end)
    ns, values = ns[mask], values[mask]
    rpy = values[:, 6:9]
    T = np.repeat(np.eye(4)[None], ns.size, axis=0)
    if ns.size:
        T[:, :3, :3] = Rotation.from_euler("xyz", rpy).as_matrix().transpose(0, 2, 1)
        T[:, :3, 3] = values[:, :3]
    return PoseTrajectory(ns, T, rpy, {"source_path": str(path), "world_frame": "ENU_ref",
                          "sensor_frame": sensor_frame, "angles_unit": "rad", "ground_truth": True})


def discover_lidar_scans(sequence_root: str | Path, *, source: str = "velodyne",
                         time_start=None, time_end=None) -> list[LidarScan]:
    """Discover filename timestamps without reading binary point clouds."""
    if source not in LIDAR_SOURCES:
        raise ValueError(f"Unknown LiDAR source: {source}")
    folder = Path(sequence_root).expanduser() / LIDAR_SOURCES[source][0]
    if not folder.is_dir():
        raise FileNotFoundError(f"{source} directory is unavailable: {folder}")
    paths = sorted(folder.glob("*.bin"), key=lambda p: int(p.stem))
    ns = timestamps_to_ns([p.stem for p in paths], "us")
    validate_timestamps(ns)
    mask = interval_mask(ns, time_start, time_end)
    return [LidarScan(p, int(t), source, LIDAR_SOURCES[source][0]) for p, t, keep in zip(paths, ns, mask) if keep]


def load_lidar_scan(path: str | Path, *, source: str = "velodyne", max_points: int | None = None) -> np.ndarray:
    """Return raw float32 storage fields without altering point times or flags.

    For Velodyne, columns are [x,y,z,intensity,laser_id,offset_s]; the final column
    is signed seconds relative to the filename's scan midpoint.
    Optional deterministic evenly spaced subsampling uses a memory map and does
    not materialize the full scan. No filtering or motion compensation is applied.

    Aeva returns the original ten float32 storage words: XYZ, radial velocity,
    intensity, quality, reflectivity, raw point time, and two opaque flag words.
    The last two words are not treated as floating-point measurements. The real
    download's point-time values are not seconds; no unverified conversion or
    flag interpretation is applied. XYZ-based registration does not use them.
    """
    path = Path(path).expanduser()
    if source not in LIDAR_SOURCES:
        raise ValueError(f"Unknown LiDAR source: {source}")
    fields = LIDAR_SOURCES[source][1]
    record_size = 4 * fields
    size = path.stat().st_size
    if size == 0 or size % record_size:
        raise ValueError(f"Empty or truncated {source} file: {path} ({size} bytes; expected a multiple of {record_size})")
    if max_points is not None and (not isinstance(max_points, (int, np.integer)) or max_points <= 0):
        raise ValueError("max_points must be a positive integer")
    count = size // record_size
    if max_points is not None and count > max_points:
        raw = np.memmap(path, dtype="<f4", mode="r", shape=(count, fields))
        points = np.array(raw[np.linspace(0, count - 1, max_points, dtype=int)])
        del raw
    else:
        points = np.fromfile(path, dtype="<f4").reshape(-1, fields)
    if not np.isfinite(points[:, :8] if source == "aeva" else points).all():
        raise ValueError(f"Non-finite measurement fields in {source} scan: {path}")
    return points


def iter_lidar_scans(scans, *, max_points=None):
    """Yield (index entry, point array), holding one decoded scan at a time."""
    for scan in scans:
        yield scan, scan.load(max_points=max_points)


def motion_magnitudes(relative_poses) -> tuple[np.ndarray, np.ndarray]:
    """Translation norm in metres and SO(3) geodesic angle in radians."""
    T = validate_rigid_transform(relative_poses)
    return np.linalg.norm(T[..., :3, 3], axis=-1), Rotation.from_matrix(T[..., :3, :3]).magnitude()


def accumulate_lidar_scans(scans, trajectory: PoseTrajectory, *, points_per_scan: int = 10_000,
                           tolerance_s: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Small ground-truth-aligned cloud and trajectory in the first scan frame.

    Returns XYZ (M,3), T_first_lidar (N,4,4). No registration, deskew or SLAM.
    """
    scans = list(scans)
    if not scans:
        raise ValueError("Select at least one scan")
    poses = trajectory.match([scan.timestamp_ns for scan in scans], tolerance_s=tolerance_s)
    local = np.linalg.solve(poses[0], poses)
    clouds = [points[:, :3] @ T[:3, :3].T + T[:3, 3]
              for (_, points), T in zip(iter_lidar_scans(scans, max_points=points_per_scan), local)]
    return np.concatenate(clouds), local


def load_lidar_relative_poses(filepath: str | Path, *, frame_stride: int = 1,
                              time_start=None, time_end=None) -> LidarData:
    """Load Applanix-derived relative ground truth with the existing LidarData API."""
    return load_lidar_poses(filepath, time_start=time_start, time_end=time_end).relative(frame_stride=frame_stride)
