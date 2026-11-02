"""Adapt obscalib windows or source LiDAR/IMU arrays to ROS-free LI-Init inputs.

This module does not alter extrinsics or re-register scans. Source LiDAR transforms
are consecutive, relative, end-frame-to-start-frame SE(3) increments:
    delta_T[k] = inverse(T_WL[k]) @ T_WL[k + 1].
IMU angular velocity is in rad/s and acceleration in m/s^2. All clocks use
seconds on a common epoch or a common relative-window origin.

``WindowSample`` stores LiDAR end times in ``timestamps`` and interval starts
in ``interval_start_timestamps``. These MUST NOT be reconstructed with an
assumed constant LiDAR scan period. We use duck typing to avoid importing
obscalib's training pipeline into a classical baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from .structures import IMUSeries, LidarIncrements


@dataclass(frozen=True)
class CalibrationReference:
    """Reference LiDAR -> IMU extrinsics in LI-Init's coordinate convention.

    ``R_LI`` maps LiDAR vectors to IMU vectors. ``p_IL`` is the IMU origin
    expressed in LiDAR coordinates; ``time_offset`` means
    ``t_imu = t_lidar + time_offset``. This object is for EVALUATION ONLY,
    never passed to the numerical estimator.
    """

    R_LI: np.ndarray
    p_IL: np.ndarray | None = None
    time_offset: float | None = None

    def __post_init__(self) -> None:
        R = np.asarray(self.R_LI, dtype=np.float64)
        if R.shape != (3, 3) or not np.all(np.isfinite(R)) or not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1.0, atol=1e-5):
            raise ValueError("R_LI must be a valid 3x3 rotation matrix")
        object.__setattr__(self, "R_LI", R.copy())
        if self.p_IL is not None:
            p = np.asarray(self.p_IL, dtype=np.float64)
            if p.shape != (3,) or not np.all(np.isfinite(p)):
                raise ValueError("p_IL must be a finite 3-vector")
            object.__setattr__(self, "p_IL", p.copy())
        if self.time_offset is not None and not np.isfinite(self.time_offset):
            raise ValueError("time_offset must be finite")


def _numpy(value: Any) -> np.ndarray:
    """Convert CPU/GPU torch tensors or ordinary arrays without modifying data."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def make_imu_series(t: ArrayLike, data_gyr: ArrayLike, data_acc: ArrayLike | None = None) -> IMUSeries:
    """Adapt raw IMU timestamps, angular velocities and optional acceleration."""
    return IMUSeries(t=_numpy(t), data_gyr=_numpy(data_gyr), data_acc=None if data_acc is None else _numpy(data_acc))


def make_lidar_increments(t_start: ArrayLike, t_end: ArrayLike, delta_T: ArrayLike, *, increment_convention: Literal["start_to_end", "end_to_start"] = "start_to_end") -> LidarIncrements:
    """Adapt relative LiDAR SE(3) observations without integrating to poses."""
    return LidarIncrements(t_start=_numpy(t_start), t_end=_numpy(t_end), delta_T=_numpy(delta_T), increment_convention=increment_convention)


def window_sample_to_li_init(window: Any, *, gyro_key: str = "gyro", accel_key: str | None = "accel", lidar_key: str = "lidar", increment_convention: Literal["start_to_end", "end_to_start"] = "start_to_end") -> tuple[IMUSeries, LidarIncrements]:
    """Read one ``obscalib.data.structures.WindowSample`` without any augmentation.

    In dataset-produced samples timestamps are already relative to the same
    window start, including LiDAR ``interval_start_timestamps``. The current
    calibration and target labels are *intentionally ignored*.
    """
    streams = window.streams
    gyro = streams[gyro_key]
    accel = streams[accel_key] if accel_key is not None else None
    lidar = streams[lidar_key]
    if lidar.interval_start_timestamps is None:
        raise ValueError("LiDAR interval_start_timestamps are required; never infer them from end times")
    t_g = _numpy(gyro.timestamps)
    if accel is not None:
        t_a = _numpy(accel.timestamps)
        if t_g.shape != t_a.shape or not np.allclose(t_g, t_a, atol=1e-8, rtol=0):
            raise ValueError("Gyroscope and accelerometer timestamps differ; synchronize explicitly before adapting")
    imu = make_imu_series(t_g, gyro.values, None if accel is None else accel.values)
    increments = make_lidar_increments(lidar.interval_start_timestamps, lidar.timestamps, lidar.values, increment_convention=increment_convention)
    return imu, increments


def window_batch_to_li_init(batch: Any, index: int = 0, *, gyro_key: str = "gyro", accel_key: str | None = "accel", lidar_key: str = "lidar", increment_convention: Literal["start_to_end", "end_to_start"] = "start_to_end") -> tuple[IMUSeries, LidarIncrements]:
    """Extract one valid unpadded element from ``WindowBatch``.

    Uses the per-stream ``sample_mask``; padded timestamps and SE(3) matrices
    must never enter the numerical optimizer.
    """
    streams = batch.streams
    gyro = streams[gyro_key]
    accel = streams[accel_key] if accel_key is not None else None
    lidar = streams[lidar_key]

    def unpack(stream: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        mask = np.asarray(stream.sample_mask[index].detach().cpu() if hasattr(stream.sample_mask, "detach") else stream.sample_mask[index], dtype=bool)
        ts = _numpy(stream.timestamps[index])[mask]
        values = _numpy(stream.values[index])[mask]
        starts = None if stream.interval_start_timestamps is None else _numpy(stream.interval_start_timestamps[index])[mask]
        return ts, values, starts

    tg, wg, _ = unpack(gyro)
    if accel is not None:
        ta, aa, _ = unpack(accel)
        if tg.shape != ta.shape or not np.allclose(tg, ta, rtol=0, atol=1e-8):
            raise ValueError("Gyro/accel batch timestamps differ")
    else:
        aa = None
    tl_end, dT, tl_start = unpack(lidar)
    if tl_start is None:
        raise ValueError("LiDAR batch interval_start_timestamps are required")
    return make_imu_series(tg, wg, aa), make_lidar_increments(tl_start, tl_end, dT, increment_convention=increment_convention)


def kaist_records_to_li_init(imu_record: Any, lidar_record: Any, *, increment_convention: Literal["start_to_end", "end_to_start"] = "start_to_end") -> tuple[IMUSeries, LidarIncrements]:
    """Adapt existing ``kaist_dataset.imu.IMUData`` and LiDAR-relative record.

    Exact source attributes are ``timestamps_s``, ``gyro_radps``,
    ``accel_mps2``, ``scan_timestamps_s``, ``relative_poses_se3``.
    """
    imu = make_imu_series(imu_record.timestamps_s, imu_record.gyro_radps, imu_record.accel_mps2)
    t = _numpy(lidar_record.scan_timestamps_s)
    lidar = make_lidar_increments(t[:-1], t[1:], lidar_record.relative_poses_se3, increment_convention=increment_convention)
    return imu, lidar


def slice_li_init_window(imu: IMUSeries, lidar: LidarIncrements, t_start: float, t_end: float, *, imu_padding_s: float = 0.0, min_lidar_increments: int = 2) -> tuple[IMUSeries, LidarIncrements]:
    """Select a LiDAR-time window, preserving ONLY complete scan intervals.

    Optional IMU padding is made explicit; it is needed for searching an
    unknown temporal offset, but gives the baseline more context than the
    nominal LiDAR window. Report it when comparing against the NN.

    The output IMU and LiDAR timestamps retain the original shared origin.
    There is no interpolation, invented scan interval, or measurement noise.
    """
    if not (np.isfinite(t_start) and np.isfinite(t_end) and t_start < t_end and np.isfinite(imu_padding_s) and imu_padding_s >= 0):
        raise ValueError("Invalid window time or IMU padding")
    mask_l = (lidar.t_start >= t_start - 1e-9) & (lidar.t_end <= t_end + 1e-9)
    mask_i = (imu.t >= t_start - imu_padding_s - 1e-9) & (imu.t <= t_end + imu_padding_s + 1e-9)
    if mask_l.sum() < min_lidar_increments:
        raise ValueError(f"Only {mask_l.sum()} LiDAR increments in [{t_start}, {t_end}]")
    if mask_i.sum() < 2:
        raise ValueError("Not enough IMU samples in the selected window")
    return (
        make_imu_series(imu.t[mask_i], imu.data_gyr[mask_i], None if imu.data_acc is None else imu.data_acc[mask_i]),
        make_lidar_increments(lidar.t_start[mask_l], lidar.t_end[mask_l], lidar.delta_T[mask_l], increment_convention=lidar.increment_convention),
    )


def reference_from_body_calibrations(T_B_I: ArrayLike, T_B_L: ArrayLike, *, tau_imu_s: float | None = None, tau_lidar_s: float | None = None) -> CalibrationReference:
    """Convert body-frame sensor transforms to LI-Init's LiDAR -> IMU reference.

    T_B_I maps IMU coordinates to body; T_B_L maps LiDAR to body. Therefore
    T_I_L = inverse(T_B_I) @ T_B_L, with R_I_L = R_LI and the LiDAR origin
    expressed in IMU coordinates p_LI = T_I_L[:3, 3]. Our p_IL is the IMU
    origin in LiDAR axes = -R_LI.T @ p_LI.

    If corrected/common time = sensor timestamp + tau_sensor, then
    t_imu = t_lidar + (tau_lidar - tau_imu).
    """
    T_i, T_l = _numpy(T_B_I), _numpy(T_B_L)
    for label, T in (("T_B_I", T_i), ("T_B_L", T_l)):
        if T.shape != (4, 4) or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-7) or not np.all(np.isfinite(T)):
            raise ValueError(f"{label} must be a finite 4x4 homogeneous matrix")
        R = T[:3, :3]
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1., atol=1e-5):
            raise ValueError(f"{label} must contain a valid SO(3) rotation")
    T_I_L = np.linalg.inv(T_i) @ T_l
    if (tau_imu_s is None) != (tau_lidar_s is None):
        raise ValueError("Set both tau_imu_s and tau_lidar_s, or neither")
    return CalibrationReference(R_LI=T_I_L[:3, :3], p_IL=-(T_I_L[:3, :3].T @ T_I_L[:3, 3]), time_offset=None if tau_imu_s is None else float(tau_lidar_s - tau_imu_s))
