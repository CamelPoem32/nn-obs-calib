"""Measured sensor coverage and bounded raw-data statistics for research notebooks."""

from pathlib import Path

import numpy as np

from .lidar import LIDAR_SOURCES, discover_lidar_scans
from .timing import timestamp_statistics


def imu_statistics(imu) -> dict:
    """Signal variability is descriptive, not a stationary sensor-noise estimate."""
    timing = timestamp_statistics(imu.timestamps_ns)
    return {"source": imu.name, "frame": imu.frame_id, "samples": timing["count"],
            "duration_s": timing["duration_s"], "median_rate_hz": timing["frequency_hz"],
            "gaps": len(timing["gap_indices"]), "max_dt_ms": (timing["max_dt_s"] or 0) * 1000,
            "gyro_axis_mean_radps": imu.data_gyr.mean(0) if timing["count"] else np.full(3, np.nan),
            "gyro_axis_std_radps": imu.data_gyr.std(0) if timing["count"] else np.full(3, np.nan),
            "gyro_norm_p50_radps": float(np.median(np.linalg.norm(imu.data_gyr, axis=1))) if timing["count"] else None,
            "gyro_norm_p95_radps": float(np.percentile(np.linalg.norm(imu.data_gyr, axis=1), 95)) if timing["count"] else None,
            "accel_axis_mean_mps2": imu.data_acc.mean(0) if timing["count"] else np.full(3, np.nan),
            "accel_axis_std_mps2": imu.data_acc.std(0) if timing["count"] else np.full(3, np.nan),
            "accel_norm_p50_mps2": float(np.median(np.linalg.norm(imu.data_acc, axis=1))) if timing["count"] else None,
            "independent_of_ground_truth": imu.metadata["independent_of_ground_truth"],
            "resampled": imu.metadata["resampled"]}


def lidar_statistics(sequence_root: str | Path, *, source: str, sample_scans: int = 5,
                       max_points_per_scan: int | None = 20_000) -> dict:
    """Count all scans; measure geometry on deterministic subsets of a few clouds.

    Point counts come from complete file sizes. Range/FOV statistics come only
    from the stated sample; observed angular coverage is not a hardware FOV spec.
    """
    if not isinstance(sample_scans, int) or sample_scans < 1:
        raise ValueError("sample_scans must be positive")
    scans = discover_lidar_scans(sequence_root, source=source)
    timing = timestamp_statistics([scan.timestamp_ns for scan in scans])
    record_bytes = LIDAR_SOURCES[source][1] * 4
    sizes = np.array([scan.path.stat().st_size for scan in scans], dtype=np.int64)
    if np.any(sizes % record_bytes) or np.any(sizes == 0):
        raise ValueError(f"Empty or truncated {source} scans in inventory")
    selected = np.linspace(0, len(scans) - 1, min(len(scans), sample_scans), dtype=int)
    points = [scans[index].load(max_points=max_points_per_scan) for index in selected]
    xyz = np.concatenate([p[:, :3] for p in points]) if points else np.empty((0, 3))
    ranges = np.linalg.norm(xyz, axis=1)
    angles = np.sort(np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]), 2 * np.pi))
    coverage = 2 * np.pi - np.max(np.diff(np.r_[angles, angles[:1] + 2 * np.pi])) if len(angles) else 0
    return {"source": source, "scans": timing["count"], "duration_s": timing["duration_s"],
            "median_rate_hz": timing["frequency_hz"], "gaps": len(timing["gap_indices"]),
            "max_dt_ms": (timing["max_dt_s"] or 0) * 1000, "scan_time_reference": LIDAR_SOURCES[source][2],
            "total_bytes": int(sizes.sum()), "points_per_scan_p50": float(np.median(sizes // record_bytes)) if len(sizes) else 0,
            "points_per_scan_min": int((sizes // record_bytes).min()) if len(sizes) else 0,
            "points_per_scan_max": int((sizes // record_bytes).max()) if len(sizes) else 0,
            "sampled_scans": len(selected), "sampled_points": len(xyz),
            "range_p50_m": float(np.median(ranges)) if len(ranges) else None,
            "range_p95_m": float(np.percentile(ranges, 95)) if len(ranges) else None,
            "observed_azimuth_span_deg": float(np.rad2deg(coverage)),
            "point_time_unit": "s" if source == "velodyne" else "raw; apparent ns scale, unverified",
            "doppler_available": source == "aeva"}
