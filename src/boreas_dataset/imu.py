"""Boreas-RT IMU CSV loaders. Native measurements and acquisition gaps are preserved."""

from pathlib import Path
import csv
import warnings

import numpy as np

from .data import IMUData
from .timing import interval_mask, timestamp_statistics, timestamps_ns_to_s, timestamps_to_ns, validate_timestamps


# source: relative path, timestamp unit, physical coordinate frame
IMU_SOURCES = {
    "dmu": ("imu/dmu_imu.csv", "ns", "dmu"),
    "dmu_infilled": ("imu/dmu_imu_infilled.csv", "ns", "dmu"),
    "aeva": ("imu/aeva_imu.csv", "us", "aeva_imu"),
    "applanix": ("applanix/imu.csv", "s", "applanix"),
}
IMU_COLUMNS = ("time", "wx", "wy", "wz", "ax", "ay", "az")
APPLANIX_COLUMNS = ("GPSTime", "angvel_z", "angvel_y", "angvel_x", "accelz", "accely", "accelx")


def discover_imu_files(dataset_root: str | Path) -> list[Path]:
    """List supported IMU CSVs in a sequence, without loading measurements."""
    root = Path(dataset_root).expanduser()
    return [root / spec[0] for spec in IMU_SOURCES.values() if (root / spec[0]).is_file()]


def load_imu_file(path: str | Path, *, source: str | None = None, time_start=None, time_end=None,
                  name: str | None = None, frame_id: str | None = None) -> IMUData:
    """Load SI measurements with UTC seconds and exact integer nanoseconds.

    DMU and Aeva acceleration includes gravity. Applanix is a correlated reference
    source; its provided acceleration is preserved without gravity correction.
    Parsing streams the CSV; only requested interval measurements are retained.
    """
    path = Path(path).expanduser()
    if source is None:
        source = next((key for key, spec in IMU_SOURCES.items() if Path(spec[0]).name == path.name), None)
    if source not in IMU_SOURCES:
        raise ValueError(f"Specify one of {tuple(IMU_SOURCES)} for IMU source")
    _, unit, frame = IMU_SOURCES[source]
    expected = APPLANIX_COLUMNS if source == "applanix" else IMU_COLUMNS
    if source == "applanix":
        warnings.warn("Applanix IMU contributes to ground truth and is not independent calibration input", UserWarning, stacklevel=2)
    # Validate bounds even for an empty file/window.
    interval_mask(np.empty(0, dtype=np.int64), time_start, time_end)
    bounds = [None if t is None else timestamps_to_ns([t], "s")[0] for t in (time_start, time_end)]
    stamps, measurements = [], []
    previous = None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        if tuple(x.strip() for x in header) != expected:
            raise ValueError(f"Unexpected IMU CSV header in {path}: {header}; expected {expected}")
        for line, row in enumerate(reader, 2):
            if not row:
                continue
            if len(row) != 7:
                raise ValueError(f"Expected seven IMU fields at {path}:{line}")
            stamp = timestamps_to_ns([row[0]], unit)[0]
            if previous is not None and stamp <= previous:
                raise ValueError(f"IMU timestamps must be strictly increasing at {path}:{line}")
            previous = stamp
            if (bounds[0] is not None and stamp < bounds[0]) or (bounds[1] is not None and stamp > bounds[1]):
                continue
            values = np.asarray(row[1:], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite IMU measurements at {path}:{line}")
            stamps.append(stamp)
            measurements.append(values)
    ns = np.asarray(stamps, dtype=np.int64)
    validate_timestamps(ns)
    values = np.asarray(measurements, dtype=float).reshape(-1, 6)
    if source == "applanix":
        values = values[:, [2, 1, 0, 5, 4, 3]]
    stats = timestamp_statistics(ns)
    return IMUData(timestamps_ns_to_s(ns), values[:, 3:6], values[:, :3], name or source, frame_id or frame,
                   stats["frequency_hz"], {"source": source, "source_path": str(path), "timestamp_unit": unit,
                   "gyro_unit": "rad/s", "accel_unit": "m/s^2", "resampled": source == "dmu_infilled",
                   "independent_of_ground_truth": source != "applanix", "source_columns": expected}, ns)


def load_imu(sequence_root: str | Path, source: str = "dmu", *, time_start=None, time_end=None) -> IMUData:
    """Load an explicitly chosen source; raw DMU is always the default."""
    if source not in IMU_SOURCES:
        raise ValueError(f"Unknown IMU source {source!r}; choose from {tuple(IMU_SOURCES)}")
    return load_imu_file(Path(sequence_root) / IMU_SOURCES[source][0], source=source, time_start=time_start, time_end=time_end)
