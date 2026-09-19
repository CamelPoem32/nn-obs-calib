"""KAIST Xsens IMU loading and optional resampling utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import IMUData, timestamps_ns_to_s


IMU_EXTENSIONS = {".csv"}

KAIST_IMU_COLUMNS_V1 = (
    "timestamp_ns",
    "quat_x", "quat_y", "quat_z", "quat_w",
    "euler_x_deg", "euler_y_deg", "euler_z_deg",
)

KAIST_IMU_COLUMNS_V2 = (
    "timestamp_ns",
    "quat_x", "quat_y", "quat_z", "quat_w",
    "euler_x_deg", "euler_y_deg", "euler_z_deg",
    "gyro_x_radps", "gyro_y_radps", "gyro_z_radps",
    "accel_x_mps2", "accel_y_mps2", "accel_z_mps2",
    "mag_x", "mag_y", "mag_z",
)


def discover_imu_files(dataset_root: str | Path) -> list[Path]:
    """Find likely KAIST Xsens IMU CSV files below a supplied path."""

    root = Path(dataset_root).expanduser()

    if root.is_file():
        return [root] if root.suffix.lower() in IMU_EXTENSIONS else []

    if not root.exists():
        return []

    candidates: list[Path] = []

    for path in root.rglob("*.csv"):
        path_name = path.name.lower()

        if "imu" in path_name or "xsens" in path_name:
            candidates.append(path)

    return sorted(candidates)


def load_imu_file(path: str | Path, *, target_frequency_hz: float | None = None, name: str | None = None, frame_id: str | None = None) -> IMUData:
    """
    Load one KAIST Xsens IMU CSV.

    By default the native timestamps and measurements are preserved after sorting and duplicate-timestamp removal.

    Set target_frequency_hz to a positive value only when explicit uniform resampling is desired.

    KAIST Ver2 rows are expected to contain:

        timestamp_ns,
        qx,qy,qz,qw,
        euler_x,euler_y,euler_z,
        gyro_x,gyro_y,gyro_z,
        accel_x,accel_y,accel_z,
        mag_x,mag_y,mag_z.
    """

    source_path = Path(path).expanduser()

    if not source_path.is_file():
        raise FileNotFoundError(f"IMU file does not exist: {source_path}")

    table = _read_kaist_imu_table(source_path)
    number_columns = table.shape[1]

    if number_columns == len(KAIST_IMU_COLUMNS_V1):
        raise ValueError(f"KAIST IMU file {source_path} uses the 8-column Ver1 format, which does not contain gyroscope or accelerometer measurements.")

    if number_columns != len(KAIST_IMU_COLUMNS_V2):
        raise ValueError(f"Expected either 8 KAIST IMU Ver1 columns or 17 KAIST IMU Ver2 columns; received {number_columns} in {source_path}.")

    table.columns = KAIST_IMU_COLUMNS_V2

    timestamps_ns = _integer_timestamp_column(table["timestamp_ns"], source_path)
    timestamps_s = timestamps_ns_to_s(timestamps_ns)

    gyro_radps = _finite_measurement_matrix(table, ("gyro_x_radps", "gyro_y_radps", "gyro_z_radps"), "gyroscope")
    accel_mps2 = _finite_measurement_matrix(table, ("accel_x_mps2", "accel_y_mps2", "accel_z_mps2"), "accelerometer")

    # Keep every IMU channel aligned while sorting. Duplicate source timestamps are removed stably, preserving the first occurrence.
    order = np.argsort(timestamps_ns, kind="stable")
    timestamps_ns = timestamps_ns[order]
    timestamps_s = timestamps_s[order]
    accel_mps2 = accel_mps2[order]
    gyro_radps = gyro_radps[order]

    _, unique_indices = np.unique(timestamps_ns, return_index=True)
    unique_indices = np.sort(unique_indices)

    timestamps_s = timestamps_s[unique_indices]
    accel_mps2 = accel_mps2[unique_indices]
    gyro_radps = gyro_radps[unique_indices]

    source_frequency_hz = _estimate_frequency(timestamps_s)

    imu = IMUData(
        timestamps_s=timestamps_s,
        accel_mps2=accel_mps2,
        gyro_radps=gyro_radps,
        name=name or source_path.stem,
        frame_id=frame_id,
        frequency_hz=source_frequency_hz,
        metadata={
            "source_path": str(source_path),
            "raw_format": "KAIST Xsens IMU Ver2",
            "timestamp_unit": "nanoseconds",
            "timestamp_column": "timestamp_ns",
            "gyro_columns": ["gyro_x_radps", "gyro_y_radps", "gyro_z_radps"],
            "accel_columns": ["accel_x_mps2", "accel_y_mps2", "accel_z_mps2"],
            "orientation_columns": ["quat_x", "quat_y", "quat_z", "quat_w"],
            "euler_columns": ["euler_x_deg", "euler_y_deg", "euler_z_deg"],
            "magnetometer_columns": ["mag_x", "mag_y", "mag_z"],
            "source_frequency_hz_estimate": source_frequency_hz,
            "resampled": False,
        },
    )

    if target_frequency_hz is None:
        return imu

    return resample_imu(imu, target_frequency_hz=target_frequency_hz)


def load_imus(dataset_root: str | Path, *, target_frequency_hz: float | None = None, max_files: int | None = None) -> dict[str, IMUData]:
    """Load every discoverable KAIST IMU stream below a supplied directory."""

    files = discover_imu_files(dataset_root)

    if max_files is not None:
        if max_files < 0:
            raise ValueError("max_files must be non-negative when provided.")

        files = files[:max_files]

    return {f"{path.parent.name}_{path.stem}": load_imu_file(path, target_frequency_hz=target_frequency_hz, name=path.stem) for path in files}


def resample_imu(imu_data: IMUData, *, target_frequency_hz: float = 100.0) -> IMUData:
    """Resample one IMU stream onto a uniform time grid using linear interpolation."""

    if not np.isfinite(target_frequency_hz) or target_frequency_hz <= 0.0:
        raise ValueError("target_frequency_hz must be finite and positive.")

    if imu_data.timestamps_s.size < 2:
        raise ValueError("At least two IMU samples are required for resampling.")

    sample_period_s = 1.0 / float(target_frequency_hz)
    start_time_s = float(imu_data.timestamps_s[0])
    stop_time_s = float(imu_data.timestamps_s[-1])

    number_samples = int(np.floor((stop_time_s - start_time_s) / sample_period_s)) + 1
    timestamps_s = start_time_s + np.arange(number_samples, dtype=float) * sample_period_s
    timestamps_s = timestamps_s[timestamps_s <= stop_time_s]

    accel_mps2 = np.column_stack([np.interp(timestamps_s, imu_data.timestamps_s, imu_data.accel_mps2[:, axis]) for axis in range(3)])
    gyro_radps = np.column_stack([np.interp(timestamps_s, imu_data.timestamps_s, imu_data.gyro_radps[:, axis]) for axis in range(3)])

    metadata = dict(imu_data.metadata)
    metadata.update({"source_frequency_hz_estimate": _estimate_frequency(imu_data.timestamps_s), "resampled": True, "target_frequency_hz": float(target_frequency_hz)})

    return IMUData(timestamps_s=timestamps_s, accel_mps2=accel_mps2, gyro_radps=gyro_radps, name=imu_data.name, frame_id=imu_data.frame_id, frequency_hz=float(target_frequency_hz), metadata=metadata)


def _read_kaist_imu_table(path: Path) -> pd.DataFrame:
    """Read the headerless KAIST Xsens CSV without coercing timestamps to float."""

    try:
        table = pd.read_csv(path, header=None, comment="#", skip_blank_lines=True, dtype={0: "int64"})
    except (ValueError, TypeError) as error:
        raise ValueError(f"Could not parse KAIST IMU CSV: {path}") from error

    table = table.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if table.empty:
        raise ValueError(f"IMU file contains no usable rows: {path}")

    return table


def _integer_timestamp_column(values: pd.Series, source_path: Path) -> np.ndarray:
    """Return a validated int64 nanosecond timestamp vector."""

    try:
        timestamps_ns = values.to_numpy(dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Invalid int64 timestamps in IMU file: {source_path}") from error

    if timestamps_ns.ndim != 1 or timestamps_ns.size == 0:
        raise ValueError(f"IMU file has no timestamps: {source_path}")

    return timestamps_ns


def _finite_measurement_matrix(table: pd.DataFrame, columns: tuple[str, str, str], signal_name: str) -> np.ndarray:
    """Return one finite [N, 3] measurement matrix."""

    values = table.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError(f"{signal_name} columns must contain finite numeric values.")

    return values


def _estimate_frequency(timestamps_s: np.ndarray) -> float | None:
    """Estimate sampling frequency from the median positive time increment."""

    if timestamps_s.size < 2:
        return None

    deltas = np.diff(timestamps_s)
    deltas = deltas[deltas > 0.0]

    if deltas.size == 0:
        return None

    return float(1.0 / np.median(deltas))