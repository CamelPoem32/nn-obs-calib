"""IMU loading and resampling utilities for the New College dataset."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data import IMUData


##################################################
# Supported files and column conventions
##################################################

IMU_EXTENSIONS = {".csv", ".txt", ".tsv", ".log", ".data"}

TIMESTAMP_ALIASES = (
    "timestamp",
    "time",
    "time_s",
    "stamp",
    "t",
    "rosbagtimestamp",
    "header.stamp",
)
SECOND_ALIASES = ("sec", "secs", "second", "seconds")
NANOSECOND_ALIASES = ("nsec", "nsecs", "nanosec", "nanosecs", "nanosecond", "nanoseconds")

ACCEL_ALIASES = (
    ("accel_x", "accel_y", "accel_z"),
    ("acc_x", "acc_y", "acc_z"),
    ("ax", "ay", "az"),
    ("linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z"),
    ("imu.linear_acceleration.x", "imu.linear_acceleration.y", "imu.linear_acceleration.z"),
)
GYRO_ALIASES = (
    ("gyro_x", "gyro_y", "gyro_z"),
    ("gyr_x", "gyr_y", "gyr_z"),
    ("gx", "gy", "gz"),
    ("wx", "wy", "wz"),
    ("angular_velocity.x", "angular_velocity.y", "angular_velocity.z"),
    ("imu.angular_velocity.x", "imu.angular_velocity.y", "imu.angular_velocity.z"),
)


##################################################
# Public loading interface
##################################################

def discover_imu_files(dataset_root: str | Path) -> list[Path]:
    """Find likely raw IMU tables under a New College experiment directory.

    Args:
        dataset_root: Path to the experiment root.

    Returns:
        Sorted candidate CSV/TXT-like files whose paths suggest IMU,
        accelerometer, gyroscope, inertial, or Xsens data.
    """
    root = Path(dataset_root).expanduser()
    candidates: list[Path] = []

    # Search conservatively for table-like files and avoid point-cloud formats.
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMU_EXTENSIONS:
            continue

        path_text = str(path).lower()
        if any(token in path_text for token in ("imu", "inertial", "accel", "gyro", "xsens")):
            candidates.append(path)

    return sorted(candidates)


def load_imu_file(
    path: str | Path,
    *,
    target_frequency_hz: float = 100.0,
    name: str | None = None,
    frame_id: str | None = None,
) -> IMUData:
    """Load and uniformly resample one raw IMU table.

    The New College format is supported directly::

        #counter,sec,nsec,wx [rad s^-1],wy [rad s^-1],wz [rad s^-1],
        ax [m s^-2],ay [m s^-2],az [m s^-2]

    For this representation, timestamps are computed as
    ``sec + nsec * 1e-9`` and shifted so the first chronological sample is at
    zero seconds. The subtraction is performed component-wise before forming
    floating-point Unix seconds to preserve sub-second precision.

    Args:
        path: CSV, TSV, whitespace, or delimited text file containing time,
            acceleration, and gyroscope columns.
        target_frequency_hz: Output sampling frequency in hertz.
        name: Optional stream name. Defaults to the file stem.
        frame_id: Optional sensor-frame identifier.

    Returns:
        Resampled ``IMUData`` with relative timestamps in seconds,
        acceleration in m/s², and angular velocity in rad/s.

    Raises:
        ValueError: If required columns, timestamps, or measurements are
            missing or invalid.
    """
    source_path = Path(path).expanduser()
    table = _read_table(source_path)

    timestamps_s, timestamp_description, timestamp_columns = _extract_timestamps(table)
    accel_columns = _find_vector_columns(table.columns, ACCEL_ALIASES)
    gyro_columns = _find_vector_columns(table.columns, GYRO_ALIASES)

    accel_mps2 = _finite_measurement_matrix(table, accel_columns, "accelerometer")
    gyro_radps = _finite_measurement_matrix(table, gyro_columns, "gyroscope")
    gyro_radps = _convert_gyro_units(gyro_radps, gyro_columns)

    # Sort all channels together because raw tables are not assumed ordered.
    order = np.argsort(timestamps_s, kind="stable")
    timestamps_s = timestamps_s[order]
    accel_mps2 = accel_mps2[order]
    gyro_radps = gyro_radps[order]

    # Interpolation requires strictly increasing timestamps. Keep the first
    # sample at each repeated timestamp and apply the same selection to signals.
    timestamps_s, unique_indices = np.unique(timestamps_s, return_index=True)
    accel_mps2 = accel_mps2[unique_indices]
    gyro_radps = gyro_radps[unique_indices]

    imu = IMUData(
        timestamps_s=timestamps_s,
        accel_mps2=accel_mps2,
        gyro_radps=gyro_radps,
        name=name or source_path.stem,
        frame_id=frame_id,
        metadata={
            "source_path": str(source_path),
            "timestamp_column": timestamp_description,
            "timestamp_columns": list(timestamp_columns),
            "timestamp_origin": "first chronological sample",
            "accel_columns": list(accel_columns),
            "gyro_columns": list(gyro_columns),
        },
    )
    return resample_imu(imu, target_frequency_hz=target_frequency_hz)


def load_imus(
    dataset_root: str | Path,
    *,
    target_frequency_hz: float = 100.0,
    max_files: int | None = None,
) -> dict[str, IMUData]:
    """Load every discoverable IMU stream in an experiment directory.

    Args:
        dataset_root: Path to the New College experiment root.
        target_frequency_hz: Output sampling frequency in hertz.
        max_files: Optional file-count cap for quick inspection.

    Returns:
        Mapping from file stem to resampled ``IMUData``.
    """
    files = discover_imu_files(dataset_root)
    if max_files is not None:
        files = files[:max_files]

    out_dict = {
        path.parent.name+"_"+path.stem : load_imu_file(
            path,
            target_frequency_hz=target_frequency_hz,
            name=path.stem,
        )
        for path in files
    }
    return out_dict


def resample_imu(imu_data: IMUData, *, target_frequency_hz: float = 100.0) -> IMUData:
    """Resample one IMU stream onto a uniform time grid.

    Args:
        imu_data: Input IMU stream with strictly increasing timestamps.
        target_frequency_hz: Desired output frequency in hertz.

    Returns:
        New ``IMUData`` sampled uniformly at ``target_frequency_hz``.

    Raises:
        ValueError: If the requested frequency is invalid or fewer than two
            input samples are available.
    """
    if target_frequency_hz <= 0.0:
        raise ValueError("target_frequency_hz must be positive.")
    if imu_data.timestamps_s.size < 2:
        raise ValueError("At least two IMU samples are required for resampling.")

    sample_period_s = 1.0 / float(target_frequency_hz)
    start_time_s = float(imu_data.timestamps_s[0])
    stop_time_s = float(imu_data.timestamps_s[-1])

    timestamps_s = np.arange(
        start_time_s,
        stop_time_s + 0.5 * sample_period_s,
        sample_period_s,
    )
    timestamps_s = timestamps_s[timestamps_s <= stop_time_s]

    # Interpolate each physical axis independently onto the common time grid.
    accel_mps2 = np.column_stack(
        [
            np.interp(
                timestamps_s,
                imu_data.timestamps_s,
                imu_data.accel_mps2[:, axis],
            )
            for axis in range(3)
        ]
    )
    gyro_radps = np.column_stack(
        [
            np.interp(
                timestamps_s,
                imu_data.timestamps_s,
                imu_data.gyro_radps[:, axis],
            )
            for axis in range(3)
        ]
    )

    metadata = dict(imu_data.metadata)
    metadata["source_frequency_hz_estimate"] = _estimate_frequency(imu_data.timestamps_s)

    return IMUData(
        timestamps_s=timestamps_s,
        accel_mps2=accel_mps2,
        gyro_radps=gyro_radps,
        name=imu_data.name,
        frame_id=imu_data.frame_id,
        frequency_hz=float(target_frequency_hz),
        metadata=metadata,
    )


##################################################
# Table parsing and column matching
##################################################

def _read_table(path: Path) -> pd.DataFrame:
    """Read a delimited IMU table, including a ``#``-prefixed header.

    New College files use a real CSV header beginning with ``#counter``. A
    normal ``comment="#"`` read would discard this line, so the header is
    detected and supplied explicitly while later comment lines remain ignored.

    Args:
        path: Raw table path.

    Returns:
        Nonempty table with normalized column names.

    Raises:
        ValueError: If the file contains no usable rows or columns.
    """
    commented_header = _commented_header(path)

    if commented_header is not None:
        header_names, delimiter = commented_header
        table = pd.read_csv(
            path,
            sep=delimiter,
            names=header_names,
            header=None,
            comment="#",
            engine="python",
        )
    else:
        try:
            table = pd.read_csv(path, sep=None, engine="python", comment="#")
        except Exception:
            table = pd.read_csv(path, sep=r"\s+", engine="python", comment="#")

    table = table.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if table.empty or table.shape[1] == 0:
        raise ValueError(f"IMU table {path} contains no usable data.")

    table.columns = [_normalize_column_name(column) for column in table.columns]
    return table


def _commented_header(path: Path) -> tuple[list[str], str] | None:
    """Return names and delimiter for a leading commented table header.

    Args:
        path: Candidate table path.

    Returns:
        ``(column_names, delimiter)`` when a leading line such as
        ``#counter,sec,nsec,...`` is detected; otherwise ``None``.
    """
    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for raw_line in stream:
            stripped = raw_line.strip()
            if not stripped:
                continue
            if not stripped.startswith("#"):
                return None

            header_text = stripped[1:].strip()
            delimiter = _header_delimiter(header_text)
            if delimiter is None:
                continue

            column_names = next(csv.reader([header_text], delimiter=delimiter))
            lookup_keys = {_column_lookup_key(name) for name in column_names}
            has_sensor_columns = (
                {"sec", "nsec"}.issubset(lookup_keys)
                and {"wx", "wy", "wz"}.issubset(lookup_keys)
                and {"ax", "ay", "az"}.issubset(lookup_keys)
            )
            if has_sensor_columns:
                return column_names, delimiter

    return None


def _header_delimiter(header_text: str) -> str | None:
    """Infer a simple delimiter used by a commented header."""
    for delimiter in (",", "\t", ";"):
        if delimiter in header_text:
            return delimiter
    return None


def _normalize_column_name(column: object) -> str:
    """Normalize superficial column-name formatting while retaining units."""
    normalized = str(column).strip().lstrip("#").strip().lower()
    normalized = re.sub(r"\s+", "_", normalized)
    return normalized


def _column_lookup_key(column: object) -> str:
    """Return a unit-free key used for matching column aliases."""
    normalized = _normalize_column_name(column)
    normalized = re.sub(r"_?\[[^\]]*\]", "", normalized)
    normalized = re.sub(r"_?\([^)]*\)$", "", normalized)
    return normalized.strip("_")


def _find_column(columns: Iterable[str], aliases: Iterable[str]) -> str:
    """Find one column using normalized aliases.

    Args:
        columns: Available table columns.
        aliases: Accepted semantic aliases.

    Returns:
        Original normalized table-column name.

    Raises:
        ValueError: If none of the aliases is present.
    """
    columns = list(columns)
    normalized = {_column_lookup_key(column): column for column in columns}

    for alias in aliases:
        lookup_key = _column_lookup_key(alias)
        if lookup_key in normalized:
            return normalized[lookup_key]

    raise ValueError(f"Could not find any of these columns: {list(aliases)}")


def _find_optional_column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    """Find one aliased column or return ``None`` when absent."""
    try:
        return _find_column(columns, aliases)
    except ValueError:
        return None


def _find_vector_columns(
    columns: Iterable[str],
    aliases: Iterable[tuple[str, str, str]],
) -> tuple[str, str, str]:
    """Find one three-axis column group using normalized aliases.

    Args:
        columns: Available table columns.
        aliases: Accepted three-column alias groups.

    Returns:
        Three original normalized table-column names.

    Raises:
        ValueError: If no complete alias group is present.
    """
    columns = list(columns)
    normalized = {_column_lookup_key(column): column for column in columns}

    for group in aliases:
        lookup_keys = tuple(_column_lookup_key(name) for name in group)
        if all(name in normalized for name in lookup_keys):
            return tuple(normalized[name] for name in lookup_keys)

    raise ValueError(f"Could not find vector columns matching aliases: {list(aliases)}")


##################################################
# Timestamp and measurement normalization
##################################################

def _extract_timestamps(
    table: pd.DataFrame,
) -> tuple[np.ndarray, str, tuple[str, ...]]:
    """Extract relative timestamps from one- or two-column representations.

    Args:
        table: Parsed IMU table.

    Returns:
        Tuple containing relative seconds, a metadata description, and the
        source timestamp-column names.

    Raises:
        ValueError: If no supported timestamp representation is available.
    """
    seconds_column = _find_optional_column(table.columns, SECOND_ALIASES)
    nanoseconds_column = _find_optional_column(table.columns, NANOSECOND_ALIASES)

    if seconds_column is not None and nanoseconds_column is not None:
        timestamps_s = _timestamps_from_sec_nsec(
            table[seconds_column].to_numpy(),
            table[nanoseconds_column].to_numpy(),
        )
        return timestamps_s, "sec+nsec", (seconds_column, nanoseconds_column)

    timestamp_column = _find_optional_column(table.columns, TIMESTAMP_ALIASES)
    if timestamp_column is None:
        raise ValueError(
            "Could not find either a timestamp column or both sec and nsec columns."
        )

    return (
        _normalize_timestamps(table[timestamp_column].to_numpy()),
        timestamp_column,
        (timestamp_column,),
    )


def _timestamps_from_sec_nsec(
    raw_seconds: np.ndarray,
    raw_nanoseconds: np.ndarray,
) -> np.ndarray:
    """Combine Unix second and nanosecond fields into relative seconds.

    The large Unix-second component is subtracted before adding the fractional
    nanoseconds, retaining substantially more precision than first constructing
    absolute floating-point Unix timestamps.

    Args:
        raw_seconds: Integer-like Unix seconds, shape ``(N,)``.
        raw_nanoseconds: Nanoseconds within each second, shape ``(N,)``.

    Returns:
        Relative timestamps in seconds, shape ``(N,)``.

    Raises:
        ValueError: If fields are nonnumeric, nonfinite, differently sized, or
            nanoseconds lie outside ``[0, 1e9)``.
    """
    seconds = np.asarray(pd.to_numeric(raw_seconds, errors="coerce"), dtype=float)
    nanoseconds = np.asarray(pd.to_numeric(raw_nanoseconds, errors="coerce"), dtype=float)

    if seconds.ndim != 1 or nanoseconds.ndim != 1 or seconds.shape != nanoseconds.shape:
        raise ValueError("sec and nsec columns must be one-dimensional and equally sized.")
    if not np.all(np.isfinite(seconds)) or not np.all(np.isfinite(nanoseconds)):
        raise ValueError("sec and nsec columns must contain only finite numeric values.")
    if np.any(nanoseconds < 0.0) or np.any(nanoseconds >= 1e9):
        raise ValueError("nsec values must satisfy 0 <= nsec < 1e9.")

    reference_seconds = float(seconds[0])
    reference_nanoseconds = float(nanoseconds[0])
    timestamps_s = (
        seconds
        + (nanoseconds) * 1e-9
    )
    return timestamps_s


def _finite_measurement_matrix(
    table: pd.DataFrame,
    columns: tuple[str, str, str],
    signal_name: str,
) -> np.ndarray:
    """Return one finite ``(N, 3)`` measurement matrix."""
    values = table.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError(f"{signal_name} columns must contain finite numeric values.")
    return values


def _normalize_timestamps(raw_timestamps: np.ndarray) -> np.ndarray:
    """Normalize a single timestamp column to relative seconds."""
    values = np.asarray(raw_timestamps)

    if np.issubdtype(values.dtype, np.datetime64):
        seconds = values.astype("datetime64[ns]").astype(np.int64) / 1e9
    else:
        numeric = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
        if np.all(np.isfinite(numeric)):
            seconds = numeric
        else:
            parsed = pd.to_datetime(values, errors="coerce")
            if np.any(pd.isna(parsed)):
                raise ValueError("Timestamp column contains invalid values.")
            seconds = np.asarray(parsed.astype("int64"), dtype=float) / 1e9

    if not np.all(np.isfinite(seconds)) or seconds.size == 0:
        raise ValueError("Timestamp column contains no finite values.")

    # Infer common absolute timestamp units from magnitude.
    median_abs = float(np.median(np.abs(seconds)))
    if median_abs > 1e17:
        seconds = seconds / 1e9
    elif median_abs > 1e14:
        seconds = seconds / 1e6
    elif median_abs > 1e11:
        seconds = seconds / 1e3

    return seconds


def _convert_gyro_units(
    gyro: np.ndarray,
    columns: tuple[str, str, str],
) -> np.ndarray:
    """Convert angular velocity to rad/s when headers specify degrees."""
    joined = " ".join(columns).lower()
    if "deg" in joined or "dps" in joined:
        return np.deg2rad(gyro)
    return gyro


def _estimate_frequency(timestamps_s: np.ndarray) -> float | None:
    """Estimate sampling frequency from the median positive time increment."""
    if timestamps_s.size < 2:
        return None

    deltas = np.diff(timestamps_s)
    deltas = deltas[deltas > 0.0]
    if deltas.size == 0:
        return None

    return float(1.0 / np.median(deltas))