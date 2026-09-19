"""KAIST scan-to-map LiDAR pose loading and relative-pose conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from transform import se3_to_relative_se3

from .data import LidarData


POSE_CSV_COLUMNS = (
    "timestamp",
    "r11", "r12", "r13", "t11",
    "r21", "r22", "r23", "t21",
    "r31", "r32", "r33", "t31",
)


def load_lidar_map_poses(filepath: str | Path, *, only_successful: bool = True) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load absolute KAIST scan-to-map LiDAR poses T_W_L from CSV.

    The expected pose columns are:

        timestamp,
        r11,r12,r13,t11,
        r21,r22,r23,t21,
        r31,r32,r33,t31.

    Additional diagnostic columns such as success, fitness, and inlier_rmse are preserved in the returned DataFrame.

    Failed or non-finite rows are removed before usable poses are returned.
    """

    source_path = Path(filepath).expanduser()

    if not source_path.is_file():
        raise FileNotFoundError(f"LiDAR pose CSV does not exist: {source_path}")

    table = pd.read_csv(source_path)

    missing_columns = [column for column in POSE_CSV_COLUMNS if column not in table.columns]

    if missing_columns:
        raise ValueError(f"LiDAR pose CSV is missing required columns: {missing_columns}")

    if table.empty:
        raise ValueError(f"LiDAR pose CSV is empty: {source_path}")

    table = table.copy()
    table["_source_row_index"] = np.arange(len(table), dtype=int)

    timestamps = pd.to_numeric(table["timestamp"], errors="coerce").to_numpy(dtype=float)
    pose_values = table.loc[:, POSE_CSV_COLUMNS[1:]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    valid_mask = np.isfinite(timestamps) & np.all(np.isfinite(pose_values), axis=1)

    if only_successful and "success" in table.columns:
        valid_mask &= _success_mask(table["success"])

    table = table.loc[valid_mask].copy()

    if len(table) < 2:
        raise ValueError("At least two finite successful absolute LiDAR poses are required to construct relative poses.")

    # Sort every column together so pose diagnostics stay aligned.
    table = table.sort_values("timestamp", kind="stable").reset_index(drop=True)

    timestamps_s = pd.to_numeric(table["timestamp"], errors="raise").to_numpy(dtype=float)

    if np.any(np.diff(timestamps_s) <= 0.0):
        raise ValueError("Usable LiDAR pose timestamps must be strictly increasing.")

    pose_values = table.loc[:, POSE_CSV_COLUMNS[1:]].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)

    poses_T_W_L = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(table), axis=0)
    poses_T_W_L[:, :3, :] = pose_values.reshape(-1, 3, 4)

    return timestamps_s, poses_T_W_L, table


def map_poses_to_relative_lidar_data(scan_timestamps_s: np.ndarray, poses_T_W_L: np.ndarray, *, fitness: np.ndarray | None = None, inlier_rmse: np.ndarray | None = None, metadata: dict | None = None) -> LidarData:
    """
    Convert absolute scan-to-map poses into consecutive relative LiDAR poses.

    For absolute poses

        T_W_L[i],

    the returned relative pose is

        T_L_previous_L_current[i] = inverse(T_W_L[i]) @ T_W_L[i + 1].

    Relative measurement timestamps use the end scan timestamp:

        timestamps_s[i] = scan_timestamps_s[i + 1].

    This convention matches the augmentation renderer, where each relative measurement is interpreted as an interval ending at its stored timestamp.
    """

    scan_timestamps_s = np.asarray(scan_timestamps_s, dtype=float).reshape(-1)
    poses_T_W_L = np.asarray(poses_T_W_L, dtype=float)

    if poses_T_W_L.ndim != 3 or poses_T_W_L.shape[1:] != (4, 4):
        raise ValueError("poses_T_W_L must have shape [N, 4, 4].")

    if scan_timestamps_s.shape != (poses_T_W_L.shape[0],):
        raise ValueError("scan_timestamps_s must contain one timestamp per absolute LiDAR pose.")

    if poses_T_W_L.shape[0] < 2:
        raise ValueError("At least two absolute LiDAR poses are required.")

    if not np.all(np.isfinite(poses_T_W_L)):
        raise ValueError("poses_T_W_L contains non-finite values.")

    if not np.all(np.isfinite(scan_timestamps_s)):
        raise ValueError("scan_timestamps_s contains non-finite values.")

    if np.any(np.diff(scan_timestamps_s) <= 0.0):
        raise ValueError("scan_timestamps_s must be strictly increasing.")

    relative_poses = se3_to_relative_se3(poses_T_W_L)
    relative_timestamps_s = scan_timestamps_s[1:].copy()
    number_relative_poses = relative_poses.shape[0]

    relative_fitness = _normalize_interval_diagnostics(fitness, absolute_pose_count=poses_T_W_L.shape[0], relative_pose_count=number_relative_poses, name="fitness")
    relative_inlier_rmse = _normalize_interval_diagnostics(inlier_rmse, absolute_pose_count=poses_T_W_L.shape[0], relative_pose_count=number_relative_poses, name="inlier_rmse")

    output_metadata = dict(metadata or {})
    output_metadata.update({"absolute_pose_convention": "T_W_L", "relative_pose_convention": "T_L_previous_L_current", "relative_pose_formula": "inverse(T_W_L_previous) @ T_W_L_current", "relative_timestamp_semantics": "end_scan_timestamp"})

    return LidarData(timestamps_s=relative_timestamps_s, relative_poses_se3=relative_poses, scan_timestamps_s=scan_timestamps_s, fitness=relative_fitness, inlier_rmse=relative_inlier_rmse, metadata=output_metadata)


def load_lidar_relative_poses(filepath: str | Path, *, only_successful: bool = True) -> LidarData:
    """
    Load scan-to-map T_W_L poses and convert them to relative LiDAR observations.

    This is the main LiDAR-loading entry point for the learned calibration pipeline. It does not load raw point clouds and performs no ICP or scan processing.
    """

    source_path = Path(filepath).expanduser()

    scan_timestamps_s, poses_T_W_L, table = load_lidar_map_poses(source_path, only_successful=only_successful)

    fitness = _optional_numeric_column(table, "fitness")
    inlier_rmse = _optional_numeric_column(table, "inlier_rmse")

    return map_poses_to_relative_lidar_data(
        scan_timestamps_s,
        poses_T_W_L,
        fitness=fitness,
        inlier_rmse=inlier_rmse,
        metadata={
            "source_path": str(source_path),
            "source_format": "KAIST scan-to-map pose CSV",
            "only_successful": bool(only_successful),
            "number_absolute_poses": int(len(scan_timestamps_s)),
            "source_row_indices": table["_source_row_index"].to_numpy(dtype=int).tolist(),
            "diagnostics_semantics": "fitness and inlier_rmse are scan-to-map diagnostics associated with the endpoint absolute pose of each relative interval",
        },
    )


def _normalize_interval_diagnostics(values: np.ndarray | None, *, absolute_pose_count: int, relative_pose_count: int, name: str) -> np.ndarray | None:
    """
    Convert optional diagnostics to one value per relative interval.

    A diagnostic vector with one value per absolute scan is mapped to the interval endpoint using values[1:]. A vector already containing one value per relative interval is preserved.
    """

    if values is None:
        return None

    values = np.asarray(values, dtype=float).reshape(-1)

    if values.shape == (absolute_pose_count,):
        values = values[1:]
    elif values.shape != (relative_pose_count,):
        raise ValueError(f"{name} must contain either one value per absolute pose or one value per relative interval.")

    if not np.all(np.isfinite(values)):
        return None

    return values.copy()


def _optional_numeric_column(table: pd.DataFrame, column_name: str) -> np.ndarray | None:
    """Read one optional finite numeric diagnostic column."""

    if column_name not in table.columns:
        return None

    values = pd.to_numeric(table[column_name], errors="coerce").to_numpy(dtype=float)

    if not np.all(np.isfinite(values)):
        return None

    return values


def _success_mask(values: pd.Series) -> np.ndarray:
    """Convert a common CSV success representation to a boolean mask."""

    if values.dtype == bool:
        return values.to_numpy(dtype=bool)

    numeric = pd.to_numeric(values, errors="coerce")
    numeric_values = numeric.to_numpy(dtype=float)

    if np.all(np.isfinite(numeric_values)):
        return numeric_values != 0.0

    normalized = values.astype(str).str.strip().str.lower()

    valid_true = {"true", "1", "yes", "y"}
    valid_false = {"false", "0", "no", "n"}

    unknown = ~normalized.isin(valid_true | valid_false)

    if bool(unknown.any()):
        unknown_values = sorted(set(normalized[unknown].tolist()))
        raise ValueError(f"Could not interpret success column values: {unknown_values}")

    return normalized.isin(valid_true).to_numpy(dtype=bool)