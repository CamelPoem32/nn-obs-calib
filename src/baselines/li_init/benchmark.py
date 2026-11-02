"""Reproducible, calibration-only evaluation on identical LiDAR/IMU windows.

Reference extrinsics are used solely AFTER fitting, never in the numerical
estimator. Failures and underconstrained windows remain in the returned table.
No LiDAR scan rerendering, artificial perturbation, or forced success flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from .adapters import CalibrationReference, window_sample_to_li_init
from .estimator import LIInitCalibrator
from .structures import IMUSeries, LidarIncrements, LIInitConfig, LIInitializationResult


@dataclass(frozen=True)
class BenchmarkCase:
    """One independent calibration task; no prior is passed to LI-Init."""

    name: str
    imu: IMUSeries
    lidar: LidarIncrements
    reference: CalibrationReference | None = None
    sequence: str | None = None
    window_duration_s: float | None = None
    imu_padding_s: float = 0.0

    def __post_init__(self) -> None:
        if self.imu_padding_s < 0:
            raise ValueError("imu_padding_s must not be negative")


def rotation_error_degrees(R_est: np.ndarray, R_ref: np.ndarray) -> float:
    """SO(3) geodesic error, in degrees, invariant to rotation parameterization."""
    return float(np.rad2deg(Rotation.from_matrix(np.asarray(R_ref).T @ np.asarray(R_est)).magnitude()))


def calibration_update_rotation(R_prior_LI: np.ndarray, R_est_LI: np.ndarray) -> np.ndarray:
    """Left-multiplicative correction R_est = delta_R @ R_prior (for NN comparison)."""
    return np.asarray(R_est_LI) @ np.asarray(R_prior_LI).T


def _base_row(case: BenchmarkCase) -> dict[str, Any]:
    return {
        "name": case.name,
        "sequence": case.sequence,
        "window_duration_s": case.window_duration_s if case.window_duration_s is not None else float(case.lidar.t_end[-1] - case.lidar.t_start[0]),
        "imu_padding_s": case.imu_padding_s,
        "imu_context_duration_s": float(case.imu.t[-1] - case.imu.t[0]),
        "n_lidar_increments": int(case.lidar.t_start.size),
        "n_imu_samples": int(case.imu.t.size),
    }


def evaluate_li_init_case(case: BenchmarkCase, calibrator: LIInitCalibrator | None = None, *, config: LIInitConfig | None = None, raise_on_error: bool = False) -> tuple[dict[str, Any], LIInitializationResult | None]:
    """Evaluate one window, preserving errors/degeneracy in the result row."""
    if calibrator is not None and config is not None:
        raise ValueError("Provide calibrator OR config, not both")
    solver = calibrator or LIInitCalibrator(config)
    row = _base_row(case)
    row.update({"success": False, "rotation_success": False, "rotation_observable": False, "temporal_success": None, "translation_success": None, "translation_observable": None, "error": None,
                "rotation_error_deg": np.nan, "time_offset_error_ms": np.nan, "translation_error_m": np.nan,
                "time_offset_s": np.nan, "gyro_bias_norm_radps": np.nan, "rotation_rms_radps": np.nan, "rotation_rank": np.nan, "joint_rank": np.nan,
                "translation_rank": np.nan, "translation_rms_mps2": np.nan, "translation_x_m": np.nan, "translation_y_m": np.nan, "translation_z_m": np.nan,
                "rotation_nfev": np.nan, "translation_nfev": np.nan, "rotation_message": None, "temporal_message": None, "translation_message": None})
    start = perf_counter()
    try:
        result = solver.fit(case.imu, case.lidar)
    except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["elapsed_s"] = perf_counter() - start
        if raise_on_error:
            raise
        return row, None
    row["elapsed_s"] = perf_counter() - start
    r = result.rotation
    row.update({"success": bool(result.success), "rotation_success": bool(r.success), "rotation_observable": bool(r.observable),
                "rotation_rank": int(r.rotation_rank), "joint_rank": r.joint_rank, "rotation_nfev": int(r.nfev),
                "rotation_rms_radps": float(r.rms_residual), "time_offset_s": float(r.time_offset),
                "gyro_bias_norm_radps": float(np.linalg.norm(r.gyro_bias)), "rotation_message": r.message})
    if result.temporal is not None:
        row["temporal_success"] = bool(result.temporal.success)
        row["temporal_message"] = result.temporal.message
    if result.translation is not None:
        tr = result.translation
        row.update({"translation_success": bool(tr.success), "translation_observable": bool(tr.observable), "translation_rank": int(tr.translation_rank),
                    "translation_rms_mps2": float(tr.rms_residual), "translation_nfev": int(tr.nfev), "translation_message": tr.message,
                    "translation_x_m": float(tr.p_IL[0]), "translation_y_m": float(tr.p_IL[1]), "translation_z_m": float(tr.p_IL[2])})
    if case.reference is not None:
        # Report raw estimates even for failures, but only successful estimates
        # should be used in any accuracy aggregate.
        row["rotation_error_deg"] = rotation_error_degrees(r.R_LI, case.reference.R_LI)
        if case.reference.time_offset is not None:
            row["time_offset_error_ms"] = 1000.0 * abs(r.time_offset - case.reference.time_offset)
        if case.reference.p_IL is not None and result.translation is not None:
            row["translation_error_m"] = float(np.linalg.norm(result.translation.p_IL - case.reference.p_IL))
    return row, result


def benchmark_li_init(cases: Iterable[BenchmarkCase], config: LIInitConfig | None = None, *, raise_on_error: bool = False, return_results: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, LIInitializationResult]]:
    """Benchmark all windows and return a tabular record including failures.

    Use ``return_results=True`` to inspect individual rotations, observability
    spectra, temporal candidates and translation estimates after the run.
    """
    solver = LIInitCalibrator(config)
    rows: list[dict[str, Any]] = []
    outputs: dict[str, LIInitializationResult] = {}
    for case in cases:
        row, result = evaluate_li_init_case(case, solver, raise_on_error=raise_on_error)
        rows.append(row)
        if result is not None:
            if case.name in outputs:
                raise ValueError(f"Duplicate BenchmarkCase name {case.name!r}; names must be unique when results are returned")
            outputs[case.name] = result
    df = pd.DataFrame(rows)
    return (df, outputs) if return_results else df


def summarize_li_init(results: pd.DataFrame, *, group_by: tuple[str, ...] = ("window_duration_s",)) -> pd.DataFrame:
    """Aggregate count, valid-rate, runtime, and errors of successful windows.

    Use failure rate as a separate KPI. Accuracy excludes failed/degenerate
    windows rather than allowing an inaccurate 'success' to dominate metrics.
    """
    if results.empty:
        return pd.DataFrame()
    if any(column not in results for column in group_by):
        raise ValueError("Unknown group_by column")
    def aggregate(group: pd.DataFrame) -> pd.Series:
        ok = group[group["success"].fillna(False).astype(bool)]
        rot = pd.to_numeric(ok["rotation_error_deg"], errors="coerce").dropna()
        offset = pd.to_numeric(ok["time_offset_error_ms"], errors="coerce").dropna()
        time = pd.to_numeric(group["elapsed_s"], errors="coerce").dropna()
        return pd.Series({"n_windows": len(group), "n_success": len(ok), "success_rate": len(ok) / len(group),
                          "runtime_median_ms": 1000.0 * float(time.median()) if len(time) else np.nan,
                          "rotation_median_deg": float(rot.median()) if len(rot) else np.nan,
                          "rotation_p90_deg": float(rot.quantile(0.9)) if len(rot) else np.nan,
                          "rotation_success_under_5deg": float((rot < 5).sum() / len(group)) if len(rot) else 0.,
                          "time_offset_median_ms": float(offset.median()) if len(offset) else np.nan})
    if not group_by:
        return aggregate(results).to_frame().T
    summary_rows = []
    for key, group in results.groupby(list(group_by), dropna=False, sort=True):
        keys = key if isinstance(key, tuple) else (key,)
        summary_rows.append({**dict(zip(group_by, keys)), **aggregate(group).to_dict()})
    return pd.DataFrame(summary_rows)


def cases_from_window_dataset(dataset: Any, *, reference: CalibrationReference | None = None, max_windows: int | None = None, stride: int = 1, increment_convention: str = "start_to_end", gyro_key: str = "gyro", accel_key: str | None = "accel", lidar_key: str = "lidar") -> list[BenchmarkCase]:
    """Convert stored WindowSample records, without their prior/target labels.

    ``reference`` must come from independently verified rig parameters. Do not
    derive ground truth from a synthetically perturbed ``current_calibration``.
    """
    if stride < 1 or (max_windows is not None and max_windows < 1):
        raise ValueError("Invalid stride or max_windows")
    indices = list(range(0, len(dataset), stride))
    if max_windows is not None:
        indices = indices[:max_windows]
    cases = []
    for index in indices:
        imu, lidar = window_sample_to_li_init(dataset[index], gyro_key=gyro_key, accel_key=accel_key, lidar_key=lidar_key, increment_convention=increment_convention)
        sequence = dataset.sequence_name_for_window(index) if hasattr(dataset, "sequence_name_for_window") else None
        cases.append(BenchmarkCase(name=f"window_{index:06d}", sequence=sequence, imu=imu, lidar=lidar, reference=reference))
    return cases
