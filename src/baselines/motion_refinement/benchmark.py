"""Repeatable local-refinement evaluation on successive physical windows.

This is a numerical-only measurement harness. Ground-truth rotations are read
only *after* fitting, and never supplied as an objective or stopping criterion.
The future FM notebook can use exactly the same window list and initial pose.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd

from baselines.li_init.structures import IMUSeries, LidarIncrements
from .estimator import MotionRefinementCalibrator
from .structures import MotionRefinementResult
from .geometry import log_so3, validate_pose, validate_rotation


@dataclass(frozen=True)
class MotionRefinementWindow:
    name: str
    imu: IMUSeries
    lidar: LidarIncrements
    right_lidar: LidarIncrements | None = None
    t_start: float | None = None
    t_end: float | None = None


def rotation_error_degrees(R: np.ndarray, R_ref: np.ndarray) -> float:
    """SO(3) geodesic error, only for evaluation and never for calibration."""
    return float(np.rad2deg(np.linalg.norm(log_so3(np.asarray(R) @ validate_rotation(R_ref).T))))


def run_motion_refinement_rollout(windows: Sequence[MotionRefinementWindow], calibrator: MotionRefinementCalibrator, *, initial_T_I_L: np.ndarray, T_I_R: np.ndarray | None = None, initial_time_offset: float = 0.0, initial_gyro_bias: np.ndarray | None = None, reference_R_LI: np.ndarray | None = None, continue_on_failure: bool = True) -> tuple[pd.DataFrame, list[MotionRefinementResult | None]]:
    """Warm-start each successive window with the preceding accepted solution.

    The provided `windows` define exact matched physical intervals and must be
    ordered chronologically. `initial_T_I_L` is NOT replaced with GT. On a
    numerical error, retain the preceding estimate and record the failure.
    """
    if not windows:
        raise ValueError('No motion refinement windows provided')
    T = validate_pose(initial_T_I_L, 'initial_T_I_L')
    R_gt = None if reference_R_LI is None else validate_rotation(reference_R_LI, 'reference_R_LI')
    b = np.zeros(3) if initial_gyro_bias is None else np.asarray(initial_gyro_bias, dtype=float).copy()
    if b.shape != (3,):
        raise ValueError('initial_gyro_bias must be 3D')
    tau = float(initial_time_offset)
    rows: list[dict] = []
    results: list[MotionRefinementResult | None] = []
    initial_t = float(windows[0].t_start if windows[0].t_start is not None else windows[0].lidar.t_start[0])
    rows.append(dict(step=0, name='initial', elapsed_physical_s=0.0, accepted=True, error_deg=np.nan if R_gt is None else rotation_error_degrees(T[:3, :3], R_gt), computation_ms=0.0, cumulative_ms=0.0, message='initial', num_iterations=0))
    cumulative_ms = 0.0
    previous_end = initial_t
    for k, window in enumerate(windows, 1):
        a = float(window.t_start if window.t_start is not None else window.lidar.t_start[0])
        e = float(window.t_end if window.t_end is not None else window.lidar.t_end[-1])
        if a < previous_end - 1e-8 or e <= a:
            raise ValueError('Windows must be chronological and non-overlapping')
        previous_end = e
        try:
            result = calibrator.fit(window.imu, window.lidar, initial_T_I_L=T, right_lidar=window.right_lidar, T_I_R=T_I_R, initial_time_offset=tau, initial_gyro_bias=b)
            accepted = bool(result.success)
            if accepted:
                T = result.T_I_L.copy()
                b = result.gyro_bias.copy()
                tau = result.time_offset
            message = result.message
            runtime = result.elapsed_ms
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            if not continue_on_failure:
                raise
            result = None
            accepted = False
            message = f'{type(exc).__name__}: {exc}'
            runtime = np.nan
        cumulative_ms += 0.0 if not np.isfinite(runtime) else runtime
        results.append(result)
        rows.append(dict(step=k, name=window.name, elapsed_physical_s=e-initial_t, accepted=accepted, error_deg=np.nan if R_gt is None else rotation_error_degrees(T[:3, :3], R_gt), computation_ms=runtime, cumulative_ms=cumulative_ms, message=message, num_iterations=None if result is None else result.iterations))
    return pd.DataFrame(rows), results
