"""Read Boreas calibration values; document directions using devkit usage and pose checks.

Names alone do not establish direction for arbitrary external matrix files.
See calibration_direction_diagnostics and README for the Boreas-specific evidence.
"""

from pathlib import Path

import numpy as np


def validate_rigid_transform(transform, *, atol: float = 2e-5) -> np.ndarray:
    """Validate SE(3), allowing rounding in six-decimal calibration text files.

    Values are never silently projected or otherwise repaired.
    """
    T = np.asarray(transform, dtype=float)
    if T.shape[-2:] != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError("Expected finite rigid transforms with shape (..., 4, 4)")
    R = T[..., :3, :3]
    if (not np.allclose(T[..., 3, :], [0, 0, 0, 1], rtol=0, atol=atol)
            or not np.allclose(np.swapaxes(R, -1, -2) @ R, np.eye(3), rtol=0, atol=atol)
            or not np.allclose(np.linalg.det(R), 1, rtol=0, atol=atol)):
        raise ValueError("Matrix is not a valid rigid transformation")
    return T


def load_calibration(path: str | Path) -> np.ndarray:
    """Read one 4x4 T_destination_source text file, preserving stored values."""
    T = np.loadtxt(Path(path).expanduser())
    if T.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 extrinsic in {path}")
    return validate_rigid_transform(T)


def load_calibrations(sequence_root: str | Path) -> dict[str, np.ndarray]:
    """Read available T_*.txt extrinsics; absent matrices are never identities."""
    return {p.stem: load_calibration(p) for p in sorted((Path(sequence_root) / "calib").glob("T_*.txt"))}


def lidar_to_dmu(calibrations: dict[str, np.ndarray]) -> np.ndarray:
    """T_dmu_lidar = inverse(T_applanix_dmu) @ T_applanix_lidar.

    Both stored matrices map their named sensor into the Applanix frame.
    Missing required matrices raise KeyError rather than inventing extrinsics.
    """
    return np.linalg.solve(calibrations["T_applanix_dmu"], calibrations["T_applanix_lidar"])


def calibration_direction_diagnostics(sequence_root: str | Path, *, samples: int = 60) -> list[dict]:
    """Compare both extrinsic directions to the provided reference trajectories.

    Check T_world_sensor = T_world_applanix @ T_applanix_sensor against the
    inverse hypothesis. This establishes file/pose convention consistency, not
    independent calibration accuracy: these trajectories share their origin.
    No DMU pose CSV exists, so this check cannot independently verify its extrinsic.
    """
    from scipy.spatial.transform import Rotation
    from .lidar import load_lidar_poses, load_reference_trajectory

    if not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    root = Path(sequence_root)
    reference = load_reference_trajectory(root / "applanix/gps_post_process.csv",
                                          timestamp_unit="s", sensor_frame="applanix")
    cal = load_calibrations(root)
    extrinsics = {"velodyne": cal["T_applanix_lidar"]}
    if "T_aeva_lidar" in cal:
        extrinsics["aeva"] = cal["T_applanix_lidar"] @ np.linalg.inv(cal["T_aeva_lidar"])
    results = []
    for source, extrinsic in extrinsics.items():
        filename = "lidar" if source == "velodyne" else "aeva"
        path = root / f"applanix/{filename}_poses.csv"
        if not path.is_file():
            continue
        trajectory = load_lidar_poses(path, source=source)
        valid = np.flatnonzero((trajectory.timestamps_ns >= reference.timestamps_ns[0]) &
                               (trajectory.timestamps_ns <= reference.timestamps_ns[-1]))
        if not valid.size:
            continue
        selected = valid[np.linspace(0, valid.size - 1, min(samples, valid.size), dtype=int)]
        interpolated = reference.match(trajectory.timestamps_ns[selected], tolerance_s=.01,
                                        interpolate=True, max_gap_s=.02)
        for hypothesis, T in (("stored convention", extrinsic), ("inverse convention", np.linalg.inv(extrinsic))):
            error = np.linalg.solve(interpolated @ T, trajectory.poses_T_W_L[selected])
            results.append({"source": source, "hypothesis": hypothesis, "samples": len(selected),
                            "median_position_error_m": float(np.median(np.linalg.norm(error[:, :3, 3], axis=1))),
                            "median_rotation_error_deg": float(np.rad2deg(np.median(Rotation.from_matrix(error[:, :3, :3]).magnitude())))})
    return results
