"""Auxiliary LI-Init-inspired translation, accelerometer bias, and gravity fit.

Uses *relative* scan-pair SE(3) increments to reconstruct an arbitrarily
anchored local LiDAR trajectory. This is not identical to the original FAST-LO
velocity/acceleration frontend. Second derivatives of noisy LiDAR odometry are
fragile: translation must be evaluated separately and is disabled by default.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .lidar_motion import central_diff, lidar_increments_to_motion
from .preprocessing import interpolate_vectors
from .structures import IMUSeries, LIInitConfig, LidarIncrements, TranslationCalibrationResult


def reconstruct_lidar_trajectory(lidar: LidarIncrements) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compose scan-pair increments into local LiDAR poses T_WL(t).

    The first pose is identity. This is sufficient for gravity/lever-arm
    calibration; absolute world position is not required. Nonconsecutive scan
    intervals are rejected because motion through a gap is unknown.
    """
    if not np.allclose(lidar.t_start[1:], lidar.t_end[:-1], rtol=0, atol=1e-6):
        raise ValueError("Translation requires consecutive LiDAR odometry increments without gaps")
    times = np.r_[lidar.t_start[0], lidar.t_end]
    poses = np.tile(np.eye(4), (times.size, 1, 1))
    for i, increment in enumerate(lidar.delta_T):
        if lidar.increment_convention == "end_to_start":
            R = increment[:3, :3]
            p = increment[:3, 3]
            transformed = np.eye(4)
            transformed[:3, :3] = R.T
            transformed[:3, 3] = -R.T @ p
            increment = transformed
        poses[i + 1] = poses[i] @ increment
    return times, poses[:, :3, :3], poses[:, :3, 3]


def lidar_linear_acceleration(lidar: LidarIncrements, config: LIInitConfig | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return midpoint t, R_WL, world acceleration, omega_L, and alpha_L.

    Cubic splines interpolate locally integrated position. Optional spline
    smoothing acts on position only, not on accelerometer measurements.
    Angular acceleration is the central difference of body angular velocity.
    """
    cfg = config or LIInitConfig()
    t, R_edges, p_edges = reconstruct_lidar_trajectory(lidar)
    t_mid = (lidar.t_start + lidar.t_end) * 0.5
    if t.size < 5:
        raise ValueError("At least four consecutive LiDAR increments are required for acceleration")
    if cfg.translation_spline_smoothing == 0:
        splines = [CubicSpline(t, p_edges[:, j]) for j in range(3)]
    else:
        splines = [UnivariateSpline(t, p_edges[:, j], s=cfg.translation_spline_smoothing, k=3) for j in range(3)]
    a_world = np.column_stack([spline.derivative(2)(t_mid) for spline in splines])
    motion = central_diff(lidar_increments_to_motion(lidar))
    R_d = lidar.delta_T[:, :3, :3]
    if lidar.increment_convention == "end_to_start":
        R_d = np.swapaxes(R_d, -1, -2)
    # Midpoint orientation via SO(3) geodesic between adjacent pose samples.
    half_rotation = Rotation.from_rotvec(0.5 * Rotation.from_matrix(R_d).as_rotvec()).as_matrix()
    R_mid = R_edges[:-1] @ half_rotation
    return t_mid, R_mid, a_world, motion.ang_vel, motion.ang_acc


def _hat_batch(v: np.ndarray) -> np.ndarray:
    """Vectorized skew matrices; used for the rigid-body lever-arm Jacobian."""
    H = np.zeros((len(v), 3, 3), dtype=np.float64)
    H[:, 0, 1] = -v[:, 2]
    H[:, 0, 2] = v[:, 1]
    H[:, 1, 0] = v[:, 2]
    H[:, 1, 2] = -v[:, 0]
    H[:, 2, 0] = -v[:, 1]
    H[:, 2, 1] = v[:, 0]
    return H


def _tangent_basis(unit_direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.eye(3)[int(np.argmin(abs(unit_direction)))]
    a = np.cross(unit_direction, ref)
    a = a / np.linalg.norm(a)
    return a, np.cross(unit_direction, a)


def _nuisance_marginalized_singular_values(jac: np.ndarray, threshold: float) -> np.ndarray:
    """Marginalize accelerometer bias and two gravity DoFs before testing p_IL."""
    Jp = jac[:, :3]
    Jn = jac[:, 3:]
    U, s, _ = np.linalg.svd(Jn, full_matrices=False)
    rank = int(np.sum(s > max(s[0] * threshold, np.finfo(float).eps))) if len(s) else 0
    Jp_projected = Jp - U[:, :rank] @ (U[:, :rank].T @ Jp)
    return np.linalg.svd(Jp_projected, compute_uv=False)


def solve_trans_biasacc_grav(lidar: LidarIncrements, imu: IMUSeries, R_LI: np.ndarray, time_offset: float, config: LIInitConfig | None = None) -> TranslationCalibrationResult:
    """Estimate IMU lever arm, accelerometer bias and gravity direction.

    Upstream's ``solve_trans_biasacc_grav`` jointly optimizes these nuisance
    parameters using LiDAR linear/angular acceleration. Here LI-Init's
    relationship is evaluated from a composed local LiDAR trajectory:

      f_I = R_LI [R_WL.T (a_WL - g_W) + (hat(alpha)+hat(omega)^2) p_IL] + b_a

    Gravity norm is fixed; its *direction* has two DoFs. Accel bias bounds
    default to upstream +/-0.01 m/s^2. Translation is only accepted when its
    marginalized information has rank 3. Bad excitation returns ``success=False``.
    """
    cfg = config or LIInitConfig()
    if imu.data_acc is None:
        raise ValueError("solve_trans_biasacc_grav requires IMU accelerometer data")
    R_LI = np.asarray(R_LI, dtype=np.float64)
    if R_LI.shape != (3, 3) or not np.allclose(R_LI.T @ R_LI, np.eye(3), atol=1e-5) or np.linalg.det(R_LI) < 0:
        raise ValueError("R_LI must belong to SO(3)")
    t_mid, R_WL, a_world, omega, alpha = lidar_linear_acceleration(lidar, cfg)
    query = t_mid + time_offset
    valid = (query >= imu.t[0]) & (query <= imu.t[-1])
    if np.count_nonzero(valid) < cfg.translation_min_samples:
        raise ValueError("Insufficient overlapping LiDAR and accelerometer samples")
    query, R_WL, a_world, omega, alpha = query[valid], R_WL[valid], a_world[valid], omega[valid], alpha[valid]
    measured = interpolate_vectors(imu.t, imu.data_acc, query)
    J = _hat_batch(alpha) + _hat_batch(omega) @ _hat_batch(omega)
    # Find a gravity direction from approximate zero-bias, zero-lever-arm data.
    uncorrected_world = np.einsum("nij,nj->ni", R_WL, measured @ R_LI)
    g_init = np.mean(a_world - uncorrected_world, axis=0)
    if np.linalg.norm(g_init) < 1e-8:
        g_init = np.array([0.0, 0.0, -1.0])
    g_init = cfg.gravity_norm * g_init / np.linalg.norm(g_init)
    axis1, axis2 = _tangent_basis(g_init / cfg.gravity_norm)
    bias0 = np.asarray(cfg.initial_accelerometer_bias, dtype=np.float64)
    p0 = np.asarray(cfg.initial_p_IL, dtype=np.float64)
    x0 = np.r_[p0, bias0, 0.0, 0.0]
    low = np.r_[[-np.inf] * 3, [cfg.accelerometer_bias_bounds[0]] * 3, [-np.inf] * 2]
    high = np.r_[[np.inf] * 3, [cfg.accelerometer_bias_bounds[1]] * 3, [np.inf] * 2]

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        g = Rotation.from_rotvec(x[6] * axis1 + x[7] * axis2).apply(g_init)
        return x[:3], x[3:6], g

    def residual(x: np.ndarray) -> np.ndarray:
        p, b, g = unpack(x)
        linear_acc_l = np.einsum("nji,nj->ni", R_WL, a_world - g)
        predicted = (linear_acc_l + np.einsum("nij,j->ni", J, p)) @ R_LI.T + b
        return (predicted - measured).ravel()

    opt = least_squares(residual, x0, bounds=(low, high), loss=cfg.loss, f_scale=cfg.f_scale, max_nfev=cfg.max_nfev, ftol=cfg.ftol, xtol=cfg.xtol, gtol=cfg.gtol)
    p_fit, b_fit, g_fit = unpack(opt.x)
    singular = _nuisance_marginalized_singular_values(opt.jac, cfg.translation_observability_rtol)
    threshold = max(float(singular[0]) * cfg.translation_observability_rtol, np.finfo(float).eps) if len(singular) else np.inf
    rank = int(np.count_nonzero(singular > threshold))
    observable = rank == 3
    success = bool(opt.success and observable)
    # An underconstrained solution must not be interpreted as a calibrated lever arm.
    p = p_fit if success else p0.copy()
    return TranslationCalibrationResult(p_IL=p, p_LI=-R_LI @ p, accelerometer_bias=b_fit, gravity_W=g_fit, success=success, observable=observable, message=str(opt.message) if observable else f"Translation underconstrained (marginalized rank {rank}); {opt.message}", rms_residual=float(np.sqrt(np.mean(residual(opt.x)**2))), num_samples=len(query), translation_singular_values=singular, translation_rank=rank, nfev=opt.nfev)
