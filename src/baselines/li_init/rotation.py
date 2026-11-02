"""ROS-free analogues of LI-Init's solve_Rotation_only/solve_Rot_bias_gyro.

The upstream Ceres residuals are angular-velocity alignment and a first-order
Taylor time-shift correction.  SVD (optional) adds a global SO(3) initializer.
This module works with LiDAR *relative* odometry and does not need LiDAR poses.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .lidar_motion import central_diff
from .structures import AngularVelocitySeries, IMUSeries, LIInitConfig, RotationCalibrationResult


def _as_imu_motion(imu: IMUSeries | AngularVelocitySeries) -> AngularVelocitySeries:
    if isinstance(imu, IMUSeries):
        return AngularVelocitySeries(t=imu.t, ang_vel=imu.data_gyr)
    if isinstance(imu, AngularVelocitySeries):
        return imu
    raise TypeError("imu must be IMUSeries or AngularVelocitySeries")


def _interpolate(t: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    if query.size and (query.min() < t[0] - 1e-10 or query.max() > t[-1] + 1e-10):
        raise ValueError("IMU must cover the queried LiDAR times; no extrapolation is performed")
    query = np.clip(query, t[0], t[-1])
    return np.column_stack([np.interp(query, t, values[:, axis]) for axis in range(3)])


def _match(lidar: AngularVelocitySeries, imu: AngularVelocitySeries, offset: float, min_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = lidar.t + offset
    valid = (query >= imu.t[0]) & (query <= imu.t[-1])
    if np.count_nonzero(valid) < min_samples:
        raise ValueError(f"Only {np.count_nonzero(valid)} overlapping angular-velocity samples; need at least {min_samples}")
    return lidar.t[valid], lidar.ang_vel[valid], _interpolate(imu.t, imu.ang_vel, query[valid])


def _rank(s: np.ndarray, rtol: float) -> int:
    return int(np.count_nonzero(s > max(s[0] * rtol, np.finfo(float).eps))) if s.size else 0


def _align_rotation(omega_l: np.ndarray, omega_i: np.ndarray, estimate_bias: bool, rtol: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """SO(3) Kabsch/Wahba initializer, optionally eliminating constant gyro bias.

    With bias enabled, center both streams before aligning; ``b_g`` is then
    their mean difference after rotation.  This is an improvement over the
    original identity-initialized solve_Rotation_only() Ceres optimization.
    """

    x = omega_l - omega_l.mean(axis=0) if estimate_bias else omega_l
    y = omega_i - omega_i.mean(axis=0) if estimate_bias else omega_i
    _, s, _ = np.linalg.svd(x, full_matrices=False)
    rank = _rank(s, rtol)
    H = x.T @ y
    U, _, Vt = np.linalg.svd(H)
    correction = np.diag([1.0, 1.0, np.linalg.det(Vt.T @ U.T)])
    R_LI = Vt.T @ correction @ U.T
    bias = omega_i.mean(axis=0) - R_LI @ omega_l.mean(axis=0) if estimate_bias else np.zeros(3)
    return R_LI, bias, s, rank


def _result(R_LI: np.ndarray, gyro_bias: np.ndarray, offset: float, success: bool, message: str, residual: np.ndarray, omega_l: np.ndarray, cfg: LIInitConfig, nfev: int, joint_jac: np.ndarray | None = None, bias_free: bool = False) -> RotationCalibrationResult:
    x = omega_l - omega_l.mean(axis=0) if bias_free else omega_l
    s = np.linalg.svd(x, compute_uv=False)
    rank = _rank(s, cfg.observability_rtol)
    joint_s = np.linalg.svd(joint_jac, compute_uv=False) if joint_jac is not None else None
    joint_rank = _rank(joint_s, cfg.observability_rtol) if joint_s is not None else None
    observable = rank >= 2
    return RotationCalibrationResult(R_LI=R_LI, gyro_bias=gyro_bias, time_offset=float(offset), success=bool(success and observable), observable=observable, message=message if observable else f"Rotation underconstrained (excitation rank {rank}); {message}", rms_residual=float(np.sqrt(np.mean(residual**2))), num_samples=len(omega_l), rotation_singular_values=s, rotation_rank=rank, joint_singular_values=joint_s, joint_rank=joint_rank, nfev=nfev)


def solve_Rotation_only(lidar: AngularVelocitySeries, imu: IMUSeries | AngularVelocitySeries, config: LIInitConfig | None = None, time_offset: float | None = None) -> RotationCalibrationResult:
    """LI-Init's rotation-only stage: minimize ``R_LI @ omega_L - omega_I``.

    The optional 'svd' method gives a global optimum for this Wahba objective.
    By default, 'identity' reproduces the source's initial rotation with local
    SciPy optimization. No gyro bias is estimated in this stage.
    """

    cfg = config or LIInitConfig()
    if cfg.filtering_enabled:
        raise NotImplementedError("Filtering is not implemented until preprocessing.py is added")
    imu_motion = _as_imu_motion(imu)
    offset = cfg.initial_time_offset if time_offset is None else float(time_offset)
    _, omega_l, omega_i = _match(lidar, imu_motion, offset, cfg.min_samples)
    R_svd, _, _, _ = _align_rotation(omega_l, omega_i, False, cfg.observability_rtol)
    if cfg.rotation_initialization == "svd":
        R_final = R_svd
        nfev = 0
        message = "SO(3) SVD alignment"
        success = True
    else:
        opt = least_squares(lambda phi: (omega_l @ Rotation.from_rotvec(phi).as_matrix().T - omega_i).ravel(), np.zeros(3), max_nfev=cfg.max_nfev, loss=cfg.loss, f_scale=cfg.f_scale, ftol=cfg.ftol, xtol=cfg.xtol, gtol=cfg.gtol)
        R_final = Rotation.from_rotvec(opt.x).as_matrix()
        nfev, message, success = opt.nfev, opt.message, opt.success
    residual = omega_l @ R_final.T - omega_i
    return _result(R_final, np.zeros(3), offset, success, str(message), residual, omega_l, cfg, nfev)


def solve_Rot_bias_gyro(lidar: AngularVelocitySeries, imu: IMUSeries | AngularVelocitySeries, config: LIInitConfig | None = None, initial_R_LI: np.ndarray | None = None) -> RotationCalibrationResult:
    """Joint rotation/gyro-bias/time refinement, retaining LI-Init's function name.

    The default 'taylor' residual follows upstream Angular_Vel_Cost:
        R_LI omega_L - omega_I(t0) - (tau-t0) dot(omega_I)(t0) + b_g.
    'interpolate' evaluates omega_I(t_L+tau) directly, without extrapolation.
    In both cases, ``tau`` means t_imu = t_lidar + tau.

    This is LOCAL temporal refinement. The coarse xcorr_temporal_init() stage
    will be implemented in temporal.py and is needed for large clock offsets.
    """

    cfg = config or LIInitConfig()
    if cfg.filtering_enabled:
        raise NotImplementedError("Filtering is not implemented until preprocessing.py is added")
    imu_motion = _as_imu_motion(imu)
    offset0 = cfg.initial_time_offset
    t, omega_l, omega_i0 = _match(lidar, imu_motion, offset0, cfg.min_samples)
    if initial_R_LI is not None:
        R0 = np.asarray(initial_R_LI, dtype=float)
        if R0.shape != (3, 3) or not np.allclose(R0.T @ R0, np.eye(3), atol=1e-5) or np.linalg.det(R0) < 0:
            raise ValueError("initial_R_LI must belong to SO(3)")
    elif cfg.rotation_initialization == "svd":
        R0, _, _, _ = _align_rotation(omega_l, omega_i0, cfg.estimate_gyro_bias, cfg.observability_rtol)
    else:
        R0 = np.eye(3)
    bias0 = np.asarray(cfg.initial_gyro_bias, dtype=float)
    if cfg.rotation_initialization == "svd" and cfg.estimate_gyro_bias and initial_R_LI is None:
        bias0 = omega_i0.mean(axis=0) - R0 @ omega_l.mean(axis=0)

    # Under interpolation mode, require coverage throughout the entire bounded
    # temporal search interval. This fixes the residual vector length for SciPy.
    if cfg.temporal_residual == "interpolate" and cfg.estimate_time_offset:
        lo, hi = cfg.time_offset_bounds
        valid = (t + lo >= imu_motion.t[0]) & (t + hi <= imu_motion.t[-1])
        if np.count_nonzero(valid) < cfg.min_samples:
            raise ValueError("Insufficient IMU padding for the requested time_offset_bounds; reduce bounds or add IMU samples")
        t, omega_l, omega_i0 = t[valid], omega_l[valid], omega_i0[valid]

    if cfg.temporal_residual == "taylor":
        derivative = imu_motion if imu_motion.ang_acc is not None else central_diff(imu_motion)
        angular_acc = _interpolate(derivative.t, derivative.ang_acc, t + offset0)

    x0 = np.concatenate([np.zeros(3), bias0 if cfg.estimate_gyro_bias else np.empty(0), [offset0] if cfg.estimate_time_offset else np.empty(0)])
    low = np.full(x0.size, -np.inf)
    high = np.full(x0.size, np.inf)
    if cfg.estimate_time_offset:
        low[-1], high[-1] = cfg.time_offset_bounds

    def _unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        R_LI = Rotation.from_rotvec(x[:3]).as_matrix() @ R0
        pos = 3
        bias = x[pos:pos + 3] if cfg.estimate_gyro_bias else bias0
        pos += 3 if cfg.estimate_gyro_bias else 0
        tau = float(x[pos]) if cfg.estimate_time_offset else offset0
        return R_LI, bias, tau

    def residual(x: np.ndarray) -> np.ndarray:
        R_LI, bias, tau = _unpack(x)
        if cfg.temporal_residual == "taylor":
            target = omega_i0 + (tau - offset0) * angular_acc
        else:
            target = _interpolate(imu_motion.t, imu_motion.ang_vel, t + tau)
        return (omega_l @ R_LI.T + bias - target).ravel()

    opt = least_squares(residual, x0, bounds=(low, high), max_nfev=cfg.max_nfev, loss=cfg.loss, f_scale=cfg.f_scale, ftol=cfg.ftol, xtol=cfg.xtol, gtol=cfg.gtol)
    R_LI, bias, tau = _unpack(opt.x)
    return _result(R_LI, bias, tau, opt.success, str(opt.message), residual(opt.x), omega_l, cfg, opt.nfev, joint_jac=opt.jac, bias_free=cfg.estimate_gyro_bias)
