"""Focused synthetic verification of ROS-free angular LiDAR/IMU calibration."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from baselines.li_init import (
    IMUSeries, LidarIncrements, LIInitConfig, AngularVelocitySeries,
    central_diff, lidar_increments_to_motion, solve_Rotation_only, solve_Rot_bias_gyro,
)


def _true_rotation() -> np.ndarray:
    return Rotation.from_rotvec(np.deg2rad(175.0) * np.array([1., 2., -1.]) / np.sqrt(6)).as_matrix()


def _motion(t: np.ndarray) -> np.ndarray:
    return np.stack([
        0.7 * np.sin(3.0 * t) + .25 * np.cos(5.0 * t),
        .5 * np.sin(4.3 * t + .2) + .1 * np.cos(6.0 * t),
        .6 * np.cos(2.7 * t + .7) - .2 * np.sin(5.3 * t),
    ], axis=-1)


def _pair(dt: float = 0.04, delay: float = 0.0, bias: np.ndarray | None = None):
    edges = np.arange(0.0, 5.0 + 1e-10, dt)
    mid = (edges[:-1] + edges[1:]) / 2
    dR = Rotation.from_rotvec(_motion(mid) * dt).as_matrix()
    delta_T = np.tile(np.eye(4), (len(mid), 1, 1))
    delta_T[:, :3, :3] = dR
    increments = LidarIncrements(edges[:-1], edges[1:], delta_T)
    imu_t = np.arange(-0.75, 5.75, 0.005)
    bias = np.zeros(3) if bias is None else bias
    data_gyr = _motion(imu_t - delay) @ _true_rotation().T + bias
    return increments, IMUSeries(imu_t, data_gyr)


def _angle_error(Ra, Rb):
    return np.rad2deg(Rotation.from_matrix(Ra @ Rb.T).magnitude())


def test_odometry_to_angular_velocity_and_inverse_convention():
    lidar, _ = _pair()
    motion = lidar_increments_to_motion(lidar)
    assert np.max(np.abs(motion.ang_vel - _motion(motion.t))) < 1e-10
    T_inverse = lidar.delta_T.copy()
    T_inverse[:, :3, :3] = np.swapaxes(T_inverse[:, :3, :3], -1, -2)
    inverse = lidar_increments_to_motion(LidarIncrements(lidar.t_start, lidar.t_end, T_inverse, "end_to_start"))
    np.testing.assert_allclose(inverse.ang_vel, motion.ang_vel, atol=1e-10)
    derivative = central_diff(motion)
    assert derivative.ang_acc.shape == motion.ang_vel.shape


def test_near_pi_increment_uses_existing_so3_log():
    angle = np.deg2rad(179.99)
    dT = np.tile(np.eye(4), (2, 1, 1))
    dT[0, :3, :3] = Rotation.from_rotvec([angle, 0, 0]).as_matrix()
    dT[1, :3, :3] = Rotation.from_rotvec([0, angle, 0]).as_matrix()
    motion = lidar_increments_to_motion(LidarIncrements([0, 1], [1, 2], dT))
    np.testing.assert_allclose(np.linalg.norm(motion.ang_vel, axis=1), angle, atol=1e-8)


def test_rotation_only_recovers_175_degree_extrinsic():
    lidar, imu = _pair()
    config = LIInitConfig(estimate_gyro_bias=False, estimate_time_offset=False, rotation_initialization="svd")
    result = solve_Rotation_only(lidar_increments_to_motion(lidar), imu, config)
    assert result.success, result.message
    assert _angle_error(result.R_LI, _true_rotation()) < .05
    assert result.rotation_rank >= 2


def test_joint_taylor_refines_rotation_bias_and_small_time_offset():
    bias = np.array([0.015, -0.023, 0.009])
    delay = 0.013
    lidar, imu = _pair(delay=delay, bias=bias)
    config = LIInitConfig(initial_time_offset=0.0, time_offset_bounds=(-0.07, 0.07), temporal_residual="taylor", rotation_initialization="svd")
    result = solve_Rot_bias_gyro(lidar_increments_to_motion(lidar), imu, config)
    assert result.success, result.message
    assert _angle_error(result.R_LI, _true_rotation()) < 0.3
    assert abs(result.time_offset - delay) < 0.003
    np.testing.assert_allclose(result.gyro_bias, bias, atol=0.004)


def test_interpolated_joint_refinement():
    bias = np.array([0.015, -0.023, 0.009])
    delay = -0.12
    lidar, imu = _pair(delay=delay, bias=bias)
    config = LIInitConfig(initial_time_offset=-0.10, time_offset_bounds=(-0.2, -0.05), temporal_residual="interpolate", rotation_initialization="svd")
    result = solve_Rot_bias_gyro(lidar_increments_to_motion(lidar), imu, config)
    assert result.success, result.message
    assert _angle_error(result.R_LI, _true_rotation()) < 0.3
    assert abs(result.time_offset - delay) < 0.003
    np.testing.assert_allclose(result.gyro_bias, bias, atol=0.004)


def test_single_axis_motion_is_flagged_unobservable():
    t = np.linspace(0., 4.9, 100)
    w = np.stack([0 * t, 0 * t, np.sin(3*t)], axis=1)
    lidar = AngularVelocitySeries(t, w)
    imu = IMUSeries(t, w @ _true_rotation().T)
    result = solve_Rotation_only(lidar, imu, LIInitConfig(rotation_initialization="svd"))
    assert not result.observable
    assert not result.success
    assert result.rotation_rank == 1


def test_missing_time_coverage_is_explicit():
    lidar, imu = _pair()
    shifted = AngularVelocitySeries(lidar_increments_to_motion(lidar).t + 100, lidar_increments_to_motion(lidar).ang_vel)
    with pytest.raises(ValueError, match="overlapping"):
        solve_Rotation_only(shifted, imu)


def test_source_like_defaults_and_unimplemented_filter_request():
    cfg = LIInitConfig()
    assert cfg.rotation_initialization == "identity"
    assert cfg.temporal_residual == "taylor"
    assert cfg.loss == "linear"
    assert not cfg.filtering_enabled
    lidar, imu = _pair()
    with pytest.raises(NotImplementedError, match="preprocessing.py"):
        solve_Rotation_only(lidar_increments_to_motion(lidar), imu, LIInitConfig(filtering_enabled=True))
