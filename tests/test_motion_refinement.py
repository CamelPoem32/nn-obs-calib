"""Integration, frame-convention, observability and solver tests on known rigid motion."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from baselines.motion_refinement import MotionRefinementCalibrator, MotionRefinementConfig, IMUSeries, LidarIncrements, integrate_gyro_intervals, align_lidar_motion_pairs
from baselines.motion_refinement.geometry import invert_T


def rotation_error(a, b):
    return np.rad2deg(Rotation.from_matrix(a @ b.T).magnitude())


def dataset(duration=2.5, lidar_dt=0.1, gyro_bias=(0, 0, 0), motion='excited'):
    t = np.arange(0, duration + 0.015, 0.01)
    if motion == 'excited':
        omega = np.column_stack([0.7*np.sin(1.7*t) + .15, 0.4*np.cos(2.4*t) - .05, 0.6*np.cos(.8*t) + .1])
    elif motion == 'static':
        omega = np.zeros((len(t), 3))
    else:
        omega = np.column_stack([np.zeros_like(t), np.zeros_like(t), 0.4 + 0*t])
    bias = np.asarray(gyro_bias, dtype=float)
    imu = IMUSeries(t=t, data_gyr=omega + bias)
    starts = np.arange(0., duration - 0.1 + 1e-8, lidar_dt)
    ends = starts + lidar_dt
    true_R = Rotation.from_rotvec([1.0, -.6, .3]).as_matrix()
    T_I_L = np.eye(4)
    T_I_L[:3, :3] = true_R
    T_I_L[:3, 3] = [.32, -.22, .15]
    T_I_R = np.eye(4)
    T_I_R[:3, :3] = Rotation.from_rotvec([.3, -.2, .8]).as_matrix()
    T_I_R[:3, 3] = [-.45, .12, .35]

    delta_body_R = integrate_gyro_intervals(IMUSeries(t, omega), starts, ends)
    delta_body = np.repeat(np.eye(4)[None], len(starts), axis=0)
    delta_body[:, :3, :3] = delta_body_R
    for k, a in enumerate(starts):
        delta_body[k, :3, 3] = [0.15 * np.cos(a), 0.08 * np.sin(a * 1.2), 0.06 * np.cos(a * 0.6)]
    d_l = invert_T(T_I_L) @ delta_body @ T_I_L
    d_r = invert_T(T_I_R) @ delta_body @ T_I_R
    lidar = LidarIncrements(starts, ends, d_l)
    right = LidarIncrements(starts, ends, d_r)
    return imu, lidar, right, T_I_L, T_I_R


def prior(T, degrees=12.0, axis=(.3, -.6, .74)):
    out = T.copy()
    a = np.asarray(axis, dtype=float)
    a /= np.linalg.norm(a)
    out[:3, :3] = Rotation.from_rotvec(np.deg2rad(degrees)*a).as_matrix() @ T[:3, :3]
    return out


def test_preintegration_constant_rate_exact_and_interval_convention():
    t = np.arange(0, 2.01, .01)
    omega = np.tile(np.array([.2, -.3, .5]), (len(t), 1))
    imu = IMUSeries(t, omega)
    out = integrate_gyro_intervals(imu, np.array([.1, .5, 1.1]), np.array([.4, 1., 1.8]))
    for k, dt in enumerate([.3, .5, .7]):
        np.testing.assert_allclose(out[k], Rotation.from_rotvec(omega[0]*dt).as_matrix(), atol=2e-11)


def test_gyro_fine_rotation_converges_from_12_degrees():
    imu, lidar, right, gt, tr = dataset()
    x0 = prior(gt)
    config = MotionRefinementConfig(max_iterations=8, use_lidar_pair='off', prior_rotation_sigma_deg=90.0, gyro_rotation_sigma_rad=.004, loss='linear')
    r = MotionRefinementCalibrator(config).fit(imu, lidar, initial_T_I_L=x0)
    print('GYRO', rotation_error(r.R_LI, gt[:3, :3]), r.initial_cost, r.final_cost, r.observability.singular_values, r.message)
    assert r.success and r.final_cost < r.initial_cost
    assert rotation_error(r.R_LI, gt[:3, :3]) < 1.0
    assert r.observability.rank == 3


def test_two_lidar_fine_rotation_converges_with_corresponding_odometry():
    imu, lidar, right, gt, tr = dataset()
    x0 = prior(gt, 10.0)
    cfg = MotionRefinementConfig(max_iterations=8, use_lidar_pair='required', gyro_rotation_sigma_rad=.01, lidar_pair_rotation_sigma_rad=.01, lidar_pair_translation_sigma_m=.04, prior_rotation_sigma_deg=90.0, loss='linear')
    r = MotionRefinementCalibrator(cfg).fit(imu, lidar, initial_T_I_L=x0, right_lidar=right, T_I_R=tr)
    print('DUAL', rotation_error(r.R_LI, gt[:3, :3]), r.initial_cost, r.final_cost, r.lidar_pair_translation_rms_m)
    assert r.success and r.observable and r.lidar_pair_factor_count >= 10
    assert rotation_error(r.R_LI, gt[:3, :3]) < 1.
    assert r.lidar_pair_translation_rms_m < 0.02
    np.testing.assert_allclose(r.T_I_L[:3,3], gt[:3,3])


def test_pair_only_uses_translational_consistency():
    imu, lidar, right, gt, tr = dataset()
    x0 = prior(gt, 15)
    cfg = MotionRefinementConfig(use_gyro_factor=False, use_lidar_pair='required', max_iterations=10, lidar_pair_translation_sigma_m=.01, lidar_pair_rotation_sigma_rad=.01, prior_rotation_sigma_deg=90, loss='linear')
    r = MotionRefinementCalibrator(cfg).fit(imu, lidar, initial_T_I_L=x0, right_lidar=right, T_I_R=tr)
    print('PAIR', rotation_error(r.R_LI, gt[:3,:3]),r.observability.singular_values)
    assert rotation_error(r.R_LI, gt[:3,:3]) < 1


def test_exact_initial_calibration_stays_correct():
    imu, lidar, right, gt, tr = dataset()
    cfg = MotionRefinementConfig(use_lidar_pair='required', prior_rotation_sigma_deg=10)
    r = MotionRefinementCalibrator(cfg).fit(imu, lidar, initial_T_I_L=gt, right_lidar=right, T_I_R=tr)
    assert rotation_error(r.R_LI, gt[:3, :3]) < .01


def test_static_window_marks_unobservable_and_retains_prior():
    imu, lidar, right, gt, tr = dataset(motion='static')
    cfg = MotionRefinementConfig(use_lidar_pair='off', max_iterations=4)
    x0 = prior(gt, 22)
    r = MotionRefinementCalibrator(cfg).fit(imu, lidar, initial_T_I_L=x0)
    assert not r.observable
    assert r.observability.rank == 0
    assert rotation_error(r.R_LI, x0[:3,:3]) < 1e-5


def test_pair_factor_rejects_wrong_reference_input():
    imu, lidar, right, gt, tr = dataset()
    with pytest.raises(ValueError):
        MotionRefinementCalibrator().fit(imu, lidar, initial_T_I_L=gt, right_lidar=right)


def test_pair_alignment_inverse_convention_and_right_clock_shift():
    imu, lidar, right, gt, tr = dataset()
    inverses = LidarIncrements(right.t_start + .06, right.t_end + .06, invert_T(right.delta_T), increment_convention='end_to_start')
    paired = align_lidar_motion_pairs(lidar, inverses, right_time_offset_s=.06)
    np.testing.assert_allclose(paired.left, lidar.delta_T, atol=1e-10)
    np.testing.assert_allclose(paired.right, right.delta_T, atol=1e-10)


def test_gyro_offset_requires_padding():
    imu, lidar, right, gt, tr = dataset()
    cfg = MotionRefinementConfig(optimize_time_offset=True, use_lidar_pair='off')
    with pytest.raises(ValueError, match='padding'):
        MotionRefinementCalibrator(cfg).fit(imu, lidar, initial_T_I_L=gt)


def test_joint_gyro_bias_refinement():
    true_bias = np.array([.012, -.018, .009])
    imu, lidar, right, gt, tr = dataset(duration=3.5, gyro_bias=true_bias)
    init = prior(gt, 5.)
    cfg = MotionRefinementConfig(max_iterations=12, use_lidar_pair='off', optimize_gyro_bias=True, use_prior=True, prior_rotation_sigma_deg=90, prior_gyro_bias_sigma_radps=.2, gyro_rotation_sigma_rad=.003, loss='linear')
    result = MotionRefinementCalibrator(cfg).fit(imu, lidar, initial_T_I_L=init)
    print('BIAS', result.gyro_bias, true_bias, rotation_error(result.R_LI, gt[:3,:3]))
    assert rotation_error(result.R_LI, gt[:3,:3]) < 1.0
    assert np.linalg.norm(result.gyro_bias - true_bias) < .015


def test_joint_time_offset_refinement_with_padded_imu():
    duration = 3.5
    reference_t = np.arange(-.4, duration + .42, .01)
    gyro_reference = np.column_stack([.6*np.sin(reference_t*2.2), .4*np.cos(reference_t*1.6), .5*np.sin(reference_t*1.1) + .25])
    known_offset = .075
    shifted_imu = IMUSeries(t=reference_t + known_offset, data_gyr=gyro_reference)
    ideal_imu = IMUSeries(t=reference_t, data_gyr=gyro_reference)
    starts = np.arange(0, duration - .1, .1)
    ends = starts + .1
    R_true = Rotation.from_rotvec([.4, -.6, .8]).as_matrix()
    gyro_rot = integrate_gyro_intervals(ideal_imu, starts, ends)
    L = np.repeat(np.eye(4)[None], len(starts), axis=0)
    L[:, :3, :3] = R_true.T @ gyro_rot @ R_true
    lidar = LidarIncrements(starts, ends, L)
    cfg = MotionRefinementConfig(max_iterations=15, use_lidar_pair='off', optimize_time_offset=True, time_offset_bounds=(-.2,.2), prior_rotation_sigma_deg=90, prior_time_offset_sigma_s=.2, gyro_rotation_sigma_rad=.002, loss='linear', max_rotation_step_deg=8)
    result = MotionRefinementCalibrator(cfg).fit(shifted_imu, lidar, initial_R_LI=R_true, initial_time_offset=.0)
    print('TIME', result.time_offset, known_offset, rotation_error(result.R_LI, R_true))
    assert abs(result.time_offset - known_offset) < .02
    assert rotation_error(result.R_LI, R_true) < 1.0


def test_warm_start_rollout_uses_last_solution_and_keeps_initial_reference_out_of_fit():
    from baselines.motion_refinement import MotionRefinementWindow, run_motion_refinement_rollout
    imu, lidar, right, gt, tr = dataset(duration=3.5)
    windows = []
    for k in range(3):
        a = k * 1.0
        b = a + 1.0
        select_lidar = (lidar.t_start >= a) & (lidar.t_end <= b + 1e-8)
        select_imu = (imu.t >= a) & (imu.t <= b + 1e-8)
        windows.append(MotionRefinementWindow(name=f'chunk{k}', imu=IMUSeries(imu.t[select_imu], imu.data_gyr[select_imu]), lidar=LidarIncrements(lidar.t_start[select_lidar], lidar.t_end[select_lidar], lidar.delta_T[select_lidar]), t_start=a, t_end=b))
    cfg = MotionRefinementConfig(use_lidar_pair='off', prior_rotation_sigma_deg=90, gyro_rotation_sigma_rad=.003, loss='linear')
    df, fits = run_motion_refinement_rollout(windows, MotionRefinementCalibrator(cfg), initial_T_I_L=prior(gt, 10), reference_R_LI=gt[:3, :3])
    assert df.step.to_list() == [0,1,2,3]
    assert df.iloc[-1].error_deg < df.iloc[0].error_deg
    assert len(fits) == 3 and all(f is not None for f in fits)
