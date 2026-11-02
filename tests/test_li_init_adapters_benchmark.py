"""Adapter/benchmark tests exercise real obscalib-like stream fields and failures."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.spatial.transform import Rotation

from baselines.li_init import IMUSeries, LidarIncrements, LIInitConfig
from baselines.li_init.adapters import (
    kaist_records_to_li_init, make_imu_series, make_lidar_increments,
    window_sample_to_li_init, window_batch_to_li_init,
    slice_li_init_window, reference_from_body_calibrations,
)
from baselines.li_init.benchmark import (
    BenchmarkCase, benchmark_li_init, summarize_li_init,
    rotation_error_degrees, calibration_update_rotation, cases_from_window_dataset,
)


def sample_records():
    t = np.arange(0, 5.0001, 0.05)
    delta_T = np.broadcast_to(np.eye(4), (len(t)-1, 4, 4)).copy()
    delta_T[:, :3, :3] = Rotation.from_rotvec(np.array([0.03, -0.02, 0.01])).as_matrix()
    imu_t = np.arange(-0.5, 5.501, 0.01)
    imu = SimpleNamespace(timestamps_s=imu_t, gyro_radps=np.zeros((len(imu_t), 3)), accel_mps2=np.zeros((len(imu_t), 3)))
    lidar = SimpleNamespace(scan_timestamps_s=t, relative_poses_se3=delta_T)
    return imu, lidar


def test_kaist_adapter_preserves_increments_and_timestamps():
    imu_record, lidar_record = sample_records()
    imu, lidar = kaist_records_to_li_init(imu_record, lidar_record)
    np.testing.assert_allclose(imu.t, imu_record.timestamps_s)
    np.testing.assert_allclose(lidar.t_start, lidar_record.scan_timestamps_s[:-1])
    np.testing.assert_allclose(lidar.t_end, lidar_record.scan_timestamps_s[1:])
    np.testing.assert_allclose(lidar.delta_T, lidar_record.relative_poses_se3)
    imu_cut, lidar_cut = slice_li_init_window(imu, lidar, 1., 2., imu_padding_s=.2)
    assert imu_cut.t[0] >= 0.8 - 1e-9
    assert imu_cut.t[-1] <= 2.2 + 1e-9
    assert len(lidar_cut.t_start) == 20


def test_window_sample_adapter_uses_interval_start_not_inferred_uniform_dt():
    imu_t = torch.tensor([0., .1, .2, .3, .4], dtype=torch.float64)
    lidar_start = torch.tensor([.01, .23], dtype=torch.float64)
    lidar_end = torch.tensor([.13, .34], dtype=torch.float64)
    I = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
    window = SimpleNamespace(streams={
        "gyro": SimpleNamespace(values=torch.zeros(5, 3), timestamps=imu_t),
        "accel": SimpleNamespace(values=torch.ones(5, 3), timestamps=imu_t),
        "lidar": SimpleNamespace(values=I, timestamps=lidar_end, interval_start_timestamps=lidar_start),
    }, current_calibration={"ignored": 123}, targets={"ignored": 456})
    imu, lidar = window_sample_to_li_init(window)
    np.testing.assert_allclose(lidar.t_start, [.01, .23])
    np.testing.assert_allclose(lidar.t_end, [.13, .34])
    assert imu.data_acc.shape == (5, 3)
    window.streams["lidar"].interval_start_timestamps = None
    with pytest.raises(ValueError, match="interval_start"):
        window_sample_to_li_init(window)


def test_window_batch_adapter_removes_padding():
    t = torch.tensor([[0., .1, .2, 0.]], dtype=torch.float64)
    mask = torch.tensor([[True, True, True, False]])
    sensor = SimpleNamespace(timestamps=t, values=torch.zeros((1, 4, 3)), sample_mask=mask, interval_start_timestamps=None)
    end = torch.tensor([[.1, .2, .3, 0.]], dtype=torch.float64)
    starts = torch.tensor([[0., .1, .2, 0.]], dtype=torch.float64)
    mask_l = torch.tensor([[True, True, True, False]])
    T = torch.eye(4).repeat(1, 4, 1, 1)
    T[0, -1] = torch.zeros((4, 4))
    lidar = SimpleNamespace(timestamps=end, interval_start_timestamps=starts, values=T, sample_mask=mask_l)
    batch = SimpleNamespace(streams={"gyro": sensor, "lidar": lidar})
    imu, inc = window_batch_to_li_init(batch, accel_key=None)
    assert len(imu.t) == 3 and len(inc.t_start) == 3
    np.testing.assert_allclose(inc.delta_T[:, 3, 3], np.ones(3))


def test_reference_from_body_frames_and_clock_sign():
    R_BI = Rotation.from_euler("xyz", [.3, -.1, .8]).as_matrix()
    R_BL = Rotation.from_euler("xyz", [-.3, .2, .4]).as_matrix()
    T_BI = np.eye(4); T_BI[:3, :3] = R_BI; T_BI[:3, 3] = [1, 2, 3]
    T_BL = np.eye(4); T_BL[:3, :3] = R_BL; T_BL[:3, 3] = [.7, 1.4, 2.8]
    reference = reference_from_body_calibrations(T_BI, T_BL, tau_imu_s=.02, tau_lidar_s=.12)
    np.testing.assert_allclose(reference.R_LI, R_BI.T @ R_BL)
    np.testing.assert_allclose(reference.p_IL, R_BL.T @ (T_BI[:3, 3] - T_BL[:3, 3]))
    assert abs(reference.time_offset - .1) < 1e-12
    assert rotation_error_degrees(reference.R_LI, reference.R_LI) < 1e-10
    prior = Rotation.from_rotvec([.12, -.15, .05]).as_matrix() @ reference.R_LI
    correction = calibration_update_rotation(prior, reference.R_LI)
    np.testing.assert_allclose(correction @ prior, reference.R_LI, atol=1e-12)


def _omega(t):
    return np.column_stack([.7*np.sin(3*t)+.25*np.cos(5*t), .5*np.sin(4.3*t+.2)+.1*np.cos(6*t), .6*np.cos(2.7*t+.7)-.2*np.sin(5.3*t)])


def test_benchmark_success_failure_and_summary():
    from baselines.li_init.adapters import CalibrationReference
    edges = np.arange(0, 5.000001, .04)
    mid = .5 * (edges[:-1] + edges[1:])
    T = np.broadcast_to(np.eye(4), (len(mid), 4, 4)).copy()
    T[:, :3, :3] = Rotation.from_rotvec(_omega(mid)*.04).as_matrix()
    lidar = LidarIncrements(edges[:-1], edges[1:], T)
    imu_t = np.arange(-.5, 5.50001, .0025)
    R = Rotation.from_rotvec(np.deg2rad(170)*np.array([1., 2., -1.])/np.sqrt(6)).as_matrix()
    tau = .12
    imu = IMUSeries(imu_t, _omega(imu_t-tau) @ R.T + np.array([.01, -.02, .015]))
    ref = CalibrationReference(R_LI=R, time_offset=tau)
    good = BenchmarkCase("ok", imu, lidar, ref, window_duration_s=5.0)
    # Identical time window but stationary IMU & LiDAR: explicitly underconstrained.
    still_T = np.broadcast_to(np.eye(4), T.shape).copy()
    bad = BenchmarkCase("bad", IMUSeries(imu_t, np.zeros((len(imu_t), 3))), LidarIncrements(edges[:-1], edges[1:], still_T), ref, window_duration_s=5.)
    df, outputs = benchmark_li_init([good, bad], LIInitConfig(rotation_initialization="svd", xcorr_step=.01, temporal_residual="interpolate", time_offset_bounds=(-.3,.3)), return_results=True)
    assert len(df) == 2 and bool(df.loc[0, "success"])
    assert not bool(df.loc[1, "success"])
    assert df.loc[0, "rotation_error_deg"] < 1.0
    assert df.loc[0, "time_offset_error_ms"] < 15
    summary = summarize_li_init(df)
    assert summary.loc[0, "n_windows"] == 2
    assert summary.loc[0, "n_success"] == 1
    assert summary.loc[0, "success_rate"] == .5
    assert "ok" in outputs
    assert df["elapsed_s"].notna().all()
