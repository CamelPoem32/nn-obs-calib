"""Boreas-RT parsing, geometry, lazy access and partial-download contracts."""

import csv
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from boreas_dataset import (BoreasDataset, BoreasSequence, PoseTrajectory, accumulate_lidar_scans,
    discover_lidar_scans, discover_sequences, lidar_to_dmu, load_calibration, load_calibrations,
    load_imu, load_imu_file, load_lidar_poses, load_lidar_scan, motion_magnitudes,
    nearest_timestamps, relative_seconds, timestamp_statistics, timestamps_to_ns)
from boreas_dataset.imu import APPLANIX_COLUMNS, IMU_COLUMNS, IMU_SOURCES
from boreas_dataset.lidar import POSE_COLUMNS


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


@pytest.fixture
def sequence(tmp_path):
    root = tmp_path / "boreas-test"
    for source, (filename, unit, _) in IMU_SOURCES.items():
        if unit == "ns":
            times = ["1733248461000000001", "1733248461005000001", "1733248461025000001"]
        elif unit == "us":
            times = ["1733248461000000", "1733248461005000", "1733248461025000"]
        else:
            times = ["1733248461.000000001", "1733248461.005000001", "1733248461.025000001"]
        header = APPLANIX_COLUMNS if source == "applanix" else IMU_COLUMNS
        write_csv(root / filename, header, [[t, 1, 2, 3, 4, 5, 6] for t in times])
    write_csv(root / "applanix/lidar_poses.csv", POSE_COLUMNS,
              [[1733248461000000, 0, 0, 0, 0, 0, 0, .2, -.3, .7, 0, 0, 0],
               [1733248461100000, 1, 2, 0, 0, 0, 0, .3, -.4, .8, 0, 0, 0],
               [1733248461200000, 3, 2, 0, 0, 0, 0, .4, -.5, .9, 0, 0, 0]])
    (root / "lidar").mkdir()
    for t in [1733248461200000, 1733248461000000, 1733248461100000]:
        np.array([[1, 2, 3, 4, 5, -.05], [6, 7, 8, 9, 10, .05]], dtype="<f4").tofile(root / f"lidar/{t}.bin")
    (root / "calib").mkdir()
    T = np.eye(4)
    T[:3, 3] = [1, 2, 3]
    np.savetxt(root / "calib/T_applanix_lidar.txt", T)
    np.savetxt(root / "calib/T_applanix_dmu.txt", np.eye(4))
    return root


@pytest.mark.parametrize("source", IMU_SOURCES)
def test_imu_parsing_field_order_units_and_aliases(sequence, source):
    if source == "applanix":
        with pytest.warns(UserWarning, match="not independent"):
            imu = load_imu(sequence, source)
        expected_gyr, expected_acc = [3, 2, 1], [6, 5, 4]
    else:
        imu = load_imu(sequence, source)
        expected_gyr, expected_acc = [1, 2, 3], [4, 5, 6]
    assert imu.data_gyr.shape == imu.data_acc.shape == (3, 3)
    assert imu.t_utc is imu.timestamps_s
    np.testing.assert_array_equal(imu.data_gyr[0], expected_gyr)
    np.testing.assert_array_equal(imu.data_acc[0], expected_acc)
    assert imu.metadata["gyro_unit"] == "rad/s"
    assert imu.metadata["independent_of_ground_truth"] == (source != "applanix")
    assert imu.timestamps_s[0] == pytest.approx(1733248461, abs=1e-6)
    np.testing.assert_allclose(relative_seconds(imu.timestamps_ns), [0, .005, .025], atol=1e-12)
    assert imu.metadata["resampled"] == (source == "dmu_infilled")


def test_integer_precision_and_cross_sensor_alignment(sequence):
    imu = load_imu(sequence)
    assert imu.timestamps_ns[0] == 1733248461000000001
    aeva = load_imu(sequence, "aeva")
    assert imu.timestamps_ns[0] - aeva.timestamps_ns[0] == 1
    np.testing.assert_array_equal(timestamps_to_ns(["1.000000001"], "s"), [1000000001])
    with pytest.raises(ValueError):
        timestamps_to_ns(["1.0001"], "ns")


def test_gaps_and_closed_interval_selection(sequence):
    imu = load_imu(sequence)
    stats = timestamp_statistics(imu.timestamps_ns)
    np.testing.assert_array_equal(stats["gap_indices"], [1])
    # Use a fixed physical threshold for this short asymmetric sample.
    assert timestamp_statistics(imu.timestamps_ns, gap_threshold_s=.01)["gap_durations_s"][0] == .02
    window = load_imu(sequence, time_start="1733248461.005000001", time_end="1733248461.025000001")
    np.testing.assert_array_equal(window.timestamps_ns, imu.timestamps_ns[1:])
    assert imu.select(time_start=1733248462).data_acc.shape == (0, 3)
    with pytest.raises(ValueError):
        imu.select(time_start=2, time_end=1)


@pytest.mark.parametrize("rows", [
    [[100, 1, 2, 3, 4, 5, 6], [100, 1, 2, 3, 4, 5, 6]],
    [[200, 1, 2, 3, 4, 5, 6], [100, 1, 2, 3, 4, 5, 6]],
    [[100, 1, 2, 3, 4, 5, "nan"]], [[100, 1, 2]],
])
def test_malformed_imu_is_not_silently_repaired(tmp_path, rows):
    path = write_csv(tmp_path / "dmu_imu.csv", IMU_COLUMNS, rows)
    with pytest.raises(ValueError):
        load_imu_file(path)


def test_wrong_imu_header(tmp_path):
    path = write_csv(tmp_path / "dmu_imu.csv", APPLANIX_COLUMNS, [[100, 1, 2, 3, 4, 5, 6]])
    with pytest.raises(ValueError, match="header"):
        load_imu_file(path)


def test_binary_raw_fields_sorting_and_deterministic_sampling(sequence):
    scans = discover_lidar_scans(sequence)
    assert [p.path.stem for p in scans] == sorted(p.path.stem for p in scans)
    points = scans[0].load()
    assert points.dtype == np.dtype("float32")
    np.testing.assert_allclose(points[0], [1, 2, 3, 4, 5, -.05])
    np.testing.assert_array_equal(scans[0].load(max_points=1), points[:1])
    assert len(discover_lidar_scans(sequence, time_start="1733248461.1", time_end="1733248461.1")) == 1
    with pytest.raises(FileNotFoundError):
        discover_lidar_scans(sequence, source="aeva")


@pytest.mark.parametrize("payload", [b"", b"bad", b"\0" * 25])
def test_truncated_binary(tmp_path, payload):
    path = tmp_path / "1.bin"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="truncated"):
        load_lidar_scan(path)
    with pytest.raises(FileNotFoundError):
        load_lidar_scan(tmp_path / "missing.bin")


def test_pose_matches_official_nonzero_yaw_pitch_roll(sequence):
    trajectory = load_lidar_poses(sequence / "applanix/lidar_poses.csv")
    r, p, y = .2, -.3, .7
    roll = np.array([[1, 0, 0], [0, np.cos(r), np.sin(r)], [0, -np.sin(r), np.cos(r)]])
    pitch = np.array([[np.cos(p), 0, -np.sin(p)], [0, 1, 0], [np.sin(p), 0, np.cos(p)]])
    yaw = np.array([[np.cos(y), np.sin(y), 0], [-np.sin(y), np.cos(y), 0], [0, 0, 1]])
    np.testing.assert_allclose(trajectory.R[0], roll @ pitch @ yaw, atol=1e-15)
    np.testing.assert_array_equal(trajectory.timestamps_ns[:1], [1733248461000000000])
    np.testing.assert_allclose(trajectory.poses_T_W_L[1, :3, 3], [1, 2, 0])


def test_relative_transform_direction_identity_stride_and_interval(sequence):
    trajectory = load_lidar_poses(sequence / "applanix/lidar_poses.csv")
    relative = trajectory.relative()
    np.testing.assert_allclose(trajectory.poses_T_W_L[0] @ relative.relative_poses_se3[0], trajectory.poses_T_W_L[1], atol=1e-14)
    np.testing.assert_array_equal(relative.timestamps_s, trajectory.timestamps_s[1:])
    assert relative.metadata["independent_odometry"] is False
    stride = trajectory.relative(frame_stride=2)
    np.testing.assert_allclose(trajectory.poses_T_W_L[0] @ stride.relative_poses_se3[0], trajectory.poses_T_W_L[2], atol=1e-14)
    assert len(trajectory.relative(time_end="1733248461.1").timestamps_s) == 1
    repeated = PoseTrajectory([0, 1_000_000_000], np.repeat(trajectory.poses_T_W_L[:1], 2, axis=0))
    np.testing.assert_allclose(repeated.relative().relative_poses_se3[0], np.eye(4), atol=1e-15)
    translation, angle = motion_magnitudes(relative.relative_poses_se3)
    np.testing.assert_allclose(translation, [np.sqrt(5), 2])
    assert np.all(angle > 0)
    for bad_stride in (0, -1, 1.5):
        with pytest.raises(ValueError):
            trajectory.relative(frame_stride=bad_stride)


def test_exact_nearest_and_controlled_interpolation(sequence):
    trajectory = load_lidar_poses(sequence / "applanix/lidar_poses.csv")
    query = trajectory.timestamps_ns[:1] + 50_000_000
    with pytest.raises(ValueError, match="tolerance"):
        trajectory.match(query)
    np.testing.assert_array_equal(trajectory.match(query, tolerance_s=.05), trajectory.poses_T_W_L[:1])
    T = trajectory.match(query, tolerance_s=.05, interpolate=True, max_gap_s=.11)
    np.testing.assert_allclose(T[0, :3, 3], [.5, 1, 0])
    np.testing.assert_allclose(np.linalg.det(T[0, :3, :3]), 1)
    assert Rotation.from_matrix(T[0, :3, :3].T @ trajectory.R[0]).magnitude() > 0
    with pytest.raises(ValueError, match="max_gap_s"):
        trajectory.match(query, tolerance_s=.05, interpolate=True)
    with pytest.raises(ValueError, match="bracket"):
        trajectory.match(query, tolerance_s=.05, interpolate=True, max_gap_s=.09)
    with pytest.raises(ValueError, match="extrapolation"):
        trajectory.match([trajectory.timestamps_ns[0] - 1], tolerance_s=1, interpolate=True, max_gap_s=1)
    idx, delta = nearest_timestamps(trajectory.timestamps_ns, query, tolerance_s=.05)
    assert idx[0] == 0 and delta[0] == -.05
    idx, delta = nearest_timestamps(trajectory.timestamps_ns, query, tolerance_s=.01, allow_unmatched=True)
    assert idx[0] == -1 and delta[0] == -.05


def test_calibration_and_transform_composition(sequence, tmp_path):
    cal = load_calibrations(sequence)
    T = lidar_to_dmu(cal)
    np.testing.assert_allclose(T[:3, 3], [1, 2, 3])
    for bad in [np.zeros((4, 4)), np.diag([1, 1, -1, 1]), np.ones((3, 4))]:
        path = tmp_path / "bad.txt"
        np.savetxt(path, bad)
        with pytest.raises(ValueError):
            load_calibration(path)
    with pytest.raises(KeyError):
        lidar_to_dmu({})


def test_ground_truth_accumulation_uses_first_frame(sequence):
    scans = discover_lidar_scans(sequence)
    trajectory = load_lidar_poses(sequence / "applanix/lidar_poses.csv")
    cloud, local = accumulate_lidar_scans(scans[:2], trajectory, points_per_scan=1)
    np.testing.assert_allclose(local[0], np.eye(4), atol=1e-15)
    np.testing.assert_allclose(cloud[0], [1, 2, 3])
    np.testing.assert_allclose(cloud[1], local[1, :3, :3] @ [1, 2, 3] + local[1, :3, 3])


def test_discovery_lazy_construction_partial_data_and_environment(sequence, tmp_path, monkeypatch):
    monkeypatch.setenv("BOREAS_DATA_ROOT", str(sequence.parent))
    # Lazy access must not read binary files or IMU CSVs during construction.
    monkeypatch.setattr(np, "fromfile", lambda *a, **kw: pytest.fail("eager scan read"))
    dataset = BoreasDataset()
    assert dataset.sequences == [sequence.name]
    assert dataset.sequence(sequence.name).sensors["imu"] == list(IMU_SOURCES)
    assert discover_sequences(sequence) == [sequence.name]
    partial = tmp_path / "boreas-partial"
    partial.mkdir()
    s = BoreasSequence(partial)
    assert s.sensors["imu"] == [] and s.calibrations == {}
    assert not any(s.inventory()["expected_files"].values())
    assert s.summary()["duration_s"] == 0
    with pytest.raises(FileNotFoundError):
        s.imu("aeva")
    with pytest.raises(FileNotFoundError):
        s.lidar_scans()
    with pytest.raises(FileNotFoundError):
        dataset.sequence("missing")
    with pytest.raises(ValueError):
        load_imu(sequence, "unknown")


def test_real_boreas_smoke_if_available():
    """Only one complete cloud is decoded; no full point-cloud sequence is loaded."""
    from boreas_dataset.dataset import resolve_data_root
    root = resolve_data_root()
    if not root.is_dir():
        pytest.skip("Set BOREAS_DATA_ROOT to run the real-data smoke test")
    sequence = BoreasDataset(root).sequence()
    trajectory = sequence.lidar_poses()
    scans = sequence.lidar_scans()
    assert len(scans) > 0
    points = scans[0].load()
    assert points.shape[1] == 6 and points.shape[0] > 0
    trajectory.match([s.timestamp_ns for s in scans[:3]])
    time = scans[0].timestamp_s
    imu = sequence.imu(time_start=time, time_end=time + 1)
    assert imu.data_gyr.shape[0] > 0
    local = trajectory.select(time_start=time, time_end=time + 1).relative()
    assert np.isfinite(motion_magnitudes(local.relative_poses_se3)[1]).all()
    assert lidar_to_dmu(sequence.calibrations).shape == (4, 4)
