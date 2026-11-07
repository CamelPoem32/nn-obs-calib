"""Aeva decoding, independent registration and all-sequence output contracts."""

import numpy as np
import pytest

from boreas_dataset import BoreasDataset, LidarData, discover_lidar_scans, load_lidar_scan
from boreas_dataset.calibration import calibration_direction_diagnostics
from boreas_dataset.import_lidar_odometry import build_argument_parser, run
from boreas_dataset.odometry import compute_lidar_odometry, load_lidar_data_csv, save_lidar_data_csv
from boreas_dataset.sensor_statistics import lidar_statistics


def write_scans(root, source, points, times=(1733248461000000, 1733248461100000, 1733248461200000)):
    folder = root / ("lidar" if source == "velodyne" else "aeva")
    folder.mkdir(parents=True, exist_ok=True)
    fields = 6 if source == "velodyne" else 10
    for index, stamp in enumerate(times):
        raw = np.zeros((len(points), fields), dtype="<f4")
        raw[:, :3] = points - np.array([.04 * index, -.02 * index, .01 * index])
        raw[:, 3:6] = 1
        if source == "aeva":
            raw[:, 7] = np.linspace(6e6, 99e6, len(points))
        raw.tofile(folder / f"{stamp}.bin")


def test_aeva_layout_flags_and_source_discovery(tmp_path):
    root = tmp_path / "boreas-test"
    write_scans(root, "aeva", np.array([[1, 2, 3], [4, 5, 6]]))
    scans = discover_lidar_scans(root, source="aeva")
    assert scans[0].frame_id == "aeva" and scans[0].timestamp_reference == "start"
    raw = scans[0].load()
    raw[:, 8:] = np.array([np.nan, np.inf], dtype="<f4")  # bit patterns are opaque, not numerical measurements
    raw.tofile(scans[0].path)
    loaded = load_lidar_scan(scans[0].path, source="aeva")
    assert loaded.shape == (2, 10)
    np.testing.assert_array_equal(loaded.view("<u4"), raw.view("<u4"))
    assert scans[0].load(max_points=1).shape == (1, 10)
    assert BoreasDataset(tmp_path).sequence(root.name).sensors["lidar"] == ["aeva"]
    assert "aeva" not in BoreasDataset(tmp_path).sequence(root.name).sensors["detected_unimplemented"]
    with pytest.raises(ValueError, match="truncated"):
        load_lidar_scan(scans[0].path, source="velodyne")
    scans[0].path.write_bytes(b"short")
    with pytest.raises(ValueError, match="truncated"):
        scans[0].load()


def test_lidar_statistics_counts_full_files_and_bounds_sample(tmp_path):
    root = tmp_path / "boreas-test"
    write_scans(root, "aeva", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    stats = lidar_statistics(root, source="aeva", sample_scans=2, max_points_per_scan=1)
    assert stats["scans"] == 3 and stats["points_per_scan_p50"] == 3
    assert stats["sampled_scans"] == 2 and stats["sampled_points"] == 2
    assert stats["scan_time_reference"] == "start" and stats["doppler_available"]
    assert stats["median_rate_hz"] == pytest.approx(10)


def fake_odometry():
    ns = np.array([1733248461000000000, 1733248461100000000, 1733248461200000000])
    from boreas_dataset import timestamps_ns_to_s
    times = timestamps_ns_to_s(ns)
    return LidarData(times[1:], np.repeat(np.eye(4)[None], 2, axis=0), times,
                     fitness=np.array([.9, .8]), inlier_rmse=np.array([.01, .02]),
                     metadata={"independent_odometry": True, "scan_timestamps_ns": ns.tolist()})


def test_normalized_csv_roundtrip_and_invalid_intervals(tmp_path):
    original = fake_odometry()
    path = save_lidar_data_csv(original, tmp_path / "odometry.csv")
    result = load_lidar_data_csv(path)
    np.testing.assert_array_equal(result.relative_poses_se3, original.relative_poses_se3)
    np.testing.assert_allclose(result.timestamps_s, original.timestamps_s, rtol=0, atol=3e-7)
    assert result.metadata == original.metadata
    import pandas as pd
    df = pd.read_csv(path)
    df.loc[1, "scan_start_timestamp_s"] += 1
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="inconsistent"):
        load_lidar_data_csv(path)


def test_cli_all_sequences_both_sources_direct_files_and_skip(tmp_path, monkeypatch):
    roots = [tmp_path / "boreas-one", tmp_path / "boreas-two"]
    for root in roots:
        write_scans(root, "velodyne", np.array([[1, 2, 3], [2, 3, 4]]))
    write_scans(roots[0], "aeva", np.array([[1, 2, 3], [2, 3, 4]]))
    calls = []
    def compute(root, **kwargs):
        calls.append((root, kwargs))
        return fake_odometry()
    monkeypatch.setattr("boreas_dataset.import_lidar_odometry.compute_lidar_odometry", compute)
    parser = build_argument_parser()
    dry = run(parser.parse_args([str(tmp_path), "--dry-run"]))
    assert len(dry["planned"]) == 3 and not calls
    report = run(parser.parse_args([str(tmp_path), "--verbosity", "0"]))
    assert len(report["saved"]) == 3 and not report["failed"]
    assert (roots[0] / "lidar_odometry_aeva.csv").is_file()
    assert (roots[1] / "lidar_odometry_velodyne.csv").is_file()
    assert not list(tmp_path.rglob("*.partial"))
    assert sorted(p.name for p in roots[0].iterdir() if p.is_dir()) == ["aeva", "lidar"]
    report = run(parser.parse_args([str(tmp_path), "--verbosity", "0"]))
    assert len(report["skipped"]) == 4 and len(calls) == 3
    run(parser.parse_args([str(tmp_path), "--sequences", "boreas-one", "--sources", "aeva", "--overwrite", "--verbosity", "0"]))
    assert len(calls) == 4


def test_cli_failure_keeps_previous_csv_and_continues_other_sensor(tmp_path, monkeypatch):
    root = tmp_path / "boreas-test"
    for source in ("velodyne", "aeva"):
        write_scans(root, source, np.array([[1, 2, 3], [2, 3, 4]]))
    destination = root / "lidar_odometry_velodyne.csv"
    destination.write_text("previous successful output")
    def compute(root, source, **kwargs):
        if source == "velodyne":
            raise RuntimeError("registration failed")
        return fake_odometry()
    monkeypatch.setattr("boreas_dataset.import_lidar_odometry.compute_lidar_odometry", compute)
    report = run(build_argument_parser().parse_args([str(tmp_path), "--overwrite", "-v", "0"]))
    assert len(report["failed"]) == len(report["saved"]) == 1
    assert destination.read_text() == "previous successful output"


@pytest.mark.parametrize("source", ["velodyne", "aeva"])
def test_registration_recovers_current_to_previous_direction_without_gt(tmp_path, source):
    pytest.importorskip("open3d")
    rng = np.random.default_rng(123)
    plane = rng.uniform(-3, 3, (1200, 3))
    plane[:400, 0] = -3
    plane[400:800, 1] = 3
    plane[800:, 2] = -2
    root = tmp_path / "boreas-test"
    write_scans(root, source, plane)
    # There are deliberately no applanix, calibration or IMU files to consult.
    result = compute_lidar_odometry(root, source=source, voxel_size_m=.2,
        max_correspondence_distance_m=.5, max_iterations=30, local_map_scans=2, robust_kernel="l2")
    np.testing.assert_allclose(result.relative_poses_se3[:, :3, 3], [[.04, -.02, .01]] * 2, atol=.02)
    np.testing.assert_allclose(result.relative_poses_se3[:, :3, :3], np.repeat(np.eye(3)[None], 2, axis=0), atol=.01)
    np.testing.assert_array_equal(result.timestamps_s, result.scan_timestamps_s[1:])
    assert result.metadata["independent_odometry"] and not result.metadata["uses_ground_truth"]
    assert np.all(result.fitness > .8)


def test_real_aeva_and_extrinsic_direction_consistency_if_available():
    from boreas_dataset.dataset import resolve_data_root
    if not resolve_data_root().is_dir():
        pytest.skip("Boreas dataset unavailable")
    sequence = BoreasDataset().sequence()
    scans = sequence.lidar_scans(source="aeva")
    assert scans[0].load(max_points=100).shape == (100, 10)
    assert sequence.lidar_poses(source="aeva").metadata["sensor_frame"] == "aeva"
    rows = calibration_direction_diagnostics(sequence.root, samples=10)
    for row in rows:
        if row["hypothesis"] == "stored convention":
            assert row["median_position_error_m"] < .001
            assert row["median_rotation_error_deg"] < .01
        else:
            assert row["median_rotation_error_deg"] > 10
