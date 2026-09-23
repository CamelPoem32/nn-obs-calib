"""Tests for temporal construction of multi-sensor windows."""

import pytest
import torch

from obscalib.config import WindowingConfig
from obscalib.data import SensorStream, StreamWindow, build_windows


def _make_stream(timestamps: torch.Tensor, feature_dim: int = 1) -> SensorStream:
    """Create a simple stream whose values identify their original sample indices."""

    values = torch.arange(timestamps.numel(), dtype=torch.float64).unsqueeze(-1).repeat(1, feature_dim)
    return SensorStream(values=values, timestamps=timestamps.to(dtype=torch.float64))


def test_build_windows_constructs_non_overlapping_windows_by_default() -> None:
    streams = {
        "gyroscope": _make_stream(torch.arange(100.0, 111.0)),
        "accelerometer": _make_stream(torch.arange(100.0, 111.0)),
        "lidar": _make_stream(torch.arange(100.0, 111.0)),
    }

    config = WindowingConfig(window_duration_s=5.0, max_samples_per_sensor=700)
    windows = build_windows(streams, config)

    assert len(windows) == 2
    assert all(isinstance(window, StreamWindow) for window in windows)

    assert windows[0].window_start_time == pytest.approx(100.0)
    assert windows[0].window_end_time == pytest.approx(105.0)
    assert windows[1].window_start_time == pytest.approx(105.0)
    assert windows[1].window_end_time == pytest.approx(110.0)

    torch.testing.assert_close(windows[0].streams["gyroscope"].timestamps, torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
    torch.testing.assert_close(windows[1].streams["gyroscope"].timestamps, torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64))


def test_build_windows_uses_half_open_intervals() -> None:
    timestamps = torch.arange(0.0, 11.0)

    streams = {
        "gyroscope": _make_stream(timestamps),
        "accelerometer": _make_stream(timestamps),
    }

    windows = build_windows(streams, WindowingConfig(window_duration_s=5.0, max_samples_per_sensor=700))

    assert len(windows) == 2

    # The measurement at t=5 belongs only to the second [5, 10) window.
    torch.testing.assert_close(windows[0].streams["gyroscope"].timestamps, torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
    torch.testing.assert_close(windows[1].streams["gyroscope"].timestamps, torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64))

    torch.testing.assert_close(windows[0].streams["gyroscope"].values[:, 0], torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
    torch.testing.assert_close(windows[1].streams["gyroscope"].values[:, 0], torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0], dtype=torch.float64))


def test_build_windows_supports_configurable_overlap() -> None:
    timestamps = torch.arange(0.0, 10.5, 0.5)

    streams = {
        "gyroscope": _make_stream(timestamps),
        "accelerometer": _make_stream(timestamps),
    }

    config = WindowingConfig(window_duration_s=5.0, window_stride_s=2.5, max_samples_per_sensor=700)
    windows = build_windows(streams, config)

    assert len(windows) == 3

    assert windows[0].window_start_time == pytest.approx(0.0)
    assert windows[1].window_start_time == pytest.approx(2.5)
    assert windows[2].window_start_time == pytest.approx(5.0)

    assert windows[0].window_end_time == pytest.approx(5.0)
    assert windows[1].window_end_time == pytest.approx(7.5)
    assert windows[2].window_end_time == pytest.approx(10.0)


def test_build_windows_downsamples_each_sensor_independently() -> None:
    # 1400 samples lie in [0, 5), plus one endpoint sample at exactly t=5.
    imu_timestamps = torch.linspace(0.0, 5.0, 1401, dtype=torch.float64)

    # 50 samples lie in [0, 5), plus one endpoint sample at exactly t=5.
    lidar_timestamps = torch.linspace(0.0, 5.0, 51, dtype=torch.float64)

    streams = {
        "gyroscope": _make_stream(imu_timestamps),
        "lidar": _make_stream(lidar_timestamps),
    }

    windows = build_windows(streams, WindowingConfig(window_duration_s=5.0, max_samples_per_sensor=700))

    assert len(windows) == 1

    gyroscope = windows[0].streams["gyroscope"]
    lidar = windows[0].streams["lidar"]

    # 1400 samples require factor ceil(1400 / 700) = 2.
    assert gyroscope.values.shape[0] == 700

    # LiDAR is already below the cap and must remain untouched.
    assert lidar.values.shape[0] == 50

    torch.testing.assert_close(gyroscope.values[:, 0], torch.arange(0.0, 1400.0, 2.0, dtype=torch.float64))
    torch.testing.assert_close(lidar.values[:, 0], torch.arange(0.0, 50.0, dtype=torch.float64))


def test_build_windows_integer_downsampling_factor_guarantees_sample_limit() -> None:
    # Exactly 701 samples lie inside [0, 5), forcing an integer factor of two.
    timestamps = torch.linspace(0.0, 5.0, 702, dtype=torch.float64)

    streams = {
        "gyroscope": _make_stream(timestamps),
        "accelerometer": _make_stream(timestamps),
    }

    windows = build_windows(streams, WindowingConfig(window_duration_s=5.0, max_samples_per_sensor=700))

    assert len(windows) == 1

    gyroscope = windows[0].streams["gyroscope"]

    assert gyroscope.values.shape[0] == 351
    assert gyroscope.values.shape[0] <= 700

    # Current downsampling policy is simple integer-factor decimation without filtering.
    torch.testing.assert_close(gyroscope.values[:, 0], torch.arange(0.0, 701.0, 2.0, dtype=torch.float64))


def test_build_windows_rejects_candidate_when_any_required_sensor_is_missing() -> None:
    dense_timestamps = torch.arange(0.0, 11.0)

    # This sensor has no measurement inside [5, 6).
    sparse_timestamps = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float64)

    streams = {
        "gyroscope": _make_stream(dense_timestamps),
        "lidar": _make_stream(sparse_timestamps),
    }

    windows = build_windows(streams, WindowingConfig(window_duration_s=1.0, window_stride_s=1.0, max_samples_per_sensor=700))

    # Ten candidate windows exist over [0, 10), but [5, 6) must be rejected.
    assert len(windows) == 9
    assert all(window.window_start_time != pytest.approx(5.0) for window in windows)

    expected_start_times = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0]
    assert [window.window_start_time for window in windows] == pytest.approx(expected_start_times)


def test_build_windows_rejects_all_windows_when_required_stream_is_globally_empty() -> None:
    streams = {
        "gyroscope": _make_stream(torch.arange(0.0, 11.0)),
        "lidar": SensorStream(values=torch.empty(0, 3, dtype=torch.float64), timestamps=torch.empty(0, dtype=torch.float64)),
    }

    windows = build_windows(streams, WindowingConfig(window_duration_s=5.0, max_samples_per_sensor=700))

    assert windows == []


def test_build_windows_uses_common_sensor_time_interval() -> None:
    streams = {
        "gyroscope": _make_stream(torch.arange(0.0, 16.0)),
        "accelerometer": _make_stream(torch.arange(2.0, 14.0)),
        "lidar": _make_stream(torch.arange(1.0, 15.0)),
    }

    windows = build_windows(streams, WindowingConfig(window_duration_s=5.0, max_samples_per_sensor=700))

    # Common coverage is [2, 13]. Therefore the two complete windows are [2, 7) and [7, 12).
    assert len(windows) == 2
    assert windows[0].window_start_time == pytest.approx(2.0)
    assert windows[1].window_start_time == pytest.approx(7.0)


def test_build_windows_respects_explicit_start_and_end_bounds() -> None:
    timestamps = torch.arange(0.0, 21.0)

    streams = {
        "gyroscope": _make_stream(timestamps),
        "accelerometer": _make_stream(timestamps),
    }

    config = WindowingConfig(window_duration_s=5.0, max_samples_per_sensor=700)
    windows = build_windows(streams, config, start_time=5.0, end_time=15.0)

    assert len(windows) == 2

    assert windows[0].window_start_time == pytest.approx(5.0)
    assert windows[0].window_end_time == pytest.approx(10.0)
    assert windows[1].window_start_time == pytest.approx(10.0)
    assert windows[1].window_end_time == pytest.approx(15.0)


def test_build_windows_discards_incomplete_trailing_window() -> None:
    timestamps = torch.arange(0.0, 13.0)

    streams = {
        "gyroscope": _make_stream(timestamps),
        "accelerometer": _make_stream(timestamps),
    }

    windows = build_windows(streams, WindowingConfig(window_duration_s=5.0, max_samples_per_sensor=700))

    # [0, 5) and [5, 10) are complete. The remaining interval is too short.
    assert len(windows) == 2


def test_build_windows_preserves_feature_dimensions() -> None:
    timestamps = torch.arange(0.0, 6.0)

    streams = {
        "gyroscope": _make_stream(timestamps, feature_dim=3),
        "accelerometer": _make_stream(timestamps, feature_dim=3),
        "camera_pose": _make_stream(timestamps, feature_dim=16),
    }

    windows = build_windows(streams, WindowingConfig(window_duration_s=5.0, max_samples_per_sensor=700))

    assert len(windows) == 1
    assert windows[0].streams["gyroscope"].values.shape == (5, 3)
    assert windows[0].streams["accelerometer"].values.shape == (5, 3)
    assert windows[0].streams["camera_pose"].values.shape == (5, 16)


def test_build_windows_does_not_modify_original_timestamps() -> None:
    timestamps = torch.arange(100.0, 111.0, dtype=torch.float64)

    streams = {
        "gyroscope": _make_stream(timestamps),
        "accelerometer": _make_stream(timestamps),
    }

    original_timestamps = streams["gyroscope"].timestamps.clone()

    build_windows(streams, WindowingConfig(window_duration_s=5.0, max_samples_per_sensor=700))

    torch.testing.assert_close(streams["gyroscope"].timestamps, original_timestamps)


def test_build_windows_rejects_unsorted_timestamps() -> None:
    """Raw streams must have monotonically ordered timestamps."""

    streams = {
        "gyroscope": _make_stream(
            torch.tensor(
                [0.0, 2.0, 1.0, 3.0, 4.0, 5.0],
                dtype=torch.float64,
            )
        ),
        "accelerometer": _make_stream(torch.arange(0.0, 6.0)),
    }

    # Accept either the older "sorted" wording or the newer, more precise
    # "strictly increasing" wording. The test should enforce semantics rather
    # than pin one human-readable error string.
    with pytest.raises(
        ValueError,
        match=r"(sorted|strictly increasing)",
    ):
        build_windows(
            streams,
            WindowingConfig(
                window_duration_s=5.0,
                max_samples_per_sensor=700,
            ),
        )


def test_build_windows_rejects_nonfinite_timestamps() -> None:
    streams = {
        "gyroscope": _make_stream(torch.tensor([0.0, 1.0, float("nan"), 3.0, 4.0, 5.0], dtype=torch.float64)),
        "accelerometer": _make_stream(torch.arange(0.0, 6.0)),
    }

    with pytest.raises(ValueError, match="must be finite"):
        build_windows(streams, WindowingConfig(window_duration_s=5.0, max_samples_per_sensor=700))


def test_build_windows_rejects_empty_stream_dictionary() -> None:
    with pytest.raises(ValueError, match="At least one sensor stream"):
        build_windows({}, WindowingConfig())


def test_windowing_config_uses_window_duration_as_default_stride() -> None:
    config = WindowingConfig(window_duration_s=5.0)

    assert config.resolved_window_stride_s == pytest.approx(5.0)


def test_windowing_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        WindowingConfig(window_duration_s=0.0)

    with pytest.raises(ValueError):
        WindowingConfig(window_duration_s=5.0, window_stride_s=0.0)

    with pytest.raises(ValueError):
        WindowingConfig(window_duration_s=5.0, max_samples_per_sensor=0)