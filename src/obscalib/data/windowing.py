"""Temporal window construction for synchronized sensor streams."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from obscalib.config import WindowingConfig
from obscalib.data.structures import SensorStream, StreamWindow


def _validate_raw_streams(streams: Mapping[str, SensorStream]) -> None:
    """Validate input streams and require monotonically ordered timestamps."""

    if not streams:
        raise ValueError("At least one sensor stream is required.")

    for stream_name, stream in streams.items():
        stream.validate()

        if stream.timestamps.numel() == 0:
            continue

        if not torch.isfinite(stream.timestamps).all():
            raise ValueError(f"Timestamps for sensor stream {stream_name!r} must be finite.")

        if not torch.all(stream.timestamps[1:] >= stream.timestamps[:-1]):
            raise ValueError(f"Timestamps for sensor stream {stream_name!r} must be sorted in nondecreasing order.")


def _common_stream_interval(streams: Mapping[str, SensorStream]) -> tuple[float, float] | None:
    """
    Return the temporal interval jointly covered by all non-empty streams.

    If any required stream is globally empty, no valid window can be formed.
    """

    if any(stream.timestamps.numel() == 0 for stream in streams.values()):
        return None

    start_time = max(float(stream.timestamps[0].item()) for stream in streams.values())
    end_time = min(float(stream.timestamps[-1].item()) for stream in streams.values())

    if end_time <= start_time:
        return None

    return start_time, end_time


def _window_indices(timestamps: torch.Tensor, window_start_time: float, window_end_time: float) -> tuple[int, int]:
    """
    Find the half-open timestamp interval [window_start_time, window_end_time).

    Half-open windows prevent measurements exactly on a boundary from appearing
    in two adjacent non-overlapping windows.
    """

    start_tensor = timestamps.new_tensor(window_start_time)
    end_tensor = timestamps.new_tensor(window_end_time)

    start_index = int(torch.searchsorted(timestamps, start_tensor, right=False).item())
    end_index = int(torch.searchsorted(timestamps, end_tensor, right=False).item())

    return start_index, end_index


def _downsample_stream(values: torch.Tensor, timestamps: torch.Tensor, max_samples: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Downsample one stream using simple integer-factor decimation.

    No low-pass filtering is currently applied. The integer factor is chosen as

        ceil(num_samples / max_samples),

    which guarantees that the resulting sequence contains no more than
    max_samples samples.
    """

    num_samples = values.shape[0]
    downsampling_factor = max(1, math.ceil(num_samples / max_samples))

    if downsampling_factor == 1:
        return values, timestamps, downsampling_factor

    return values[::downsampling_factor], timestamps[::downsampling_factor], downsampling_factor


def _extract_stream_window(stream: SensorStream, window_start_time: float, window_end_time: float, max_samples: int) -> SensorStream | None:
    """
    Extract, downsample, and time-normalize one sensor stream for one window.

    Returns None when the sensor contributes no samples to the requested
    interval, which causes the complete multi-sensor window to be rejected.
    """

    start_index, end_index = _window_indices(stream.timestamps, window_start_time, window_end_time)

    if start_index >= end_index:
        return None

    values = stream.values[start_index:end_index]
    timestamps = stream.timestamps[start_index:end_index]

    values, timestamps, _ = _downsample_stream(values, timestamps, max_samples)

    # All model-facing times are expressed relative to the current window.
    relative_timestamps = timestamps - window_start_time

    return SensorStream(values=values, timestamps=relative_timestamps)


def build_windows(streams: Mapping[str, SensorStream], config: WindowingConfig | None = None, start_time: float | None = None, end_time: float | None = None) -> list[StreamWindow]:
    """
    Split synchronized raw sensor streams into fixed-duration temporal windows.

    Parameters
    ----------
    streams:
        Named full sensor streams sharing one source time reference. Every entry
        is treated as required. A candidate window is rejected if any sensor has
        zero samples inside it.

    config:
        Window duration, stride, and maximum samples per sensor.

    start_time:
        Optional lower bound in the source time reference. If omitted, the
        beginning of the interval jointly covered by all sensors is used.

    end_time:
        Optional upper bound in the source time reference. If omitted, the end
        of the interval jointly covered by all sensors is used.

    Returns
    -------
    list[StreamWindow]
        Full accepted windows. Sensor timestamps inside each returned window are
        expressed relative to that window's start time.
    """

    if config is None:
        config = WindowingConfig()

    _validate_raw_streams(streams)

    common_interval = _common_stream_interval(streams)

    if common_interval is None:
        return []

    common_start_time, common_end_time = common_interval

    if start_time is not None:
        common_start_time = max(common_start_time, float(start_time))

    if end_time is not None:
        common_end_time = min(common_end_time, float(end_time))

    if common_end_time <= common_start_time:
        return []

    window_duration_s = config.window_duration_s
    window_stride_s = config.resolved_window_stride_s

    available_duration_s = common_end_time - common_start_time

    # Only full windows are produced. Partial trailing windows are discarded.
    if available_duration_s < window_duration_s:
        return []

    num_candidate_windows = int(math.floor((available_duration_s - window_duration_s) / window_stride_s + 1e-12)) + 1

    windows: list[StreamWindow] = []

    for window_index in range(num_candidate_windows):
        window_start_time = common_start_time + window_index * window_stride_s
        window_end_time = window_start_time + window_duration_s

        window_streams: dict[str, SensorStream] = {}
        reject_window = False

        for stream_name, stream in streams.items():
            window_stream = _extract_stream_window(stream, window_start_time, window_end_time, config.max_samples_per_sensor)

            if window_stream is None:
                reject_window = True
                break

            window_streams[stream_name] = window_stream

        if reject_window:
            continue

        windows.append(StreamWindow(window_start_time=window_start_time, window_end_time=window_end_time, streams=window_streams))

    return windows