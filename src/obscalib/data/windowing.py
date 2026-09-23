"""Temporal window construction for synchronized point and interval sensor streams."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from obscalib.config import WindowingConfig
from obscalib.data.structures import GeometryType, SensorMetadata, SensorStream, StreamWindow


_INTERVAL_CHAIN_ATOL_SECONDS = 1e-6


def _validate_raw_streams(streams: Mapping[str, SensorStream], metadata: Mapping[str, SensorMetadata] | None = None) -> None:
    """
    Validate source streams and their point-versus-interval semantics.

    Interval semantics are carried explicitly by ``interval_start_timestamps``.
    They are not inferred from SO(3) or SE(3) geometry alone because an SE(3)
    stream may also represent point poses rather than relative motion.
    """

    if not streams:
        raise ValueError("At least one sensor stream is required.")

    if metadata is not None and set(streams) != set(metadata):
        raise ValueError("streams and metadata must contain identical keys.")

    for stream_name, stream in streams.items():
        stream.validate()

        if stream.timestamps.numel() == 0:
            continue

        if not torch.isfinite(stream.timestamps).all():
            raise ValueError(f"Timestamps for sensor stream {stream_name!r} must be finite.")

        if stream.timestamps.numel() > 1 and torch.any(stream.timestamps[1:] <= stream.timestamps[:-1]):
            raise ValueError(f"Timestamps for sensor stream {stream_name!r} must be strictly increasing.")

        if stream.interval_start_timestamps is not None:
            if not torch.isfinite(stream.interval_start_timestamps).all():
                raise ValueError(f"Interval start timestamps for sensor stream {stream_name!r} must be finite.")

            if stream.interval_start_timestamps.numel() > 1 and torch.any(stream.interval_start_timestamps[1:] <= stream.interval_start_timestamps[:-1]):
                raise ValueError(f"Interval start timestamps for sensor stream {stream_name!r} must be strictly increasing.")

        if metadata is None:
            continue

        stream_metadata = metadata[stream_name]

        if stream.interval_start_timestamps is not None and stream_metadata.geometry_type not in {GeometryType.SO3, GeometryType.SE3}:
            raise ValueError(f"Interval-valued stream {stream_name!r} must use SO3 or SE3 geometry.")

        if stream_metadata.requires_interval_timestamps and stream.interval_start_timestamps is None:
            raise ValueError(f"Measurement stream {stream_name!r} requires interval_start_timestamps.")

        if stream_metadata.requires_interval_timestamps and stream_metadata.geometry_type != GeometryType.SE3:
            raise ValueError(f"Current LIDAR_POSE stream {stream_name!r} must use SE3 geometry.")


def _common_stream_interval(streams: Mapping[str, SensorStream]) -> tuple[float, float] | None:
    """
    Return the temporal interval jointly covered by all non-empty streams.

    Point streams begin at their first sample timestamp. Interval streams begin
    at the start of their first complete measurement interval.
    """

    if any(stream.timestamps.numel() == 0 for stream in streams.values()):
        return None

    start_time = max(float(stream.interval_start_timestamps[0].item()) if stream.interval_start_timestamps is not None else float(stream.timestamps[0].item()) for stream in streams.values())
    end_time = min(float(stream.timestamps[-1].item()) for stream in streams.values())

    if end_time <= start_time:
        return None

    return start_time, end_time


def _window_indices(timestamps: torch.Tensor, window_start_time: float, window_end_time: float) -> tuple[int, int]:
    """
    Find indices for the half-open interval ``[window_start_time, window_end_time)``.

    Half-open point-measurement windows prevent a sample exactly on a boundary
    from appearing in two adjacent non-overlapping windows.
    """

    start_tensor = timestamps.new_tensor(window_start_time)
    end_tensor = timestamps.new_tensor(window_end_time)

    start_index = int(torch.searchsorted(timestamps, start_tensor, right=False).item())
    end_index = int(torch.searchsorted(timestamps, end_tensor, right=False).item())

    return start_index, end_index


def _validate_max_samples(max_samples: int) -> None:
    """
    Validate the per-stream storage/token cap.

    This cap belongs to generic dataset windowing. It is distinct from the
    observability reference trajectory grid, which is selected later.
    """

    if max_samples <= 0:
        raise ValueError("max_samples must be strictly positive.")


def _downsample_point_stream(stream: SensorStream, max_samples: int) -> SensorStream:
    """
    Cap a point-measurement stream using the existing integer-factor decimation rule.

    This operation is a generic window-size cap and is not the observability
    reference-grid reduction. Gyroscope integration and accelerometer selection
    relative to trajectory support timestamps happen later in observability
    linearization.
    """

    _validate_max_samples(max_samples)

    num_samples = stream.values.shape[0]
    downsampling_factor = max(1, math.ceil(num_samples / max_samples))

    if downsampling_factor == 1:
        return stream

    return SensorStream(values=stream.values[::downsampling_factor], timestamps=stream.timestamps[::downsampling_factor], interval_start_timestamps=None)


def _downsample_relative_stream(stream: SensorStream, max_samples: int, geometry_type: GeometryType) -> SensorStream:
    """
    Reduce consecutive relative SO(3)/SE(3) measurements by composition.

    Composition preserves the net relative motion across every retained interval,
    unlike ordinary decimation, which would discard intermediate motion.
    """

    _validate_max_samples(max_samples)

    if stream.interval_start_timestamps is None:
        raise ValueError("Relative measurement downsampling requires interval_start_timestamps.")

    if geometry_type not in {GeometryType.SO3, GeometryType.SE3}:
        raise ValueError("Relative measurement composition is supported only for SO3 and SE3 streams.")

    num_samples = stream.values.shape[0]

    if num_samples <= max_samples:
        return stream

    if num_samples > 1 and not torch.allclose(stream.timestamps[:-1], stream.interval_start_timestamps[1:], rtol=0.0, atol=_INTERVAL_CHAIN_ATOL_SECONDS):
        raise ValueError("Relative SO3/SE3 measurements must form consecutive intervals before composition-based downsampling.")

    chunk_size = math.ceil(num_samples / max_samples)

    composed_values: list[torch.Tensor] = []
    composed_start_timestamps: list[torch.Tensor] = []
    composed_end_timestamps: list[torch.Tensor] = []

    for start_index in range(0, num_samples, chunk_size):
        end_index = min(start_index + chunk_size, num_samples)
        chunk_values = stream.values[start_index:end_index]
        composed_value = chunk_values[0]

        for value in chunk_values[1:]:
            composed_value = composed_value @ value

        composed_values.append(composed_value)
        composed_start_timestamps.append(stream.interval_start_timestamps[start_index])
        composed_end_timestamps.append(stream.timestamps[end_index - 1])

    return SensorStream(values=torch.stack(composed_values), timestamps=torch.stack(composed_end_timestamps), interval_start_timestamps=torch.stack(composed_start_timestamps))


def _downsample_stream(stream: SensorStream, max_samples: int, geometry_type: GeometryType | None) -> SensorStream:
    """
    Apply the generic per-stream sample cap without changing interval semantics.

    Point streams use integer-factor decimation. Interval-valued SO(3)/SE(3)
    streams use composition so their retained measurements still represent the
    complete motion over each retained interval.
    """

    _validate_max_samples(max_samples)

    if stream.values.shape[0] <= max_samples:
        return stream

    if stream.interval_start_timestamps is None:
        return _downsample_point_stream(stream, max_samples)

    if geometry_type is None:
        raise ValueError("metadata is required when downsampling interval-valued sensor measurements.")

    return _downsample_relative_stream(stream, max_samples, geometry_type)


def _extract_stream_window(stream: SensorStream, window_start_time: float, window_end_time: float, max_samples: int, geometry_type: GeometryType | None = None) -> SensorStream | None:
    """
    Extract, cap, and time-normalize one sensor stream.

    Point measurements use the half-open interval ``[window_start_time,
    window_end_time)``. Interval measurements are retained only when both their
    start and end timestamps lie inside the window.
    """

    if stream.interval_start_timestamps is None:
        start_index, end_index = _window_indices(stream.timestamps, window_start_time, window_end_time)

        if start_index >= end_index:
            return None

        window_stream = SensorStream(values=stream.values[start_index:end_index], timestamps=stream.timestamps[start_index:end_index], interval_start_timestamps=None)
        window_stream = _downsample_stream(window_stream, max_samples, geometry_type)

        return SensorStream(values=window_stream.values, timestamps=window_stream.timestamps - window_start_time, interval_start_timestamps=None)

    # An interval ending exactly at the window boundary remains in this window because its start lies before that boundary and it cannot be represented completely in the following window.
    interval_mask = (stream.interval_start_timestamps >= window_start_time) & (stream.timestamps <= window_end_time)
    indices = torch.nonzero(interval_mask, as_tuple=False).squeeze(-1)

    if indices.numel() == 0:
        return None

    window_stream = SensorStream(values=stream.values[indices], timestamps=stream.timestamps[indices], interval_start_timestamps=stream.interval_start_timestamps[indices])
    window_stream = _downsample_stream(window_stream, max_samples, geometry_type)

    return SensorStream(values=window_stream.values, timestamps=window_stream.timestamps - window_start_time, interval_start_timestamps=window_stream.interval_start_timestamps - window_start_time)


def build_windows(streams: Mapping[str, SensorStream], config: WindowingConfig | None = None, start_time: float | None = None, end_time: float | None = None, metadata: Mapping[str, SensorMetadata] | None = None) -> list[StreamWindow]:
    """
    Split synchronized sensor streams into fixed-duration windows.

    Source measurements retain their point-versus-interval semantics. Relative
    SO(3)/SE(3) measurements preserve explicit start and end timestamps and are
    composed, rather than discarded, when the generic sample cap is exceeded.

    This function does not select the observability reference trajectory grid.
    That grid is chosen later from each completed window.
    """

    if config is None:
        config = WindowingConfig()

    _validate_max_samples(config.max_samples_per_sensor)
    _validate_raw_streams(streams, metadata)

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
            geometry_type = metadata[stream_name].geometry_type if metadata is not None else None
            window_stream = _extract_stream_window(stream, window_start_time, window_end_time, config.max_samples_per_sensor, geometry_type)

            if window_stream is None:
                reject_window = True
                break

            window_streams[stream_name] = window_stream

        if reject_window:
            continue

        windows.append(StreamWindow(window_start_time=window_start_time, window_end_time=window_end_time, streams=window_streams))

    return windows