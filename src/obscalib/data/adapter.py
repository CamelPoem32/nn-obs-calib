"""Dataset-independent adapter from preprocessed sensor arrays to obscalib data contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from obscalib.calibration.state import CalibrationState
from obscalib.config import WindowingConfig
from obscalib.data.collation import collate_windows
from obscalib.data.structures import GeometryType, MeasurementType, SensorMetadata, SensorStream, WindowBatch, WindowSample
from obscalib.data.windowing import build_windows


@dataclass(frozen=True)
class PreprocessedSensorStream:
    """
    One dataset-independent preprocessed sensor sequence.

    VECTOR measurements are point observations and do not use interval_start_timestamps_s.

    SO3 and SE3 measurements are relative interval observations:
        interval_start_timestamps_s[i] -> timestamps_s[i].
    """

    values: np.ndarray | torch.Tensor
    timestamps_s: np.ndarray | torch.Tensor
    measurement_type: MeasurementType
    geometry_type: GeometryType
    calibration_key: str
    interval_start_timestamps_s: np.ndarray | torch.Tensor | None = None


@dataclass(frozen=True)
class PreprocessedCalibration:
    """
    One preprocessed sensor calibration before conversion to CalibrationState.

    transform:
        Sensor-to-common-frame calibration transform with shape [4, 4] or [1, 4, 4].

    time_offset_s:
        Additive temporal calibration tau. A scalar, [1], or [1, 1] representation is accepted.

    The common frame is whatever frame the complete experiment consistently uses. For example, if body is used as the common frame, a LiDAR calibration can be T_B_L.
    """

    transform: np.ndarray | torch.Tensor
    time_offset_s: float | np.ndarray | torch.Tensor = 0.0


def adapt_preprocessed_stream(stream: PreprocessedSensorStream, *, dtype: torch.dtype = torch.float32, timestamp_dtype: torch.dtype = torch.float64, device: torch.device | str | None = None) -> tuple[SensorStream, SensorMetadata]:
    """Convert one preprocessed sensor sequence into the canonical obscalib stream and metadata contracts."""

    if not isinstance(stream, PreprocessedSensorStream):
        raise TypeError("stream must be a PreprocessedSensorStream.")

    measurement_type = MeasurementType(stream.measurement_type)
    geometry_type = GeometryType(stream.geometry_type)

    if not isinstance(stream.calibration_key, str) or not stream.calibration_key:
        raise ValueError("calibration_key must be a non-empty string.")

    if not dtype.is_floating_point:
        raise TypeError("dtype must be a floating-point torch dtype.")

    if not timestamp_dtype.is_floating_point:
        raise TypeError("timestamp_dtype must be a floating-point torch dtype.")

    values = _as_float_tensor(stream.values, dtype=dtype, device=device, name="values")
    timestamps = _as_float_tensor(stream.timestamps_s, dtype=timestamp_dtype, device=device, name="timestamps_s")
    interval_start_timestamps = None if stream.interval_start_timestamps_s is None else _as_float_tensor(stream.interval_start_timestamps_s, dtype=timestamp_dtype, device=device, name="interval_start_timestamps_s")

    if timestamps.ndim != 1:
        raise ValueError("timestamps_s must have shape [N].")

    if values.shape[0] != timestamps.shape[0]:
        raise ValueError("values and timestamps_s must contain the same number of measurements.")

    if timestamps.numel() == 0:
        raise ValueError("Preprocessed sensor streams must contain at least one measurement.")

    if not torch.isfinite(timestamps).all() or not torch.isfinite(values).all():
        raise ValueError("Preprocessed values and timestamps must contain only finite values.")

    if timestamps.numel() > 1 and torch.any(timestamps[1:] <= timestamps[:-1]):
        raise ValueError("timestamps_s must be strictly increasing.")

    if geometry_type == GeometryType.VECTOR:
        if interval_start_timestamps is not None:
            raise ValueError("VECTOR measurements must not define interval_start_timestamps_s.")

    elif geometry_type in {GeometryType.SO3, GeometryType.SE3}:
        if interval_start_timestamps is None:
            raise ValueError(f"{geometry_type.value.upper()} measurements require interval_start_timestamps_s.")

        if interval_start_timestamps.shape != timestamps.shape:
            raise ValueError("interval_start_timestamps_s must have shape [N].")

        if not torch.isfinite(interval_start_timestamps).all():
            raise ValueError("interval_start_timestamps_s must contain only finite values.")

        if torch.any(interval_start_timestamps >= timestamps):
            raise ValueError("Every interval start timestamp must be strictly smaller than its corresponding end timestamp.")

    _validate_geometry_shape(values, geometry_type)

    sensor_stream = SensorStream(values=values, timestamps=timestamps, interval_start_timestamps=interval_start_timestamps)
    sensor_stream.validate()

    metadata = SensorMetadata(measurement_type=measurement_type, geometry_type=geometry_type, calibration_key=stream.calibration_key)

    return sensor_stream, metadata


def adapt_preprocessed_streams(streams: Mapping[str, PreprocessedSensorStream], *, dtype: torch.dtype = torch.float32, timestamp_dtype: torch.dtype = torch.float64, device: torch.device | str | None = None) -> tuple[dict[str, SensorStream], dict[str, SensorMetadata]]:
    """Convert named preprocessed sequences into canonical sensor streams and metadata."""

    if not streams:
        raise ValueError("At least one preprocessed sensor stream is required.")

    adapted_streams: dict[str, SensorStream] = {}
    metadata: dict[str, SensorMetadata] = {}

    for stream_key, stream in streams.items():
        if not isinstance(stream_key, str) or not stream_key:
            raise ValueError("Sensor stream keys must be non-empty strings.")

        sensor_stream, sensor_metadata = adapt_preprocessed_stream(stream, dtype=dtype, timestamp_dtype=timestamp_dtype, device=device)
        adapted_streams[stream_key] = sensor_stream
        metadata[stream_key] = sensor_metadata

    return adapted_streams, metadata


def adapt_preprocessed_calibration(calibration: PreprocessedCalibration | CalibrationState, *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None) -> CalibrationState:
    """
    Convert one calibration into the singleton-batch CalibrationState used by WindowSample.

    Existing CalibrationState objects are also accepted, but they must have batch size one because this adapter constructs unbatched WindowSample objects before collation.
    """

    if isinstance(calibration, CalibrationState):
        calibration.validate()

        if calibration.transform.shape[0] != 1:
            raise ValueError("CalibrationState supplied to the preprocessed-data adapter must have batch size 1.")

        transform = calibration.transform.to(dtype=dtype, device=device).clone()
        time_offset = calibration.time_offset.to(dtype=dtype, device=device).clone()

        state = CalibrationState(transform=transform, time_offset=time_offset)
        state.validate()

        return state

    if not isinstance(calibration, PreprocessedCalibration):
        raise TypeError("calibration must be a PreprocessedCalibration or CalibrationState.")

    transform = _as_float_tensor(calibration.transform, dtype=dtype, device=device, name="transform")

    if transform.shape == (4, 4):
        transform = transform.unsqueeze(0)
    elif transform.shape != (1, 4, 4):
        raise ValueError("Preprocessed calibration transform must have shape [4, 4] or [1, 4, 4].")

    time_offset = _normalize_time_offset(calibration.time_offset_s, dtype=dtype, device=device)

    if not torch.isfinite(transform).all():
        raise ValueError("Calibration transform must contain only finite values.")

    if not torch.isfinite(time_offset).all():
        raise ValueError("Calibration time offset must contain only finite values.")

    state = CalibrationState(transform=transform, time_offset=time_offset)
    state.validate()

    return state


def adapt_preprocessed_calibrations(calibrations: Mapping[str, PreprocessedCalibration | CalibrationState], *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None) -> dict[str, CalibrationState]:
    """Convert all preprocessed calibrations into singleton-batch CalibrationState objects."""

    if not calibrations:
        raise ValueError("At least one calibration is required.")

    adapted_calibrations: dict[str, CalibrationState] = {}

    for calibration_key, calibration in calibrations.items():
        if not isinstance(calibration_key, str) or not calibration_key:
            raise ValueError("Calibration keys must be non-empty strings.")

        adapted_calibrations[calibration_key] = adapt_preprocessed_calibration(calibration, dtype=dtype, device=device)

    return adapted_calibrations


def build_window_samples_from_preprocessed(streams: Mapping[str, PreprocessedSensorStream], current_calibration: Mapping[str, PreprocessedCalibration | CalibrationState], *, windowing_config: WindowingConfig | None = None, start_time: float | None = None, end_time: float | None = None, dtype: torch.dtype = torch.float32, timestamp_dtype: torch.dtype = torch.float64, device: torch.device | str | None = None) -> list[WindowSample]:
    """Convert synchronized preprocessed streams into model-ready unbatched temporal windows."""

    adapted_streams, metadata = adapt_preprocessed_streams(streams, dtype=dtype, timestamp_dtype=timestamp_dtype, device=device)
    adapted_calibration = adapt_preprocessed_calibrations(current_calibration, dtype=dtype, device=device)

    _validate_stream_calibration_keys(metadata, adapted_calibration)

    stream_windows = build_windows(adapted_streams, config=windowing_config, start_time=start_time, end_time=end_time, metadata=metadata)

    return [WindowSample(streams=dict(stream_window.streams), current_calibration=_clone_calibration_states(adapted_calibration), metadata=dict(metadata), targets=None) for stream_window in stream_windows]


def build_window_batches_from_preprocessed(streams: Mapping[str, PreprocessedSensorStream], current_calibration: Mapping[str, PreprocessedCalibration | CalibrationState], *, batch_size: int, windowing_config: WindowingConfig | None = None, start_time: float | None = None, end_time: float | None = None, drop_last: bool = False, dtype: torch.dtype = torch.float32, timestamp_dtype: torch.dtype = torch.float64, device: torch.device | str | None = None) -> list[WindowBatch]:
    """Convert preprocessed streams directly into padded WindowBatch objects."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    window_samples = build_window_samples_from_preprocessed(streams, current_calibration, windowing_config=windowing_config, start_time=start_time, end_time=end_time, dtype=dtype, timestamp_dtype=timestamp_dtype, device=device)

    batches: list[WindowBatch] = []

    for start_index in range(0, len(window_samples), batch_size):
        batch_samples = window_samples[start_index:start_index + batch_size]

        if drop_last and len(batch_samples) < batch_size:
            break

        batches.append(collate_windows(batch_samples))

    return batches


def _as_float_tensor(values: np.ndarray | torch.Tensor, *, dtype: torch.dtype, device: torch.device | str | None, name: str) -> torch.Tensor:
    """Convert one NumPy array or tensor to the requested floating-point tensor representation."""

    if not dtype.is_floating_point:
        raise TypeError("dtype must be a floating-point torch dtype.")

    if isinstance(values, torch.Tensor):
        return values.to(dtype=dtype, device=device)

    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be convertible to a numerical array.") from error

    if array.dtype.kind not in {"f", "i", "u"}:
        raise TypeError(f"{name} must contain numerical values.")

    return torch.as_tensor(array, dtype=dtype, device=device)


def _normalize_time_offset(time_offset_s: float | np.ndarray | torch.Tensor, *, dtype: torch.dtype, device: torch.device | str | None) -> torch.Tensor:
    """Normalize one scalar temporal calibration into shape [1, 1]."""

    if isinstance(time_offset_s, torch.Tensor):
        time_offset = time_offset_s.to(dtype=dtype, device=device)
    else:
        time_offset = torch.as_tensor(time_offset_s, dtype=dtype, device=device)

    if time_offset.ndim == 0:
        return time_offset.reshape(1, 1)

    if time_offset.shape == (1,):
        return time_offset.reshape(1, 1)

    if time_offset.shape == (1, 1):
        return time_offset

    raise ValueError("time_offset_s must be a scalar, [1], or [1, 1].")


def _validate_geometry_shape(values: torch.Tensor, geometry_type: GeometryType) -> None:
    """Validate the raw tensor representation associated with one geometry type."""

    if geometry_type == GeometryType.VECTOR:
        if values.ndim != 2 or values.shape[-1] != 3:
            raise ValueError("VECTOR measurements must have shape [N, 3].")

        return

    if geometry_type == GeometryType.SO3:
        if values.ndim != 3 or values.shape[-2:] != (3, 3):
            raise ValueError("SO3 measurements must have shape [N, 3, 3].")

        return

    if geometry_type == GeometryType.SE3:
        if values.ndim != 3 or values.shape[-2:] != (4, 4):
            raise ValueError("SE3 measurements must have shape [N, 4, 4].")

        return

    raise ValueError(f"Unsupported geometry type: {geometry_type!r}.")


def _validate_stream_calibration_keys(metadata: Mapping[str, SensorMetadata], current_calibration: Mapping[str, CalibrationState]) -> None:
    """Require every stream to reference an available calibration state."""

    missing_calibration_keys = {sensor_metadata.calibration_key for sensor_metadata in metadata.values()} - set(current_calibration)

    if missing_calibration_keys:
        raise KeyError(f"Missing calibration states required by sensor streams: {sorted(missing_calibration_keys)}")


def _clone_calibration_states(states: Mapping[str, CalibrationState]) -> dict[str, CalibrationState]:
    """Create independent singleton calibration-state containers for one WindowSample."""

    return {calibration_key: CalibrationState(transform=state.transform.clone(), time_offset=state.time_offset.clone()) for calibration_key, state in states.items()}