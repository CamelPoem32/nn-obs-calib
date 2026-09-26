"""Sampling-rate augmentation for point-valued IMU and relative LiDAR measurements."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from obscalib.augmentations.config import SamplingRateAugmentationConfig

from obscalib.data.structures import GeometryType, MeasurementType, MINIMUM_REQUIRED_SAMPLES_PER_STREAM, SensorMetadata, SensorStreamBatch, WindowBatch

_IMU_MEASUREMENT_TYPES = {
    MeasurementType.IMU_GYROSCOPE,
    MeasurementType.IMU_ACCELEROMETER,
}


class SamplingRateAugmenter:
    """
    Randomly reduce IMU and LiDAR sampling rates.

    IMU measurements are linearly interpolated onto a regular lower-frequency grid. Gyroscope and accelerometer streams sharing one calibration key use the same sampled target frequency.

    LiDAR relative SE3 measurements are reduced by selecting a lower-rate sequence of retained scans and composing all original relative transforms between consecutive retained scans.
    """

    def __init__(self, config: SamplingRateAugmentationConfig) -> None:
        self.config = config

    def __call__(self, window: WindowBatch, generator: torch.Generator | None = None) -> tuple[WindowBatch, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Apply configured sampling-rate augmentation.

        Returns:
            augmented_window:
                Window with potentially resampled sensor streams.

            target_frequency_hz_by_stream:
                Sampled target frequency for every configured stream and batch item. When augmentation is not applied to one batch item, its original estimated frequency is stored.

            applied_by_stream:
                Boolean [B, 1] tensor indicating whether sampling-rate augmentation was actually applied to each stream and batch item.
        """

        _validate_generator(generator)
        _validate_window(window)

        if not self.config.enabled:
            return window, {}, {}

        batch_size = _batch_size(window)
        imu_target_frequency_by_key = self._sample_imu_target_frequencies(window, batch_size, generator)

        augmented_streams: dict[str, SensorStreamBatch] = {}
        target_frequency_hz_by_stream: dict[str, torch.Tensor] = {}
        applied_by_stream: dict[str, torch.Tensor] = {}

        for stream_key, stream in window.streams.items():
            metadata = window.metadata[stream_key]

            if _is_imu_stream(metadata) and self.config.minimum_imu_frequency_hz is not None:
                augmented_stream, target_frequency_hz, applied = _augment_imu_stream(stream, imu_target_frequency_by_key.get(metadata.calibration_key), self.config.minimum_imu_frequency_hz)

                augmented_streams[stream_key] = augmented_stream
                target_frequency_hz_by_stream[stream_key] = target_frequency_hz
                applied_by_stream[stream_key] = applied
                continue

            if _is_lidar_stream(metadata) and self.config.minimum_lidar_frequency_hz is not None:
                augmented_stream, target_frequency_hz, applied = _augment_lidar_stream(stream, minimum_frequency_hz=self.config.minimum_lidar_frequency_hz, probability=self.config.lidar_probability, generator=generator)

                augmented_streams[stream_key] = augmented_stream
                target_frequency_hz_by_stream[stream_key] = target_frequency_hz
                applied_by_stream[stream_key] = applied
                continue

            augmented_streams[stream_key] = stream

        augmented_window = WindowBatch(
            streams=augmented_streams,
            current_calibration=dict(window.current_calibration),
            metadata=dict(window.metadata),
            targets=None if window.targets is None else dict(window.targets),
        )

        return augmented_window, target_frequency_hz_by_stream, applied_by_stream

    def _sample_imu_target_frequencies(self, window: WindowBatch, batch_size: int, generator: torch.Generator | None) -> dict[str, list[float | None]]:
        """Sample one shared IMU target frequency per physical calibration key and batch item."""

        if self.config.minimum_imu_frequency_hz is None:
            return {}

        imu_streams_by_calibration_key: dict[str, list[str]] = {}

        for stream_key, metadata in window.metadata.items():
            if not _is_imu_stream(metadata):
                continue

            imu_streams_by_calibration_key.setdefault(metadata.calibration_key, []).append(stream_key)

        target_frequency_by_key: dict[str, list[float | None]] = {}

        for calibration_key, stream_keys in imu_streams_by_calibration_key.items():
            targets: list[float | None] = [None] * batch_size

            for batch_index in range(batch_size):
                source_frequencies: list[float] = []

                for stream_key in stream_keys:
                    stream = window.streams[stream_key]
                    valid_timestamps = stream.timestamps[batch_index, stream.sample_mask[batch_index]]
                    source_frequency = _estimate_point_frequency(valid_timestamps)

                    if source_frequency is None:
                        source_frequencies = []
                        break

                    source_frequencies.append(source_frequency)

                if not source_frequencies:
                    continue

                # A shared target must not exceed the slowest stream belonging to
                # this physical IMU.
                maximum_target_frequency_hz = min(source_frequencies)

                if self.config.minimum_imu_frequency_hz >= maximum_target_frequency_hz:
                    continue

                if not _sample_probability(self.config.imu_probability, generator):
                    continue

                targets[batch_index] = _sample_uniform(self.config.minimum_imu_frequency_hz, maximum_target_frequency_hz, generator)

            target_frequency_by_key[calibration_key] = targets

        return target_frequency_by_key


def _augment_imu_stream(stream: SensorStreamBatch, target_frequencies: list[float | None] | None, minimum_frequency_hz: float) -> tuple[SensorStreamBatch, torch.Tensor, torch.Tensor]:
    """Linearly resample one IMU stream according to previously sampled per-item target frequencies."""

    stream.validate()

    if stream.interval_start_timestamps is not None:
        raise ValueError("IMU sampling-rate augmentation expects point measurements without interval_start_timestamps.")

    batch_size = stream.values.shape[0]

    if target_frequencies is None:
        target_frequencies = [None] * batch_size

    if len(target_frequencies) != batch_size:
        raise ValueError("IMU target-frequency records must contain one entry per batch item.")

    samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
    target_frequency_record = torch.full((batch_size, 1), torch.nan, dtype=stream.timestamps.dtype, device=stream.timestamps.device)
    applied_record = torch.zeros((batch_size, 1), dtype=torch.bool, device=stream.timestamps.device)

    for batch_index in range(batch_size):
        sample_mask = stream.sample_mask[batch_index]
        values = stream.values[batch_index, sample_mask]
        timestamps = stream.timestamps[batch_index, sample_mask]

        source_frequency = _estimate_point_frequency(timestamps)

        if source_frequency is not None:
            target_frequency_record[batch_index, 0] = source_frequency

        target_frequency = target_frequencies[batch_index]

        if target_frequency is None or source_frequency is None or target_frequency >= source_frequency:
            samples.append((values, timestamps, None))
            continue

        if target_frequency < minimum_frequency_hz:
            raise ValueError("Sampled IMU target frequency is below the configured minimum.")

        resampled_values, resampled_timestamps = _linear_resample(values, timestamps, target_frequency)

        samples.append((resampled_values, resampled_timestamps, None))
        target_frequency_record[batch_index, 0] = target_frequency
        applied_record[batch_index, 0] = True

    return _pack_samples(samples), target_frequency_record, applied_record


def _augment_lidar_stream(stream: SensorStreamBatch, *, minimum_frequency_hz: float, probability: float, generator: torch.Generator | None) -> tuple[SensorStreamBatch, torch.Tensor, torch.Tensor]:
    """Reduce one relative SE3 LiDAR stream while preserving at least two valid relative measurements per batch item."""

    stream.validate()

    if stream.interval_start_timestamps is None:
        raise ValueError("LiDAR sampling-rate augmentation requires interval_start_timestamps.")

    if stream.values.ndim != 4 or stream.values.shape[-2:] != (4, 4):
        raise ValueError("LiDAR sampling-rate augmentation expects SE3 values with shape [B, N, 4, 4].")

    batch_size = stream.values.shape[0]

    samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
    target_frequency_record = torch.full((batch_size, 1), torch.nan, dtype=stream.timestamps.dtype, device=stream.timestamps.device)
    applied_record = torch.zeros((batch_size, 1), dtype=torch.bool, device=stream.timestamps.device)

    for batch_index in range(batch_size):
        sample_mask = stream.sample_mask[batch_index]

        values = stream.values[batch_index, sample_mask]
        end_timestamps = stream.timestamps[batch_index, sample_mask]
        start_timestamps = stream.interval_start_timestamps[batch_index, sample_mask]

        if values.shape[0] < MINIMUM_REQUIRED_SAMPLES_PER_STREAM:
            raise ValueError(f"LiDAR batch item {batch_index} contains only {values.shape[0]} valid relative measurements before sampling-rate augmentation.")

        source_frequency = _estimate_interval_frequency(start_timestamps, end_timestamps)

        if source_frequency is not None:
            target_frequency_record[batch_index, 0] = source_frequency

        if source_frequency is None or minimum_frequency_hz >= source_frequency or not _sample_probability(probability, generator):
            samples.append((values, end_timestamps, start_timestamps))
            continue

        target_frequency = _sample_uniform(minimum_frequency_hz, source_frequency, generator)

        reduced_values, reduced_end_timestamps, reduced_start_timestamps = _reduce_relative_se3_scan_rate(values, start_timestamps, end_timestamps, target_frequency)

        # Reject this particular augmentation realization if it would leave fewer than two relative measurements. Keep the original stream instead.
        if reduced_values.shape[0] < MINIMUM_REQUIRED_SAMPLES_PER_STREAM:
            samples.append((values, end_timestamps, start_timestamps))
            continue

        augmentation_applied = reduced_values.shape[0] < values.shape[0]

        # If the sampled target frequency happened to retain the complete original stream, record this item as not augmented and keep the original frequency record.
        if not augmentation_applied:
            samples.append((values, end_timestamps, start_timestamps))
            continue

        samples.append((reduced_values, reduced_end_timestamps, reduced_start_timestamps))
        target_frequency_record[batch_index, 0] = target_frequency
        applied_record[batch_index, 0] = True

    return _pack_samples(samples), target_frequency_record, applied_record


def _linear_resample(values: torch.Tensor, timestamps: torch.Tensor, target_frequency_hz: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Linearly interpolate point measurements onto a regular target-frequency grid."""

    if values.shape[0] != timestamps.shape[0]:
        raise ValueError("values and timestamps must contain the same number of samples.")

    if timestamps.numel() < 2:
        return values, timestamps

    target_timestamps = _regular_timestamp_grid(timestamps[0], timestamps[-1], target_frequency_hz)

    right_indices = torch.searchsorted(timestamps, target_timestamps, right=False).clamp(min=1, max=timestamps.shape[0] - 1)
    left_indices = right_indices - 1

    left_timestamps = timestamps[left_indices]
    right_timestamps = timestamps[right_indices]

    interpolation_weight = ((target_timestamps - left_timestamps) / (right_timestamps - left_timestamps)).to(dtype=values.dtype)

    while interpolation_weight.ndim < values.ndim:
        interpolation_weight = interpolation_weight.unsqueeze(-1)

    left_values = values[left_indices]
    right_values = values[right_indices]

    resampled_values = left_values + interpolation_weight * (right_values - left_values)

    return resampled_values, target_timestamps


def _reduce_relative_se3_scan_rate(values: torch.Tensor, start_timestamps: torch.Tensor, end_timestamps: torch.Tensor, target_frequency_hz: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce LiDAR scan rate while preserving total relative motion and at least two relative measurements."""

    if values.shape[0] != start_timestamps.shape[0] or values.shape[0] != end_timestamps.shape[0]:
        raise ValueError("LiDAR values and interval timestamps must share N.")

    if values.shape[0] < MINIMUM_REQUIRED_SAMPLES_PER_STREAM:
        return values, end_timestamps, start_timestamps

    if not torch.allclose(end_timestamps[:-1], start_timestamps[1:], rtol=0.0, atol=_timestamp_tolerance(end_timestamps)):
        raise ValueError("Relative LiDAR measurements must form consecutive intervals before scan-rate reduction.")

    scan_timestamps = torch.cat((start_timestamps[:1], end_timestamps), dim=0)
    desired_scan_timestamps = _regular_timestamp_grid(scan_timestamps[0], scan_timestamps[-1], target_frequency_hz)
    retained_scan_indices = _nearest_scan_indices(scan_timestamps, desired_scan_timestamps)

    if retained_scan_indices[0].item() != 0:
        retained_scan_indices = torch.cat((retained_scan_indices.new_tensor([0]), retained_scan_indices))

    final_scan_index = scan_timestamps.shape[0] - 1

    if retained_scan_indices[-1].item() != final_scan_index:
        retained_scan_indices = torch.cat((retained_scan_indices, retained_scan_indices.new_tensor([final_scan_index])))

    retained_scan_indices = torch.unique_consecutive(retained_scan_indices)

    # Two relative measurements require at least three retained scan timestamps.
    if retained_scan_indices.numel() < MINIMUM_REQUIRED_SAMPLES_PER_STREAM + 1:
        return values, end_timestamps, start_timestamps

    composed_values: list[torch.Tensor] = []
    reduced_start_timestamps: list[torch.Tensor] = []
    reduced_end_timestamps: list[torch.Tensor] = []

    for retained_index in range(retained_scan_indices.shape[0] - 1):
        first_scan_index = int(retained_scan_indices[retained_index].item())
        last_scan_index = int(retained_scan_indices[retained_index + 1].item())

        if last_scan_index <= first_scan_index:
            continue

        composed = values[first_scan_index]

        for measurement_index in range(first_scan_index + 1, last_scan_index):
            composed = composed @ values[measurement_index]

        composed_values.append(composed)
        reduced_start_timestamps.append(scan_timestamps[first_scan_index])
        reduced_end_timestamps.append(scan_timestamps[last_scan_index])

    if len(composed_values) < MINIMUM_REQUIRED_SAMPLES_PER_STREAM:
        return values, end_timestamps, start_timestamps

    return torch.stack(composed_values), torch.stack(reduced_end_timestamps), torch.stack(reduced_start_timestamps)


def _nearest_scan_indices(scan_timestamps: torch.Tensor, desired_timestamps: torch.Tensor) -> torch.Tensor:
    """Map desired scan times to monotonically ordered nearest real scan indices."""

    right_indices = torch.searchsorted(scan_timestamps, desired_timestamps, right=False).clamp(max=scan_timestamps.shape[0] - 1)
    left_indices = (right_indices - 1).clamp(min=0)

    left_distance = torch.abs(desired_timestamps - scan_timestamps[left_indices])
    right_distance = torch.abs(scan_timestamps[right_indices] - desired_timestamps)

    nearest_indices = torch.where(right_distance < left_distance, right_indices, left_indices)

    return torch.unique_consecutive(nearest_indices)


def _regular_timestamp_grid(start_timestamp: torch.Tensor, end_timestamp: torch.Tensor, frequency_hz: float) -> torch.Tensor:
    """Construct a regular timestamp grid while preserving both support endpoints."""

    if frequency_hz <= 0.0 or not math.isfinite(frequency_hz):
        raise ValueError("frequency_hz must be finite and positive.")

    duration_s = float((end_timestamp - start_timestamp).item())

    if duration_s <= 0.0:
        raise ValueError("Timestamp support must have positive duration.")

    number_regular_intervals = int(math.floor(duration_s * frequency_hz + 1e-12))
    regular_offsets = torch.arange(number_regular_intervals + 1, dtype=start_timestamp.dtype, device=start_timestamp.device) / frequency_hz
    timestamps = start_timestamp + regular_offsets

    tolerance = _timestamp_tolerance(timestamps)

    if float((end_timestamp - timestamps[-1]).item()) > tolerance:
        timestamps = torch.cat((timestamps, end_timestamp.reshape(1)))
    else:
        timestamps[-1] = end_timestamp

    return timestamps


def _estimate_point_frequency(timestamps: torch.Tensor) -> float | None:
    """Estimate point-measurement frequency from the median timestamp spacing."""

    if timestamps.numel() < 2:
        return None

    deltas = timestamps[1:] - timestamps[:-1]
    deltas = deltas[deltas > 0.0]

    if deltas.numel() == 0:
        return None

    return float((1.0 / torch.median(deltas)).item())


def _estimate_interval_frequency(start_timestamps: torch.Tensor, end_timestamps: torch.Tensor) -> float | None:
    """Estimate scan frequency from median relative-measurement interval duration."""

    if start_timestamps.numel() == 0:
        return None

    durations = end_timestamps - start_timestamps
    durations = durations[durations > 0.0]

    if durations.numel() == 0:
        return None

    return float((1.0 / torch.median(durations)).item())


def _pack_samples(samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]) -> SensorStreamBatch:
    """Pad independently processed batch items back into one SensorStreamBatch."""

    if not samples:
        raise ValueError("Cannot pack an empty batch.")

    batch_size = len(samples)
    sample_shape = samples[0][0].shape[1:]
    values_dtype = samples[0][0].dtype
    timestamp_dtype = samples[0][1].dtype
    device = samples[0][0].device
    has_interval_starts = samples[0][2] is not None

    for values, timestamps, interval_start_timestamps in samples:
        if timestamps.numel() < MINIMUM_REQUIRED_SAMPLES_PER_STREAM:
            raise ValueError(f"Sampling-rate augmentation produced a stream with only {timestamps.numel()} valid measurements; at least {MINIMUM_REQUIRED_SAMPLES_PER_STREAM} are required.")

        if values.shape[1:] != sample_shape:
            raise ValueError("All batch items must share the same measurement shape.")

        if values.dtype != values_dtype or timestamps.dtype != timestamp_dtype:
            raise TypeError("All batch items must share values and timestamp dtypes.")

        if values.device != device or timestamps.device != device:
            raise ValueError("All batch items must be on the same device.")

        if (interval_start_timestamps is not None) != has_interval_starts:
            raise ValueError("Interval start timestamps must be present for either all or none of the batch items.")

    max_samples = max(values.shape[0] for values, _, _ in samples)

    values_batch = torch.zeros((batch_size, max_samples, *sample_shape), dtype=values_dtype, device=device)
    timestamps_batch = torch.zeros((batch_size, max_samples), dtype=timestamp_dtype, device=device)
    sample_mask = torch.zeros((batch_size, max_samples), dtype=torch.bool, device=device)
    interval_start_batch = torch.zeros((batch_size, max_samples), dtype=timestamp_dtype, device=device) if has_interval_starts else None

    for batch_index, (values, timestamps, interval_start_timestamps) in enumerate(samples):
        number_samples = values.shape[0]

        values_batch[batch_index, :number_samples] = values
        timestamps_batch[batch_index, :number_samples] = timestamps
        sample_mask[batch_index, :number_samples] = True

        if interval_start_batch is not None:
            interval_start_batch[batch_index, :number_samples] = interval_start_timestamps

    return SensorStreamBatch(values=values_batch, timestamps=timestamps_batch, sample_mask=sample_mask, interval_start_timestamps=interval_start_batch)


def _is_imu_stream(metadata: SensorMetadata) -> bool:
    return metadata.geometry_type == GeometryType.VECTOR and metadata.measurement_type in _IMU_MEASUREMENT_TYPES


def _is_lidar_stream(metadata: SensorMetadata) -> bool:
    return metadata.geometry_type == GeometryType.SE3 and metadata.measurement_type == MeasurementType.LIDAR_POSE


def _sample_probability(probability: float, generator: torch.Generator | None) -> bool:
    if probability <= 0.0:
        return False

    if probability >= 1.0:
        return True

    return _sample_uniform(0.0, 1.0, generator) < probability


def _sample_uniform(minimum: float, maximum: float, generator: torch.Generator | None) -> float:
    if maximum < minimum:
        raise ValueError("Uniform sampling maximum must be at least the minimum.")

    if maximum == minimum:
        return minimum

    device = generator.device if generator is not None else torch.device("cpu")
    unit_sample = float(torch.rand((), dtype=torch.float64, device=device, generator=generator).item())

    return minimum + unit_sample * (maximum - minimum)


def _timestamp_tolerance(timestamps: torch.Tensor) -> float:
    if timestamps.dtype == torch.float64:
        return 1e-9

    return 1e-5


def _batch_size(window: WindowBatch) -> int:
    first_stream = next(iter(window.streams.values()))
    return first_stream.values.shape[0]


def _validate_window(window: WindowBatch) -> None:
    if not window.streams:
        raise ValueError("Sampling-rate augmentation requires at least one sensor stream.")

    if set(window.streams) != set(window.metadata):
        raise ValueError("Window streams and metadata must have identical keys.")

    batch_size = _batch_size(window)

    for stream_key, stream in window.streams.items():
        stream.validate()

        if stream.values.shape[0] != batch_size:
            raise ValueError(f"Stream {stream_key!r} does not share the common batch size.")

        valid_samples_per_batch_item = stream.sample_mask.sum(dim=1)
        invalid_batch_indices = torch.nonzero(valid_samples_per_batch_item < MINIMUM_REQUIRED_SAMPLES_PER_STREAM, as_tuple=False).squeeze(-1)

        if invalid_batch_indices.numel() > 0:
            invalid_counts = valid_samples_per_batch_item[invalid_batch_indices]
            raise ValueError(f"Every required sensor stream must contain at least {MINIMUM_REQUIRED_SAMPLES_PER_STREAM} valid measurements before sampling-rate augmentation. Stream {stream_key!r} has invalid batch indices {invalid_batch_indices.tolist()} with counts {invalid_counts.tolist()}.")


def _validate_generator(generator: torch.Generator | None) -> None:
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or None.")