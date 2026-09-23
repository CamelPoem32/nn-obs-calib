"""Explicit tensor contracts exchanged by calibration pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

import torch

from obscalib.calibration.state import CalibrationState


class MeasurementType(IntEnum):
    """
    Semantic physical measurement type.

    Its integer value is concatenated directly to each Transformer token.
    Different physical realizations of the same modality deliberately share
    the same value.
    """

    IMU_GYROSCOPE = 0
    IMU_ACCELEROMETER = 1
    LIDAR_POSE = 2
    CAMERA_POSE = 3
    GPS_POSITION = 4
    RADAR_DETECTION = 5


class GeometryType(str, Enum):
    """Geometry carried by one raw sensor stream."""

    VECTOR = "vector"
    SO3 = "so3"
    SE3 = "se3"


def _validate_unbatched_sequence_fields(values: torch.Tensor, timestamps: torch.Tensor, interval_start_timestamps: torch.Tensor | None = None) -> None:
    """Validate an unbatched variable-length sensor sequence."""

    if values.ndim < 2:
        raise ValueError("values must have shape [N, ...].")

    if timestamps.ndim != 1:
        raise ValueError("timestamps must have shape [N].")

    if values.shape[0] != timestamps.shape[0]:
        raise ValueError("values and timestamps must share N.")

    if not torch.is_floating_point(values):
        raise TypeError("values must have floating-point dtype.")

    if not torch.is_floating_point(timestamps):
        raise TypeError("timestamps must have floating-point dtype.")

    if values.device != timestamps.device:
        raise ValueError("values and timestamps must be on the same device.")

    if interval_start_timestamps is not None:
        if interval_start_timestamps.ndim != 1 or interval_start_timestamps.shape != timestamps.shape:
            raise ValueError("interval_start_timestamps must have shape [N].")

        if not torch.is_floating_point(interval_start_timestamps):
            raise TypeError("interval_start_timestamps must have floating-point dtype.")

        if interval_start_timestamps.device != timestamps.device:
            raise ValueError("interval_start_timestamps and timestamps must be on the same device.")

        if interval_start_timestamps.dtype != timestamps.dtype:
            raise TypeError("interval_start_timestamps and timestamps must have the same dtype.")

        if torch.any(interval_start_timestamps >= timestamps):
            raise ValueError("Every interval start timestamp must be strictly smaller than its corresponding end timestamp.")


def _validate_batched_sequence_fields(values: torch.Tensor, timestamps: torch.Tensor, sample_mask: torch.Tensor, interval_start_timestamps: torch.Tensor | None = None) -> None:
    """Validate a padded batch of variable-length sensor sequences."""

    if values.ndim < 3:
        raise ValueError("values must have shape [B, N, ...].")

    if timestamps.ndim != 2:
        raise ValueError("timestamps must have shape [B, N].")

    if sample_mask.ndim != 2:
        raise ValueError("sample_mask must have shape [B, N].")

    if values.shape[:2] != timestamps.shape:
        raise ValueError("values and timestamps must share B and N.")

    if timestamps.shape != sample_mask.shape:
        raise ValueError("timestamps and sample_mask must share B and N.")

    if not torch.is_floating_point(values):
        raise TypeError("values must have floating-point dtype.")

    if not torch.is_floating_point(timestamps):
        raise TypeError("timestamps must have floating-point dtype.")

    if sample_mask.dtype != torch.bool:
        raise TypeError("sample_mask must have boolean dtype.")

    if not (values.device == timestamps.device == sample_mask.device):
        raise ValueError("values, timestamps, and sample_mask must be on the same device.")

    if interval_start_timestamps is not None:
        if interval_start_timestamps.shape != timestamps.shape:
            raise ValueError("interval_start_timestamps must have shape [B, N].")

        if not torch.is_floating_point(interval_start_timestamps):
            raise TypeError("interval_start_timestamps must have floating-point dtype.")

        if interval_start_timestamps.device != timestamps.device:
            raise ValueError("interval_start_timestamps and timestamps must be on the same device.")

        if interval_start_timestamps.dtype != timestamps.dtype:
            raise TypeError("interval_start_timestamps and timestamps must have the same dtype.")

        if torch.any(interval_start_timestamps[sample_mask] >= timestamps[sample_mask]):
            raise ValueError("Every valid interval start timestamp must be strictly smaller than its corresponding end timestamp.")


@dataclass
class SensorStream:
    """
    One unbatched variable-length sensor stream.

    values:
        Measurements with shape [N, ...].

    timestamps:
        Measurement timestamps with shape [N]. For point measurements these are measurement times. For relative SO3/SE3 measurements these are interval end times.

    interval_start_timestamps:
        Optional interval start times with shape [N]. This is None for point measurements and populated for relative SO3/SE3 measurements.
    """

    values: torch.Tensor
    timestamps: torch.Tensor
    interval_start_timestamps: torch.Tensor | None = None

    def validate(self) -> None:
        """
        Validate the unbatched stream contract.

        Interval-valued streams keep their start timestamps explicitly instead of
        encoding interval semantics implicitly in the measurement geometry.
        """

        _validate_unbatched_sequence_fields(self.values, self.timestamps, self.interval_start_timestamps)

    @property
    def is_interval_valued(self) -> bool:
        """
        Return whether each measurement represents a finite time interval.

        Point measurements such as IMU samples have only ``timestamps``.
        Relative-pose measurements such as the current LiDAR odometry stream also
        carry ``interval_start_timestamps``.
        """

        return self.interval_start_timestamps is not None


@dataclass
class StreamWindow:
    """
    Temporal slice of all required sensor streams before dataset labeling.

    window_start_time and window_end_time remain in the original common source
    time reference so that GT calibration states and labels can be aligned.

    Timestamps inside streams are shifted to seconds relative to
    window_start_time.
    """

    window_start_time: float
    window_end_time: float
    streams: dict[str, SensorStream]


@dataclass
class SensorStreamBatch:
    """
    One padded raw sensor stream for a minibatch.

    values:
        [B, N_max, ...]

    timestamps:
        [B, N_max]. Point-measurement times or relative-measurement interval end times.

    sample_mask:
        [B, N_max], where True denotes a real sample.

    interval_start_timestamps:
        Optional [B, N_max] interval start times for relative SO3/SE3 measurements.
    """

    values: torch.Tensor
    timestamps: torch.Tensor
    sample_mask: torch.Tensor
    interval_start_timestamps: torch.Tensor | None = None

    def validate(self) -> None:
        """
        Validate the padded sensor-stream contract.

        Padding is ignored when checking interval ordering, so only real samples
        are required to have start timestamps strictly before their end timestamps.
        """

        _validate_batched_sequence_fields(self.values, self.timestamps, self.sample_mask, self.interval_start_timestamps)

    @property
    def is_interval_valued(self) -> bool:
        """
        Return whether the batch carries interval start timestamps.

        Every sample in one physical stream uses the same point-versus-interval
        representation.
        """

        return self.interval_start_timestamps is not None


@dataclass(frozen=True)
class SensorMetadata:
    """
    Processing metadata for one raw sensor stream.

    measurement_type:
        Physical sensor modality. Its integer value is concatenated directly to
        the final Transformer token.

    geometry_type:
        Mathematical representation used by the geometry preprocessing block.

    calibration_key:
        Internal key selecting T_sensor_in_world for this physical sensor.
        This key is used only for preprocessing and is never exposed to the
        Transformer.

        Multiple streams may share one calibration key, for example the
        gyroscope and accelerometer belonging to the same IMU.
    """

    measurement_type: MeasurementType
    geometry_type: GeometryType
    calibration_key: str

    @property
    def requires_interval_timestamps(self) -> bool:
        """
        Return whether the currently supported measurement type is interval-valued.

        The present pipeline defines ``LIDAR_POSE`` as a relative LiDAR pose over
        a scan-to-scan or scan-to-map update interval. Other SE(3) modalities may
        later represent point poses, so interval semantics must not be inferred
        from ``geometry_type`` alone.
        """

        return self.measurement_type == MeasurementType.LIDAR_POSE


@dataclass
class CalibrationTarget:
    """
    Unbatched supervision associated with the next calibration state.

    next_transform:
        Ground-truth next calibration transform with shape [4, 4].

    next_time_offset:
        Ground-truth next calibration time offset with shape [1].

    change_label:
        Binary change-event target with shape [1].

    change_time:
        Change time in seconds relative to the beginning of the current window,
        with shape [1]. The value is ignored by the loss when change_label is
        zero.
    """

    next_transform: torch.Tensor | None = None
    next_time_offset: torch.Tensor | None = None
    change_label: torch.Tensor | None = None
    change_time: torch.Tensor | None = None

    def validate(self) -> None:
        """Validate all target fields that are currently populated."""

        if self.next_transform is not None and self.next_transform.shape != (4, 4):
            raise ValueError("next_transform must have shape [4, 4].")

        if self.next_time_offset is not None and self.next_time_offset.shape != (1,):
            raise ValueError("next_time_offset must have shape [1].")

        if self.change_label is not None and self.change_label.shape != (1,):
            raise ValueError("change_label must have shape [1].")

        if self.change_time is not None and self.change_time.shape != (1,):
            raise ValueError("change_time must have shape [1].")


@dataclass
class CalibrationTargetBatch:
    """
    Batched calibration supervision.

    next_transform:
        [B, 4, 4]

    next_time_offset:
        [B, 1]

    change_label:
        [B, 1]

    change_time:
        [B, 1], measured in seconds relative to each window start.
    """

    next_transform: torch.Tensor | None = None
    next_time_offset: torch.Tensor | None = None
    change_label: torch.Tensor | None = None
    change_time: torch.Tensor | None = None

    def validate(self) -> None:
        """Validate shapes and batch agreement of all populated target fields."""

        batch_sizes: list[int] = []

        if self.next_transform is not None:
            if self.next_transform.ndim != 3 or self.next_transform.shape[-2:] != (4, 4):
                raise ValueError("next_transform must have shape [B, 4, 4].")

            batch_sizes.append(self.next_transform.shape[0])

        if self.next_time_offset is not None:
            if self.next_time_offset.ndim != 2 or self.next_time_offset.shape[1] != 1:
                raise ValueError("next_time_offset must have shape [B, 1].")

            batch_sizes.append(self.next_time_offset.shape[0])

        if self.change_label is not None:
            if self.change_label.ndim != 2 or self.change_label.shape[1] != 1:
                raise ValueError("change_label must have shape [B, 1].")

            batch_sizes.append(self.change_label.shape[0])

        if self.change_time is not None:
            if self.change_time.ndim != 2 or self.change_time.shape[1] != 1:
                raise ValueError("change_time must have shape [B, 1].")

            batch_sizes.append(self.change_time.shape[0])

        if batch_sizes and any(batch_size != batch_sizes[0] for batch_size in batch_sizes[1:]):
            raise ValueError("All populated calibration target fields must share batch size.")


@dataclass
class WindowSample:
    """
    One unbatched temporal training/inference window.

    Sensor streams retain their native dimensions and native sample counts.

    current_calibration contains the state supplied to the model for this
    window. During teacher-forced training this is expected to be the
    ground-truth current calibration rather than the previous prediction.

    CalibrationState currently follows the package's batched convention, so
    states stored in a WindowSample are expected to use batch size one.
    """

    streams: dict[str, SensorStream]
    current_calibration: dict[str, CalibrationState]
    metadata: dict[str, SensorMetadata]
    targets: dict[str, CalibrationTarget] | None = None


@dataclass
class WindowBatch:
    """
    Minibatch produced by collating WindowSample objects.

    Each sensor stream is padded independently. A high-rate IMU therefore does
    not force a low-rate LiDAR stream to use the same padded sequence length.
    """

    streams: dict[str, SensorStreamBatch]
    current_calibration: dict[str, CalibrationState]
    metadata: dict[str, SensorMetadata]
    targets: dict[str, CalibrationTargetBatch] | None = None


@dataclass
class CanonicalSensorStreamBatch:
    """
    One sensor stream after calibration-prior transformation and geometry mapping.

    values:
        Canonical vector-valued measurements with shape [B, N_s, d_s].

        Examples:
            gyro / accelerometer -> [B, N_s, 3]
            SO(3) update         -> [B, N_s, 3]
            SE(3) update         -> [B, N_s, 6]

    timestamps:
        Relative timestamps with shape [B, N_s].

    sample_mask:
        True for real samples and False for collation padding.

    measurement_type:
        Physical measurement type used to select the learned encoder.
    """

    values: torch.Tensor
    timestamps: torch.Tensor
    sample_mask: torch.Tensor
    measurement_type: MeasurementType

    def validate(self) -> None:
        """Validate the geometry-mapped padded sequence."""

        _validate_batched_sequence_fields(
            self.values,
            self.timestamps,
            self.sample_mask,
        )
        if self.values.ndim != 3:
            raise ValueError("Canonical values must have shape [B, N, D].")


@dataclass
class EncodedSensorStreamBatch:
    """
    One sensor stream after its type-specific learned measurement encoder.

    features:
        [B, N_s, d_measurement]

    timestamps:
        [B, N_s], measured in seconds relative to the current window start.

    sample_mask:
        [B, N_s], True for real measurements and False for padding.

    measurement_type:
        Physical sensor type shared by every measurement in this stream.

    Every sensor-specific encoder produces the same configurable
    d_measurement, allowing streams to be concatenated afterward.
    """

    features: torch.Tensor
    timestamps: torch.Tensor
    sample_mask: torch.Tensor
    measurement_type: MeasurementType

    def validate(self) -> None:
        """Validate one encoded sensor stream."""

        _validate_batched_sequence_fields(self.features, self.timestamps, self.sample_mask)


@dataclass
class MeasurementSequenceBatch:
    """
    Concatenated, not necessarily time-sorted Transformer-input sequence.

    x already contains the complete per-measurement feature vector

        [measurement | timestamp | measurement type | optional observability].

    timestamps is retained separately only so the complete x vectors can be
    sorted chronologically after all streams have been concatenated.
    """

    x: torch.Tensor
    timestamps: torch.Tensor
    token_mask: torch.Tensor

    def validate(self) -> None:
        """Validate the concatenated sequence."""

        _validate_batched_sequence_fields(self.x, self.timestamps, self.token_mask)


@dataclass
class TokenBatch:
    """
    Final Transformer token sequence.

    x:
        [B, N, d_x]

        with feature ordering

            [measurement | relative timestamp | measurement type | observability]

        where the observability block is absent when observability is disabled.

    token_mask:
        [B, N], True for real tokens and False for padding.
    """

    x: torch.Tensor
    token_mask: torch.Tensor

    def validate(self) -> None:
        """Validate final Transformer tokens."""

        if self.x.ndim != 3:
            raise ValueError("x must have shape [B, N, d_x].")

        if self.token_mask.shape != self.x.shape[:2]:
            raise ValueError("token_mask must have shape [B, N].")

        if self.token_mask.dtype != torch.bool:
            raise TypeError("token_mask must have boolean dtype.")

        if self.token_mask.device != self.x.device:
            raise ValueError("token_mask and x must be on the same device.")