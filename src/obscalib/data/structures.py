"""Explicit tensor contracts exchanged by calibration pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

import torch

from obscalib.calibration.state import CalibrationState


class MeasurementType(IntEnum):
    """
    Semantic type of one measurement stream.

    Values are contiguous so they can be stored directly in tensors and used
    as indices for learned measurement-type embeddings.
    """

    IMU_GYROSCOPE = 0
    IMU_ACCELEROMETER = 1
    LIDAR_POSE = 2
    CAMERA_POSE = 3
    GPS_POSITION = 4


class GeometryType(str, Enum):
    """Geometry carried by one raw sensor stream."""

    VECTOR = "vector"
    SO3 = "so3"
    SE3 = "se3"


def _validate_unbatched_sequence_fields(
    values: torch.Tensor,
    timestamps: torch.Tensor,
) -> None:
    """Validate an unbatched variable-length sensor sequence."""

    if values.ndim != 2:
        raise ValueError("values must have shape [N, D].")

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


def _validate_batched_sequence_fields(
    values: torch.Tensor,
    timestamps: torch.Tensor,
    sample_mask: torch.Tensor,
) -> None:
    """Validate a padded batch of variable-length sensor sequences."""

    if values.ndim != 3:
        raise ValueError("values must have shape [B, N, D].")

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

    if not (
        values.device
        == timestamps.device
        == sample_mask.device
    ):
        raise ValueError(
            "values, timestamps, and sample_mask must be on the same device."
        )


@dataclass
class SensorStream:
    """
    One unbatched variable-length sensor stream.

    values:
        Sensor measurements with shape [N_s, d_s].

    timestamps:
        Measurement timestamps with shape [N_s]. Full input streams use their
        common source time reference. Streams returned by build_windows() use
        seconds relative to the beginning of their window.
    """

    values: torch.Tensor
    timestamps: torch.Tensor

    def validate(self) -> None:
        """Validate the unbatched stream contract."""

        _validate_unbatched_sequence_fields(self.values, self.timestamps)

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
        [B, N_s_max, d_s]

    timestamps:
        [B, N_s_max]

    sample_mask:
        [B, N_s_max], where True denotes a real sensor sample and False
        denotes padding introduced during minibatch collation.
    """

    values: torch.Tensor
    timestamps: torch.Tensor
    sample_mask: torch.Tensor

    def validate(self) -> None:
        """Validate the padded sensor-stream contract."""

        _validate_batched_sequence_fields(
            self.values,
            self.timestamps,
            self.sample_mask,
        )


@dataclass(frozen=True)
class SensorMetadata:
    """
    Stable identity and representation metadata for one sensor stream.

    measurement_type serves both as the semantic stream type used to select a
    type-specific encoder and, through its integer value, as the type index
    used by learned metadata embeddings.
    """

    sensor_id: int
    measurement_type: MeasurementType
    geometry_type: GeometryType

    @property
    def type_index(self) -> int:
        """Return the contiguous integer code used for tensor embeddings."""

        return int(self.measurement_type)


@dataclass
class CalibrationTarget:
    """
    Unbatched supervision associated with the next calibration state.

    All times are measured relative to the beginning of the current window.

    change_time is meaningful only when change_label is one. Its loss must be
    masked out for no-change samples.
    """

    next_transform: torch.Tensor | None = None
    next_time_offset: torch.Tensor | None = None
    change_label: torch.Tensor | None = None
    change_time: torch.Tensor | None = None


@dataclass
class CalibrationTargetBatch:
    """Batched version of CalibrationTarget produced during collation."""

    next_transform: torch.Tensor | None = None
    next_time_offset: torch.Tensor | None = None
    change_label: torch.Tensor | None = None
    change_time: torch.Tensor | None = None


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
    Geometry-mapped stream retaining its native canonical feature dimension.

    Examples:

        SE(3) -> [phi, rho] : D = 6
        SO(3) -> phi        : D = 3
        vector observation  : D = original vector dimension

    Different sensor streams are deliberately allowed to have different D.
    """

    values: torch.Tensor
    timestamps: torch.Tensor
    sample_mask: torch.Tensor

    def validate(self) -> None:
        """Validate the geometry-mapped padded sequence."""

        _validate_batched_sequence_fields(
            self.values,
            self.timestamps,
            self.sample_mask,
        )


@dataclass
class MeasurementSequenceBatch:
    """
    Merged sequence after type-specific measurement encoding.

    All sensor streams have already been projected to the common learned
    measurement width d_measurement before entering this structure.

    features:
        [B, N, d_measurement]

    timestamps:
        [B, N]

    sensor_ids:
        [B, N]

    measurement_types:
        Integer MeasurementType values with shape [B, N].

    token_mask:
        [B, N], True for real measurement tokens and False for padding.
    """

    features: torch.Tensor
    timestamps: torch.Tensor
    sensor_ids: torch.Tensor
    measurement_types: torch.Tensor
    token_mask: torch.Tensor

    def validate(self) -> None:
        """Validate the merged common-width measurement sequence."""

        _validate_batched_sequence_fields(
            self.features,
            self.timestamps,
            self.token_mask,
        )

        expected_shape = self.timestamps.shape

        if self.sensor_ids.shape != expected_shape:
            raise ValueError("sensor_ids must have shape [B, N].")

        if self.measurement_types.shape != expected_shape:
            raise ValueError(
                "measurement_types must have shape [B, N]."
            )

        if self.sensor_ids.dtype != torch.long:
            raise TypeError("sensor_ids must have dtype torch.long.")

        if self.measurement_types.dtype != torch.long:
            raise TypeError(
                "measurement_types must have dtype torch.long."
            )

        real_measurement_types = self.measurement_types[self.token_mask]

        if real_measurement_types.numel() > 0:
            min_type = int(real_measurement_types.min())
            max_type = int(real_measurement_types.max())

            if min_type < 0 or max_type >= len(MeasurementType):
                raise ValueError(
                    "measurement_types contains an unknown MeasurementType index."
            )

        if torch.any(
            ~torch.isfinite(
                self.timestamps[self.token_mask]
            )
        ):
            raise ValueError(
                "timestamps must be finite for real measurement tokens."
            )


@dataclass
class TokenBatch:
    """
    Prepared Transformer token sequence.

    x:
        [B, N, d_x]

    token_mask:
        [B, N], True for real tokens and False for padded positions.
    """

    x: torch.Tensor
    token_mask: torch.Tensor
    sensor_ids: torch.Tensor
    measurement_types: torch.Tensor

    def validate(self) -> None:
        """Validate aligned token and metadata sequence dimensions."""

        if self.x.ndim != 3:
            raise ValueError("x must have shape [B, N, d_x].")

        expected_shape = self.x.shape[:2]

        if self.token_mask.shape != expected_shape:
            raise ValueError("token_mask must have shape [B, N].")

        if self.sensor_ids.shape != expected_shape:
            raise ValueError("sensor_ids must have shape [B, N].")

        if self.measurement_types.shape != expected_shape:
            raise ValueError(
                "measurement_types must have shape [B, N]."
            )

        if self.token_mask.dtype != torch.bool:
            raise TypeError("token_mask must have boolean dtype.")

        if self.sensor_ids.dtype != torch.long:
            raise TypeError("sensor_ids must have dtype torch.long.")

        if self.measurement_types.dtype != torch.long:
            raise TypeError(
                "measurement_types must have dtype torch.long."
            )