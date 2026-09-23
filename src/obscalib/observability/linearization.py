"""One-window sensor-factor linearization and deterministic global Jacobian assembly."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from obscalib.calibration.state import CalibrationState
from obscalib.data.structures import MeasurementType, SensorMetadata, SensorStream
from obscalib.geometry.processing import interpolate_se3_trajectory_with_twist
from obscalib.observability.factors.accelerometer import linearize_simple_accelerometer_factor
from obscalib.observability.factors.gyroscope import GYROSCOPE_BIAS_PARAMETER_NAMES, gyroscope_bias_nuisance_key, gyroscope_interval_is_supported, linearize_gyroscope_factor
from obscalib.observability.factors.lidar import linearize_lidar_factor
from obscalib.observability.layout import CalibrationParameterLayout, NuisanceParameterLayout
from obscalib.observability.timebase import ReferenceTimebase


TRAJECTORY_POSE_DIM = 6
SUPPORT_TIME_TOLERANCE_S = 1e-9


@dataclass(frozen=True)
class AccelerometerSampleSupport:
    """Describe one selected accelerometer measurement and its corrected time.

    Only a low-rate subset of the raw accelerometer signal is used for the
    conservative first observability baseline. The selected samples themselves
    remain raw measurements and are not averaged.
    """

    sample_index: int
    sensor_time: float
    true_time: float


@dataclass(frozen=True)
class LidarIntervalSupport:
    """Describe one LiDAR relative-pose measurement and its two corrected times.

    sensor_start_time and sensor_end_time are expressed on the LiDAR clock.
    true_start_time and true_end_time apply the current additive LiDAR offset.
    """

    measurement_index: int
    sensor_start_time: float
    sensor_end_time: float
    true_start_time: float
    true_end_time: float


def _validate_scientific_tensor(tensor: torch.Tensor, *, name: str) -> None:
    """Validate one tensor at the scientific observability boundary.

    Window linearization uses detached CPU float64 tensors independently of the
    dtype and device used by the learned model.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be stored on CPU.")
    if tensor.requires_grad:
        raise ValueError(f"{name} must be detached from autograd.")
    if tensor.dtype != torch.float64:
        raise ValueError(f"{name} must use torch.float64.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values.")


def _to_scientific_tensor(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    """Move one pipeline tensor once into the scientific CPU float64 domain.

    The returned tensor is detached from autograd and validated for finite
    values.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")

    result = tensor.detach().to(device="cpu", dtype=torch.float64)

    if not torch.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values.")

    return result


@dataclass(frozen=True)
class TrajectoryParameterLayout:
    """Define the window-dependent ordering of trajectory nuisance variables.

    Every timestamp contributes one six-dimensional left SE(3) perturbation in
    rotation-first ordering.
    """

    timestamps: torch.Tensor

    def validate(self) -> None:
        """Validate trajectory timestamps and their deterministic ordering.

        At least two strictly increasing finite timestamps are required.
        """

        _validate_scientific_tensor(self.timestamps, name="trajectory timestamps")

        if self.timestamps.ndim != 1:
            raise ValueError("Trajectory timestamps must have shape [N].")
        if self.timestamps.numel() < 2:
            raise ValueError("At least two trajectory timestamps are required.")
        if torch.any(self.timestamps[1:] <= self.timestamps[:-1]):
            raise ValueError("Trajectory timestamps must be strictly increasing.")

    @property
    def pose_count(self) -> int:
        """Return the number of trajectory pose blocks.

        Each block contributes six nuisance dimensions.
        """

        return int(self.timestamps.numel())

    @property
    def total_dimension(self) -> int:
        """Return the total trajectory nuisance dimension.

        The dimension is six times the number of trajectory support poses.
        """

        return self.pose_count * TRAJECTORY_POSE_DIM

    def pose_slice(self, pose_index: int) -> slice:
        """Return the global trajectory columns for one support pose.

        pose_index follows chronological trajectory-layout ordering.
        """

        if pose_index < 0 or pose_index >= self.pose_count:
            raise IndexError(f"pose_index {pose_index} is outside [0, {self.pose_count}).")

        start = pose_index * TRAJECTORY_POSE_DIM

        return slice(start, start + TRAJECTORY_POSE_DIM)


@dataclass(frozen=True)
class SensorFactorLinearization:
    """Store one individual residual and its global Jacobian blocks.

    trajectory_jacobian, nuisance_jacobian, and calibration_jacobian already
    use the complete window-level column layouts, making later row assembly a
    mechanical concatenation step.

    residual_covariance remains optional during migration. Fisher construction
    must reject missing covariance rather than silently assume identity noise.
    """

    stream_key: str
    calibration_key: str
    residual: torch.Tensor
    trajectory_jacobian: torch.Tensor
    calibration_jacobian: torch.Tensor
    residual_covariance: torch.Tensor | None = None
    nuisance_jacobian: torch.Tensor | None = None
    measurement_type: MeasurementType | None = None
    factor_index: int | None = None

    def validate(self) -> None:
        """Validate residual and Jacobian dimensions for one factor.

        Scientific tensors must remain on CPU, detached from autograd, and use
        float64.
        """

        if not self.stream_key or not self.calibration_key:
            raise ValueError("stream_key and calibration_key must be non-empty.")

        _validate_scientific_tensor(self.residual, name="residual")
        _validate_scientific_tensor(self.trajectory_jacobian, name="trajectory_jacobian")
        _validate_scientific_tensor(self.calibration_jacobian, name="calibration_jacobian")

        if self.residual.ndim != 1:
            raise ValueError("residual must have shape [R].")

        residual_dimension = self.residual.shape[0]

        if self.trajectory_jacobian.ndim != 2 or self.trajectory_jacobian.shape[0] != residual_dimension:
            raise ValueError("trajectory_jacobian must have shape [R, D_trajectory].")
        if self.calibration_jacobian.ndim != 2 or self.calibration_jacobian.shape[0] != residual_dimension:
            raise ValueError("calibration_jacobian must have shape [R, D_calibration].")

        if self.nuisance_jacobian is not None:
            _validate_scientific_tensor(self.nuisance_jacobian, name="nuisance_jacobian")
            if self.nuisance_jacobian.ndim != 2 or self.nuisance_jacobian.shape[0] != residual_dimension:
                raise ValueError("nuisance_jacobian must have shape [R, D_nuisance].")

        if self.residual_covariance is not None:
            _validate_scientific_tensor(self.residual_covariance, name="residual_covariance")
            if self.residual_covariance.shape != (residual_dimension, residual_dimension):
                raise ValueError("residual_covariance must have shape [R, R].")

        if self.factor_index is not None and self.factor_index < 0:
            raise ValueError("factor_index must be nonnegative when provided.")


@dataclass(frozen=True)
class WindowLinearization:
    """Store every sensor factor and variable layout for one temporal window.

    The trajectory and explicit nuisance blocks are later marginalized, while
    calibration columns remain in the final Fisher information matrix.
    """

    factors: tuple[SensorFactorLinearization, ...]
    calibration_layout: CalibrationParameterLayout
    reference_timebase: ReferenceTimebase
    trajectory_layout: TrajectoryParameterLayout | None = None
    nuisance_layout: NuisanceParameterLayout = field(default_factory=NuisanceParameterLayout.empty)

    def validate(self) -> None:
        """Validate all factor widths against the shared window layouts.

        A nonempty nuisance layout requires every factor to carry an explicit
        global nuisance Jacobian, even when that factor's block is all zeros.
        """

        if not self.factors:
            raise ValueError("A window linearization requires at least one factor.")

        self.reference_timebase.validate()
        if self.trajectory_layout is not None:
            self.trajectory_layout.validate()
            trajectory_dimension = self.trajectory_layout.total_dimension
        else:
            trajectory_dimension = None
        nuisance_dimension = self.nuisance_layout.total_dimension

        for factor in self.factors:
            factor.validate()

            if factor.calibration_key not in self.calibration_layout.calibration_keys:
                raise ValueError(f"Factor {factor.stream_key!r} uses calibration key {factor.calibration_key!r} outside the calibration layout.")
            if factor.calibration_jacobian.shape[1] != self.calibration_layout.total_dimension:
                raise ValueError("Factor calibration Jacobian width must match the calibration layout.")
            if trajectory_dimension is not None and factor.trajectory_jacobian.shape[1] != trajectory_dimension:
                raise ValueError("Factor trajectory Jacobian width must match the trajectory layout.")

            if nuisance_dimension == 0:
                if factor.nuisance_jacobian is not None and factor.nuisance_jacobian.shape[1] != 0:
                    raise ValueError("Factor nuisance Jacobian must be empty when the nuisance layout is empty.")
            else:
                if factor.nuisance_jacobian is None:
                    raise ValueError("Every factor must carry a global nuisance Jacobian when the window nuisance layout is non-empty.")
                if factor.nuisance_jacobian.shape[1] != nuisance_dimension:
                    raise ValueError("Factor nuisance Jacobian width must match the nuisance layout.")


def build_gyroscope_nuisance_layout(metadata: Mapping[str, SensorMetadata], calibration_layout: CalibrationParameterLayout) -> NuisanceParameterLayout:
    """Build one three-axis gyroscope-bias nuisance block per IMU calibration key.

    Ordering follows the already explicit calibration layout rather than
    incidental dictionary insertion order.
    """

    gyroscope_calibration_keys = {stream_metadata.calibration_key for stream_metadata in metadata.values() if stream_metadata.measurement_type == MeasurementType.IMU_GYROSCOPE}
    parameter_blocks = [(gyroscope_bias_nuisance_key(calibration_key), GYROSCOPE_BIAS_PARAMETER_NAMES) for calibration_key in calibration_layout.calibration_keys if calibration_key in gyroscope_calibration_keys]

    if not parameter_blocks:
        return NuisanceParameterLayout.empty()

    return NuisanceParameterLayout.from_parameter_blocks(parameter_blocks)


def _single_window_calibration_state(calibration_state: CalibrationState, *, calibration_key: str) -> tuple[torch.Tensor, float]:
    """Extract one unbatched scientific calibration state.

    CalibrationState uses the project batch convention, therefore a
    WindowSample must supply batch size one here.
    """

    calibration_state.validate()

    if calibration_state.transform.shape[0] != 1:
        raise ValueError(f"Single-window observability expects calibration {calibration_key!r} to have batch size one.")

    transform = _to_scientific_tensor(calibration_state.transform[0], name=f"calibration[{calibration_key!r}].transform")
    time_offset_tensor = _to_scientific_tensor(calibration_state.time_offset[0, 0], name=f"calibration[{calibration_key!r}].time_offset")

    return transform, float(time_offset_tensor.item())


def _gyroscope_bias_for_key(gyro_bias_by_calibration_key: Mapping[str, torch.Tensor] | None, calibration_key: str) -> torch.Tensor:
    """Return the gyroscope-bias linearization point for one IMU.

    A missing bias estimate uses zero as the initial nuisance-state value, while
    the corresponding bias columns remain present and are marginalized later.
    """

    if gyro_bias_by_calibration_key is None or calibration_key not in gyro_bias_by_calibration_key:
        return torch.zeros(3, dtype=torch.float64)

    bias = _to_scientific_tensor(gyro_bias_by_calibration_key[calibration_key], name=f"gyro_bias_by_calibration_key[{calibration_key!r}]")

    if bias.shape == (1, 3):
        bias = bias[0]

    if bias.shape != (3,):
        raise ValueError(f"Gyroscope bias for calibration key {calibration_key!r} must have shape [3] or [1, 3].")

    return bias


def _residual_covariance_for_stream(residual_covariance_by_stream: Mapping[str, torch.Tensor] | None, stream_key: str, residual_dimension: int) -> torch.Tensor | None:
    """Return one constant residual covariance for a stream when supplied.

    The current interface deliberately does not invent an identity covariance.
    More detailed interval-dependent IMU covariance belongs to the later
    whitening/noise migration.
    """

    if residual_covariance_by_stream is None or stream_key not in residual_covariance_by_stream:
        return None

    covariance = _to_scientific_tensor(residual_covariance_by_stream[stream_key], name=f"residual_covariance_by_stream[{stream_key!r}]")

    if covariance.shape != (residual_dimension, residual_dimension):
        raise ValueError(f"Residual covariance for stream {stream_key!r} must have shape [{residual_dimension}, {residual_dimension}].")

    return covariance


def _require_strictly_increasing_timestamps(timestamps: torch.Tensor, *, name: str) -> None:
    """Require a nonempty one-dimensional strictly increasing time sequence.

    searchsorted-based support selection and interval reconstruction both rely
    on this ordering.
    """

    if timestamps.ndim != 1:
        raise ValueError(f"{name} must have shape [N].")
    if timestamps.numel() == 0:
        raise ValueError(f"{name} must not be empty.")
    if timestamps.numel() > 1 and torch.any(timestamps[1:] <= timestamps[:-1]):
        raise ValueError(f"{name} must be strictly increasing.")


def _select_accelerometer_support(stream: SensorStream, reference_timestamps: torch.Tensor, tau_I: float) -> tuple[AccelerometerSampleSupport, ...]:
    """Select approximately one actual accelerometer sample per reference timestamp.

    Selection happens on the sensor clock using t_sensor = t_true - tau_I.
    Duplicate nearest-neighbor sample indices are removed, so a low-rate or
    irregular stream is never counted twice merely because two reference
    timestamps choose the same measurement.
    """

    stream.validate()

    timestamps = _to_scientific_tensor(stream.timestamps, name="accelerometer timestamps")
    _require_strictly_increasing_timestamps(timestamps, name="accelerometer timestamps")

    target_sensor_times = reference_timestamps - float(tau_I)
    selected_indices: list[int] = []

    for target_sensor_time in target_sensor_times:
        if target_sensor_time < timestamps[0] or target_sensor_time > timestamps[-1]:
            continue

        upper_index = int(torch.searchsorted(timestamps, target_sensor_time, right=False).item())
        upper_index = min(max(upper_index, 0), timestamps.numel() - 1)
        lower_index = max(upper_index - 1, 0)

        if abs(float(timestamps[lower_index] - target_sensor_time)) <= abs(float(timestamps[upper_index] - target_sensor_time)):
            selected_index = lower_index
        else:
            selected_index = upper_index

        if not selected_indices or selected_index != selected_indices[-1]:
            selected_indices.append(selected_index)

    supports: list[AccelerometerSampleSupport] = []

    for sample_index in selected_indices:
        sensor_time = float(timestamps[sample_index].item())
        true_time = sensor_time + float(tau_I)

        if true_time < float(reference_timestamps[0].item()) or true_time > float(reference_timestamps[-1].item()):
            continue

        supports.append(AccelerometerSampleSupport(sample_index=sample_index, sensor_time=sensor_time, true_time=true_time))

    return tuple(supports)


def _lidar_interval_support(stream: SensorStream, tau_L: float, reference_timestamps: torch.Tensor, interval_start_timestamps: torch.Tensor | None) -> tuple[LidarIntervalSupport, ...]:
    """Recover LiDAR relative-pose intervals and apply the current time offset.

    When exact interval-start timestamps are supplied, every valid relative
    pose can be used. With the current SensorStream contract, which stores only
    one timestamp per relative pose, the fallback infers each start from the
    previous stored end timestamp and therefore drops the first measurement of
    the window.

    The fallback is valid only for an undecimated consecutive relative-pose
    stream. Carrying exact interval starts through data structures/windowing is
    the preferred next repository change.
    """

    stream.validate()

    timestamps = _to_scientific_tensor(stream.timestamps, name="lidar timestamps")
    _require_strictly_increasing_timestamps(timestamps, name="lidar timestamps")

    if interval_start_timestamps is not None:
        starts = _to_scientific_tensor(interval_start_timestamps, name="lidar interval_start_timestamps")

        if starts.shape != timestamps.shape:
            raise ValueError("LiDAR interval_start_timestamps must have the same shape as stream.timestamps.")

        measurement_indices = range(timestamps.numel())
        sensor_start_times = starts
        sensor_end_times = timestamps
    else:
        if timestamps.numel() < 2:
            return ()

        measurement_indices = range(1, timestamps.numel())
        sensor_start_times = timestamps[:-1]
        sensor_end_times = timestamps[1:]

    supports: list[LidarIntervalSupport] = []
    reference_start = float(reference_timestamps[0].item())
    reference_end = float(reference_timestamps[-1].item())

    for local_index, measurement_index in enumerate(measurement_indices):
        sensor_start_time = float(sensor_start_times[local_index].item())
        sensor_end_time = float(sensor_end_times[local_index].item())
        true_start_time = sensor_start_time + float(tau_L)
        true_end_time = sensor_end_time + float(tau_L)

        if true_end_time <= true_start_time + SUPPORT_TIME_TOLERANCE_S:
            continue
        if true_start_time < reference_start or true_end_time > reference_end:
            continue

        supports.append(LidarIntervalSupport(measurement_index=int(measurement_index), sensor_start_time=sensor_start_time, sensor_end_time=sensor_end_time, true_start_time=true_start_time, true_end_time=true_end_time))

    return tuple(supports)


def _coalesce_support_times(reference_timestamps: torch.Tensor, accelerometer_support_by_stream: Mapping[str, tuple[AccelerometerSampleSupport, ...]], lidar_support_by_stream: Mapping[str, tuple[LidarIntervalSupport, ...]]) -> torch.Tensor:
    """Build one sorted low-rate trajectory support grid for the window.

    The grid always contains the configured reference timestamps and adds only
    selected accelerometer times plus LiDAR interval endpoints. Raw high-rate
    IMU timestamps are not inserted wholesale.
    """

    values = [float(value.item()) for value in reference_timestamps]

    for supports in accelerometer_support_by_stream.values():
        values.extend(support.true_time for support in supports)

    for supports in lidar_support_by_stream.values():
        for support in supports:
            values.append(support.true_start_time)
            values.append(support.true_end_time)

    values.sort()

    coalesced: list[float] = []

    for value in values:
        if not coalesced or abs(value - coalesced[-1]) > SUPPORT_TIME_TOLERANCE_S:
            coalesced.append(value)

    if len(coalesced) < 2:
        raise ValueError("Trajectory support construction produced fewer than two unique timestamps.")

    return torch.tensor(coalesced, dtype=torch.float64)


def _trajectory_index_for_time(trajectory_layout: TrajectoryParameterLayout, timestamp: float) -> int:
    """Find the trajectory support index corresponding to one factor time.

    Support times were inserted into the layout before interpolation, so only a
    small floating-point coalescing tolerance is permitted here.
    """

    differences = torch.abs(trajectory_layout.timestamps - float(timestamp))
    index = int(torch.argmin(differences).item())

    if float(differences[index].item()) > SUPPORT_TIME_TOLERANCE_S:
        raise ValueError(f"Timestamp {timestamp} is not represented in the trajectory layout.")

    return index


def linearize_gyroscope_stream(stream_key: str, stream: SensorStream, stream_metadata: SensorMetadata, calibration_state: CalibrationState, calibration_layout: CalibrationParameterLayout, nuisance_layout: NuisanceParameterLayout, trajectory_layout: TrajectoryParameterLayout, trajectory_poses: torch.Tensor, *, gyro_bias: torch.Tensor | None = None, residual_covariance: torch.Tensor | None = None) -> tuple[SensorFactorLinearization, ...]:
    """Linearize every supported trajectory interval for one gyroscope stream.

    All raw gyroscope samples contribute through exact piecewise-linear
    integration inside each trajectory interval. The raw signal is not
    decimated to the trajectory rate.
    """

    if not stream_key:
        raise ValueError("stream_key must be non-empty.")
    if stream_metadata.measurement_type != MeasurementType.IMU_GYROSCOPE:
        raise ValueError(f"Stream {stream_key!r} is not an IMU gyroscope stream.")

    stream.validate()
    trajectory_layout.validate()

    gyroscope_samples = _to_scientific_tensor(stream.values, name=f"{stream_key}.values")
    gyroscope_timestamps = _to_scientific_tensor(stream.timestamps, name=f"{stream_key}.timestamps")
    trajectory_poses = _to_scientific_tensor(trajectory_poses, name="trajectory_poses")

    if gyroscope_samples.ndim != 2 or gyroscope_samples.shape[1] != 3:
        raise ValueError(f"Gyroscope stream {stream_key!r} must contain values with shape [N, 3].")
    if trajectory_poses.shape != (trajectory_layout.pose_count, 4, 4):
        raise ValueError(f"trajectory_poses must have shape [{trajectory_layout.pose_count}, 4, 4].")

    body_from_imu, imu_time_offset = _single_window_calibration_state(calibration_state, calibration_key=stream_metadata.calibration_key)
    bias = torch.zeros(3, dtype=torch.float64) if gyro_bias is None else _to_scientific_tensor(gyro_bias, name=f"{stream_key}.gyro_bias")

    if bias.shape != (3,):
        raise ValueError("gyro_bias must have shape [3].")

    calibration_block = calibration_layout.block_for(stream_metadata.calibration_key)
    nuisance_block = nuisance_layout.block_for(gyroscope_bias_nuisance_key(stream_metadata.calibration_key))
    factors: list[SensorFactorLinearization] = []

    for interval_index in range(trajectory_layout.pose_count - 1):
        true_start_time = trajectory_layout.timestamps[interval_index]
        true_end_time = trajectory_layout.timestamps[interval_index + 1]

        if not gyroscope_interval_is_supported(gyroscope_timestamps, true_start_time, true_end_time, imu_time_offset):
            continue

        local = linearize_gyroscope_factor(trajectory_poses[interval_index], trajectory_poses[interval_index + 1], body_from_imu, bias, imu_time_offset, true_start_time, true_end_time, gyroscope_timestamps, gyroscope_samples)

        trajectory_jacobian = torch.zeros((3, trajectory_layout.total_dimension), dtype=torch.float64)
        trajectory_jacobian[:, trajectory_layout.pose_slice(interval_index)] = local.H_start_pose
        trajectory_jacobian[:, trajectory_layout.pose_slice(interval_index + 1)] = local.H_end_pose

        nuisance_jacobian = torch.zeros((3, nuisance_layout.total_dimension), dtype=torch.float64)
        nuisance_jacobian[:, nuisance_block.parameter_slice] = local.H_b_g

        calibration_jacobian = torch.zeros((3, calibration_layout.total_dimension), dtype=torch.float64)
        calibration_jacobian[:, calibration_block.spatial_slice] = local.H_T_B_I
        calibration_jacobian[:, calibration_block.time_offset_slice] = local.H_tau_I

        factor = SensorFactorLinearization(stream_key=stream_key, calibration_key=stream_metadata.calibration_key, residual=local.residual, trajectory_jacobian=trajectory_jacobian, calibration_jacobian=calibration_jacobian, residual_covariance=residual_covariance, nuisance_jacobian=nuisance_jacobian, measurement_type=MeasurementType.IMU_GYROSCOPE, factor_index=interval_index)
        factor.validate()
        factors.append(factor)

    return tuple(factors)


def linearize_accelerometer_stream(stream_key: str, stream: SensorStream, stream_metadata: SensorMetadata, calibration_state: CalibrationState, calibration_layout: CalibrationParameterLayout, nuisance_layout: NuisanceParameterLayout, trajectory_layout: TrajectoryParameterLayout, trajectory_poses: torch.Tensor, trajectory_spatial_twists: torch.Tensor, supports: tuple[AccelerometerSampleSupport, ...], gravity_world: torch.Tensor, *, residual_covariance: torch.Tensor | None = None) -> tuple[SensorFactorLinearization, ...]:
    """Linearize the selected simple accelerometer factors for one stream.

    One actual accelerometer sample is selected near each reference timestamp.
    This avoids creating a trajectory variable for every high-rate IMU sample
    while preserving low-rate gravity-alignment information.
    """

    if stream_metadata.measurement_type != MeasurementType.IMU_ACCELEROMETER:
        raise ValueError(f"Stream {stream_key!r} is not an IMU accelerometer stream.")

    values = _to_scientific_tensor(stream.values, name=f"{stream_key}.values")

    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"Accelerometer stream {stream_key!r} must contain values with shape [N, 3].")

    body_from_imu, tau_I = _single_window_calibration_state(calibration_state, calibration_key=stream_metadata.calibration_key)
    calibration_block = calibration_layout.block_for(stream_metadata.calibration_key)
    factors: list[SensorFactorLinearization] = []

    for support in supports:
        pose_index = _trajectory_index_for_time(trajectory_layout, support.true_time)
        local = linearize_simple_accelerometer_factor(trajectory_poses[pose_index], body_from_imu, values[support.sample_index], gravity_world, trajectory_spatial_twists[pose_index], sensor_time=support.sensor_time, tau_I=tau_I)

        trajectory_jacobian = torch.zeros((3, trajectory_layout.total_dimension), dtype=torch.float64)
        trajectory_jacobian[:, trajectory_layout.pose_slice(pose_index)] = local.H_pose

        nuisance_jacobian = torch.zeros((3, nuisance_layout.total_dimension), dtype=torch.float64)

        calibration_jacobian = torch.zeros((3, calibration_layout.total_dimension), dtype=torch.float64)
        calibration_jacobian[:, calibration_block.spatial_slice] = local.H_T_B_I
        calibration_jacobian[:, calibration_block.time_offset_slice] = local.H_tau_I

        factor = SensorFactorLinearization(stream_key=stream_key, calibration_key=stream_metadata.calibration_key, residual=local.residual, trajectory_jacobian=trajectory_jacobian, calibration_jacobian=calibration_jacobian, residual_covariance=residual_covariance, nuisance_jacobian=nuisance_jacobian, measurement_type=MeasurementType.IMU_ACCELEROMETER, factor_index=support.sample_index)
        factor.validate()
        factors.append(factor)

    return tuple(factors)


def linearize_lidar_stream(stream_key: str, stream: SensorStream, stream_metadata: SensorMetadata, calibration_state: CalibrationState, calibration_layout: CalibrationParameterLayout, nuisance_layout: NuisanceParameterLayout, trajectory_layout: TrajectoryParameterLayout, trajectory_poses: torch.Tensor, trajectory_spatial_twists: torch.Tensor, supports: tuple[LidarIntervalSupport, ...], *, residual_covariance: torch.Tensor | None = None) -> tuple[SensorFactorLinearization, ...]:
    """Linearize all supported LiDAR relative-pose factors for one stream.

    Each factor connects the two trajectory support nodes corresponding to the
    LiDAR interval endpoints after applying the current temporal offset.
    """

    if stream_metadata.measurement_type != MeasurementType.LIDAR_POSE:
        raise ValueError(f"Stream {stream_key!r} is not a LiDAR pose stream.")

    values = _to_scientific_tensor(stream.values, name=f"{stream_key}.values")

    if values.ndim != 3 or values.shape[-2:] != (4, 4):
        raise ValueError(f"LiDAR stream {stream_key!r} must contain relative poses with shape [N, 4, 4].")

    body_from_lidar, _ = _single_window_calibration_state(calibration_state, calibration_key=stream_metadata.calibration_key)
    calibration_block = calibration_layout.block_for(stream_metadata.calibration_key)
    factors: list[SensorFactorLinearization] = []

    for support in supports:
        start_pose_index = _trajectory_index_for_time(trajectory_layout, support.true_start_time)
        end_pose_index = _trajectory_index_for_time(trajectory_layout, support.true_end_time)

        local = linearize_lidar_factor(trajectory_poses[start_pose_index], trajectory_poses[end_pose_index], body_from_lidar, values[support.measurement_index], trajectory_spatial_twists[start_pose_index], trajectory_spatial_twists[end_pose_index])

        trajectory_jacobian = torch.zeros((6, trajectory_layout.total_dimension), dtype=torch.float64)
        trajectory_jacobian[:, trajectory_layout.pose_slice(start_pose_index)] = local.H_start_pose
        trajectory_jacobian[:, trajectory_layout.pose_slice(end_pose_index)] = local.H_end_pose

        nuisance_jacobian = torch.zeros((6, nuisance_layout.total_dimension), dtype=torch.float64)

        calibration_jacobian = torch.zeros((6, calibration_layout.total_dimension), dtype=torch.float64)
        calibration_jacobian[:, calibration_block.spatial_slice] = local.H_T_B_L
        calibration_jacobian[:, calibration_block.time_offset_slice] = local.H_tau_L

        factor = SensorFactorLinearization(stream_key=stream_key, calibration_key=stream_metadata.calibration_key, residual=local.residual, trajectory_jacobian=trajectory_jacobian, calibration_jacobian=calibration_jacobian, residual_covariance=residual_covariance, nuisance_jacobian=nuisance_jacobian, measurement_type=MeasurementType.LIDAR_POSE, factor_index=support.measurement_index)
        factor.validate()
        factors.append(factor)

    return tuple(factors)


def linearize_gyroscope_factors_single_window(measurements: Mapping[str, SensorStream], metadata: Mapping[str, SensorMetadata], calibration: Mapping[str, CalibrationState], calibration_layout: CalibrationParameterLayout, reference_timebase: ReferenceTimebase, trajectory_poses: torch.Tensor, *, gyro_bias_by_calibration_key: Mapping[str, torch.Tensor] | None = None, residual_covariance_by_stream: Mapping[str, torch.Tensor] | None = None) -> WindowLinearization:
    """Build the explicitly gyroscope-only linearization used during migration tests.

    This compatibility helper preserves the previous public entry point. It
    uses exactly the configured reference timebase and does not add
    accelerometer or LiDAR support timestamps.
    """

    if not measurements:
        raise ValueError("At least one sensor stream is required.")
    if set(measurements) != set(metadata):
        raise ValueError("measurements and metadata must contain identical stream keys.")

    reference_timebase.validate()

    trajectory_timestamps = _to_scientific_tensor(reference_timebase.timestamps, name="reference_timebase.timestamps")
    trajectory_layout = TrajectoryParameterLayout(timestamps=trajectory_timestamps)
    trajectory_layout.validate()
    trajectory_poses_scientific = _to_scientific_tensor(trajectory_poses, name="trajectory_poses")

    if trajectory_poses_scientific.shape != (trajectory_layout.pose_count, 4, 4):
        raise ValueError(f"trajectory_poses must have shape [{trajectory_layout.pose_count}, 4, 4].")

    nuisance_layout = build_gyroscope_nuisance_layout(metadata, calibration_layout)
    factors: list[SensorFactorLinearization] = []

    for stream_key in sorted(metadata):
        stream_metadata = metadata[stream_key]

        if stream_metadata.measurement_type != MeasurementType.IMU_GYROSCOPE:
            continue
        if stream_metadata.calibration_key not in calibration:
            raise KeyError(f"Gyroscope stream {stream_key!r} requires missing calibration key {stream_metadata.calibration_key!r}.")

        gyro_bias = _gyroscope_bias_for_key(gyro_bias_by_calibration_key, stream_metadata.calibration_key)
        residual_covariance = _residual_covariance_for_stream(residual_covariance_by_stream, stream_key, 3)
        factors.extend(linearize_gyroscope_stream(stream_key, measurements[stream_key], stream_metadata, calibration[stream_metadata.calibration_key], calibration_layout, nuisance_layout, trajectory_layout, trajectory_poses_scientific, gyro_bias=gyro_bias, residual_covariance=residual_covariance))

    if not factors:
        raise ValueError("No supported gyroscope factors could be constructed for this window.")

    result = WindowLinearization(factors=tuple(factors), calibration_layout=calibration_layout, reference_timebase=reference_timebase, trajectory_layout=trajectory_layout, nuisance_layout=nuisance_layout)
    result.validate()

    return result


def linearize_single_window(measurements: Mapping[str, SensorStream], metadata: Mapping[str, SensorMetadata], calibration: Mapping[str, CalibrationState], calibration_layout: CalibrationParameterLayout, reference_timebase: ReferenceTimebase, *, trajectory_poses: torch.Tensor | None = None, gyro_bias_by_calibration_key: Mapping[str, torch.Tensor] | None = None, residual_covariance_by_stream: Mapping[str, torch.Tensor] | None = None, gravity_world: torch.Tensor | None = None, lidar_interval_start_timestamps_by_stream: Mapping[str, torch.Tensor] | None = None) -> WindowLinearization:
    """Linearize every currently supported sensor factor in one temporal window.

    The reference stream defines the coarse trajectory source. Additional
    low-rate support times are added only for selected accelerometer samples and
    LiDAR interval endpoints; the complete high-rate IMU timestamp set is not
    promoted to trajectory variables.

    All raw gyroscope samples still contribute through interval integration.
    Accelerometer uses approximately one real sample per reference timestamp.
    LiDAR uses every relative-pose interval whose corrected endpoints lie inside
    the reference trajectory support.

    Unsupported measurement modalities raise explicitly so a partial Fisher
    matrix cannot masquerade as complete-window observability.
    """

    if not measurements:
        raise ValueError("At least one sensor stream is required.")
    if set(measurements) != set(metadata):
        raise ValueError("measurements and metadata must contain identical stream keys.")

    reference_timebase.validate()

    unsupported_streams = [stream_key for stream_key, stream_metadata in metadata.items() if stream_metadata.measurement_type not in (MeasurementType.IMU_GYROSCOPE, MeasurementType.IMU_ACCELEROMETER, MeasurementType.LIDAR_POSE)]

    if unsupported_streams:
        raise NotImplementedError(f"Observability factors are not yet implemented for streams {unsupported_streams!r}.")

    for stream_key, stream_metadata in metadata.items():
        if stream_metadata.calibration_key not in calibration:
            raise KeyError(f"Stream {stream_key!r} requires missing calibration key {stream_metadata.calibration_key!r}.")
        if stream_metadata.calibration_key not in calibration_layout.calibration_keys:
            raise KeyError(f"Stream {stream_key!r} uses calibration key {stream_metadata.calibration_key!r} outside the calibration layout.")

    if trajectory_poses is None:
        raise ValueError("trajectory_poses must be provided for complete single-window linearization.")

    reference_timestamps = _to_scientific_tensor(reference_timebase.timestamps, name="reference_timebase.timestamps")
    reference_trajectory_poses = _to_scientific_tensor(trajectory_poses, name="trajectory_poses")

    if reference_trajectory_poses.shape != (reference_timestamps.numel(), 4, 4):
        raise ValueError(f"trajectory_poses must have shape [{reference_timestamps.numel()}, 4, 4] matching the reference timebase.")

    if gravity_world is not None:
        gravity_scientific = _to_scientific_tensor(gravity_world, name="gravity_world")
        if gravity_scientific.shape != (3,):
            raise ValueError("gravity_world must have shape [3].")
    else:
        gravity_scientific = None

    accelerometer_present = any(stream_metadata.measurement_type == MeasurementType.IMU_ACCELEROMETER for stream_metadata in metadata.values())

    if accelerometer_present and gravity_scientific is None:
        raise ValueError("gravity_world must be provided when accelerometer observability factors are present.")

    accelerometer_support_by_stream: dict[str, tuple[AccelerometerSampleSupport, ...]] = {}
    lidar_support_by_stream: dict[str, tuple[LidarIntervalSupport, ...]] = {}

    for stream_key in sorted(metadata):
        stream_metadata = metadata[stream_key]

        if stream_metadata.measurement_type == MeasurementType.IMU_ACCELEROMETER:
            _, tau_I = _single_window_calibration_state(calibration[stream_metadata.calibration_key], calibration_key=stream_metadata.calibration_key)
            accelerometer_support_by_stream[stream_key] = _select_accelerometer_support(measurements[stream_key], reference_timestamps, tau_I)

        elif stream_metadata.measurement_type == MeasurementType.LIDAR_POSE:
            _, tau_L = _single_window_calibration_state(calibration[stream_metadata.calibration_key], calibration_key=stream_metadata.calibration_key)
            interval_start_timestamps = None if lidar_interval_start_timestamps_by_stream is None else lidar_interval_start_timestamps_by_stream.get(stream_key)
            lidar_support_by_stream[stream_key] = _lidar_interval_support(measurements[stream_key], tau_L, reference_timestamps, interval_start_timestamps)

    trajectory_timestamps = _coalesce_support_times(reference_timestamps, accelerometer_support_by_stream, lidar_support_by_stream)
    trajectory_layout = TrajectoryParameterLayout(timestamps=trajectory_timestamps)
    trajectory_layout.validate()

    trajectory_poses_scientific, _, trajectory_spatial_twists = interpolate_se3_trajectory_with_twist(reference_timestamps, reference_trajectory_poses, trajectory_timestamps)
    nuisance_layout = build_gyroscope_nuisance_layout(metadata, calibration_layout)
    factors: list[SensorFactorLinearization] = []

    for stream_key in sorted(metadata):
        stream_metadata = metadata[stream_key]
        stream = measurements[stream_key]

        if stream_metadata.measurement_type == MeasurementType.IMU_GYROSCOPE:
            gyro_bias = _gyroscope_bias_for_key(gyro_bias_by_calibration_key, stream_metadata.calibration_key)
            residual_covariance = _residual_covariance_for_stream(residual_covariance_by_stream, stream_key, 3)
            factors.extend(linearize_gyroscope_stream(stream_key, stream, stream_metadata, calibration[stream_metadata.calibration_key], calibration_layout, nuisance_layout, trajectory_layout, trajectory_poses_scientific, gyro_bias=gyro_bias, residual_covariance=residual_covariance))

        elif stream_metadata.measurement_type == MeasurementType.IMU_ACCELEROMETER:
            residual_covariance = _residual_covariance_for_stream(residual_covariance_by_stream, stream_key, 3)
            factors.extend(linearize_accelerometer_stream(stream_key, stream, stream_metadata, calibration[stream_metadata.calibration_key], calibration_layout, nuisance_layout, trajectory_layout, trajectory_poses_scientific, trajectory_spatial_twists, accelerometer_support_by_stream[stream_key], gravity_scientific, residual_covariance=residual_covariance))

        elif stream_metadata.measurement_type == MeasurementType.LIDAR_POSE:
            residual_covariance = _residual_covariance_for_stream(residual_covariance_by_stream, stream_key, 6)
            factors.extend(linearize_lidar_stream(stream_key, stream, stream_metadata, calibration[stream_metadata.calibration_key], calibration_layout, nuisance_layout, trajectory_layout, trajectory_poses_scientific, trajectory_spatial_twists, lidar_support_by_stream[stream_key], residual_covariance=residual_covariance))

    if not factors:
        raise ValueError("No supported sensor factors could be constructed for this window.")

    result = WindowLinearization(factors=tuple(factors), calibration_layout=calibration_layout, reference_timebase=reference_timebase, trajectory_layout=trajectory_layout, nuisance_layout=nuisance_layout)
    result.validate()

    return result


__all__ = [
    "AccelerometerSampleSupport",
    "LidarIntervalSupport",
    "SUPPORT_TIME_TOLERANCE_S",
    "SensorFactorLinearization",
    "TRAJECTORY_POSE_DIM",
    "TrajectoryParameterLayout",
    "WindowLinearization",
    "build_gyroscope_nuisance_layout",
    "linearize_accelerometer_stream",
    "linearize_gyroscope_factors_single_window",
    "linearize_gyroscope_stream",
    "linearize_lidar_stream",
    "linearize_single_window",
]