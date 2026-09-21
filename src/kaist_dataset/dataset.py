"""RAM-backed KAIST window dataset and DataLoader construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from obscalib.config import WindowingConfig
from obscalib.data.adapter import PreprocessedCalibration, PreprocessedSensorStream, build_window_samples_from_preprocessed
from obscalib.data.collation import collate_windows
from obscalib.data.structures import GeometryType, MeasurementType, SensorStream, WindowBatch, WindowSample

from .imu import IMUData, load_imus
from .lidar import load_lidar_relative_poses


@dataclass(frozen=True)
class KAISTSequenceSpec:
    """Paths and optional IMU selection for one KAIST sequence."""

    name: str
    sensor_data_root: Path
    lidar_pose_csv: Path
    imu_stream_key: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("KAIST sequence name must be non-empty.")

        object.__setattr__(self, "sensor_data_root", Path(self.sensor_data_root).expanduser())
        object.__setattr__(self, "lidar_pose_csv", Path(self.lidar_pose_csv).expanduser())


@dataclass(frozen=True)
class KAISTRigCalibration:
    """
    Static calibration shared by KAIST sequences.

    T_B_I:
        IMU sensor frame to common body frame.

    T_B_L:
        LiDAR sensor frame to common body frame.

    tau_imu_s / tau_lidar_s:
        Initial additive temporal calibrations.
    """

    T_B_I: np.ndarray
    T_B_L: np.ndarray
    tau_imu_s: float = 0.0
    tau_lidar_s: float = 0.0

    def __post_init__(self) -> None:
        T_B_I = np.asarray(self.T_B_I, dtype=float)
        T_B_L = np.asarray(self.T_B_L, dtype=float)

        if T_B_I.shape != (4, 4):
            raise ValueError("T_B_I must have shape [4, 4].")

        if T_B_L.shape != (4, 4):
            raise ValueError("T_B_L must have shape [4, 4].")

        if not np.all(np.isfinite(T_B_I)) or not np.all(np.isfinite(T_B_L)):
            raise ValueError("KAIST rig calibration transforms must contain only finite values.")

        if not np.isfinite(self.tau_imu_s) or not np.isfinite(self.tau_lidar_s):
            raise ValueError("KAIST temporal calibrations must be finite.")

        object.__setattr__(self, "T_B_I", T_B_I)
        object.__setattr__(self, "T_B_L", T_B_L)


@dataclass(frozen=True)
class KAISTSequenceInfo:
    """Summary of one sequence loaded into a KAISTCalibrationDataset."""

    name: str
    imu_stream_key: str
    num_imu_samples: int
    num_lidar_relative_poses: int
    num_windows: int
    common_start_time_s: float
    common_end_time_s: float


def build_kaist_sequence_specs(sequence_names: Sequence[str], kaist_dataset_root: str | Path, lidar_pose_root: str | Path, lidar_pose_filename: str = "lidar_map_poses_vlp_left.csv") -> list[KAISTSequenceSpec]:
    """
    Build conventional KAIST paths from sequence names.

    For sequence Urban16 this assumes:

        <kaist_dataset_root>/Urban16/urban16_data/urban16/sensor_data

    and:

        <lidar_pose_root>/Urban16/lidar_map_poses_vlp_left.csv
    """

    kaist_dataset_root = Path(kaist_dataset_root).expanduser()
    lidar_pose_root = Path(lidar_pose_root).expanduser()

    specs: list[KAISTSequenceSpec] = []

    for sequence_name in sequence_names:
        sequence_slug = sequence_name.lower()

        specs.append(
            KAISTSequenceSpec(
                name=sequence_name,
                sensor_data_root=kaist_dataset_root / sequence_name / f"{sequence_slug}_data" / sequence_slug / "sensor_data",
                lidar_pose_csv=lidar_pose_root / sequence_name / lidar_pose_filename,
            )
        )

    return specs


class KAISTCalibrationDataset(Dataset[WindowSample]):
    """
    RAM-backed collection of preprocessed KAIST temporal windows.

    Source files are loaded sequence-by-sequence from disk during construction.
    Each sequence is converted to WindowSample objects immediately and the
    resulting windows remain in RAM for the lifetime of the dataset.

    Raw point clouds are never loaded. LiDAR input consists only of the
    precomputed scan-to-map pose sequence converted to relative SE3 updates.
    """

    def __init__(self, sequence_specs: Sequence[KAISTSequenceSpec], rig_calibration: KAISTRigCalibration, windowing_config: WindowingConfig, *, measurement_dtype: torch.dtype = torch.float32, source_timestamp_dtype: torch.dtype = torch.float64, window_timestamp_dtype: torch.dtype = torch.float32) -> None:
        if not sequence_specs:
            raise ValueError("KAISTCalibrationDataset requires at least one sequence.")

        if not measurement_dtype.is_floating_point:
            raise TypeError("measurement_dtype must be floating-point.")

        if not source_timestamp_dtype.is_floating_point:
            raise TypeError("source_timestamp_dtype must be floating-point.")

        if not window_timestamp_dtype.is_floating_point:
            raise TypeError("window_timestamp_dtype must be floating-point.")

        sequence_names = [spec.name for spec in sequence_specs]

        if len(sequence_names) != len(set(sequence_names)):
            raise ValueError("KAIST sequence names must be unique within one dataset.")

        self.sequence_specs = tuple(sequence_specs)
        self.rig_calibration = rig_calibration
        self.windowing_config = windowing_config
        self.measurement_dtype = measurement_dtype
        self.source_timestamp_dtype = source_timestamp_dtype
        self.window_timestamp_dtype = window_timestamp_dtype

        self._windows: list[WindowSample] = []
        self._window_sequence_names: list[str] = []
        self._sequence_info: list[KAISTSequenceInfo] = []

        for spec in self.sequence_specs:
            sequence_windows, sequence_info = self._load_sequence(spec)

            self._windows.extend(sequence_windows)
            self._window_sequence_names.extend([spec.name] * len(sequence_windows))
            self._sequence_info.append(sequence_info)

        if not self._windows:
            raise ValueError("No usable windows were produced from the requested KAIST sequences.")

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> WindowSample:
        return self._windows[index]

    @property
    def sequence_info(self) -> tuple[KAISTSequenceInfo, ...]:
        return tuple(self._sequence_info)

    @property
    def sequence_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.sequence_specs)

    def sequence_name_for_window(self, index: int) -> str:
        """Return the source sequence name for one materialized window."""

        return self._window_sequence_names[index]

    def estimated_tensor_bytes(self) -> int:
        """Return the approximate number of tensor bytes occupied by all materialized windows."""

        return sum(_window_sample_tensor_bytes(window) for window in self._windows)

    def estimated_tensor_megabytes(self) -> float:
        """Return the approximate tensor storage occupied by all windows in MiB."""

        return self.estimated_tensor_bytes() / (1024.0 ** 2)

    def summary(self) -> str:
        """Return a compact human-readable dataset summary."""

        lines = [
            f"KAISTCalibrationDataset: {len(self)} windows from {len(self.sequence_specs)} sequences",
            f"approximate tensor storage: {self.estimated_tensor_megabytes():.1f} MiB",
        ]

        for info in self._sequence_info:
            duration_s = info.common_end_time_s - info.common_start_time_s
            lines.append(f"  {info.name}: windows={info.num_windows}, duration={duration_s:.1f}s, IMU={info.num_imu_samples}, LiDAR={info.num_lidar_relative_poses}")

        return "\n".join(lines)

    def _load_sequence(self, spec: KAISTSequenceSpec) -> tuple[list[WindowSample], KAISTSequenceInfo]:
        """Load one KAIST sequence and materialize its accepted windows."""

        if not spec.sensor_data_root.is_dir():
            raise FileNotFoundError(f"KAIST sensor-data directory does not exist for {spec.name!r}: {spec.sensor_data_root}")

        if not spec.lidar_pose_csv.is_file():
            raise FileNotFoundError(f"KAIST LiDAR map-pose CSV does not exist for {spec.name!r}: {spec.lidar_pose_csv}")

        imu_streams = load_imus(spec.sensor_data_root, target_frequency_hz=None)
        imu_stream_key, imu = _select_primary_imu(imu_streams, spec)

        lidar = load_lidar_relative_poses(spec.lidar_pose_csv)

        streams = {
            "gyro": PreprocessedSensorStream(
                values=imu.gyro_radps,
                timestamps_s=imu.timestamps_s,
                measurement_type=MeasurementType.IMU_GYROSCOPE,
                geometry_type=GeometryType.VECTOR,
                calibration_key="imu",
            ),
            "accel": PreprocessedSensorStream(
                values=imu.accel_mps2,
                timestamps_s=imu.timestamps_s,
                measurement_type=MeasurementType.IMU_ACCELEROMETER,
                geometry_type=GeometryType.VECTOR,
                calibration_key="imu",
            ),
            "lidar": PreprocessedSensorStream(
                values=lidar.relative_poses_se3,
                timestamps_s=lidar.scan_timestamps_s[1:],
                interval_start_timestamps_s=lidar.scan_timestamps_s[:-1],
                measurement_type=MeasurementType.LIDAR_POSE,
                geometry_type=GeometryType.SE3,
                calibration_key="lidar",
            ),
        }

        current_calibration = {
            "imu": PreprocessedCalibration(transform=self.rig_calibration.T_B_I, time_offset_s=self.rig_calibration.tau_imu_s),
            "lidar": PreprocessedCalibration(transform=self.rig_calibration.T_B_L, time_offset_s=self.rig_calibration.tau_lidar_s),
        }

        common_start_time_s = max(float(imu.timestamps_s[0]), float(lidar.scan_timestamps_s[0]))
        common_end_time_s = min(float(imu.timestamps_s[-1]), float(lidar.scan_timestamps_s[-1]))

        windows = build_window_samples_from_preprocessed(
            streams,
            current_calibration,
            windowing_config=self.windowing_config,
            start_time=common_start_time_s,
            end_time=common_end_time_s,
            dtype=self.measurement_dtype,
            timestamp_dtype=self.source_timestamp_dtype,
        )

        windows = [_cast_window_relative_timestamps(window, self.window_timestamp_dtype) for window in windows]

        info = KAISTSequenceInfo(
            name=spec.name,
            imu_stream_key=imu_stream_key,
            num_imu_samples=int(len(imu.timestamps_s)),
            num_lidar_relative_poses=int(len(lidar.relative_poses_se3)),
            num_windows=len(windows),
            common_start_time_s=common_start_time_s,
            common_end_time_s=common_end_time_s,
        )

        return windows, info


def make_kaist_dataloader(dataset: KAISTCalibrationDataset, *, batch_size: int, shuffle: bool, drop_last: bool = False, num_workers: int = 0, generator: torch.Generator | None = None) -> DataLoader:
    """
    Construct a DataLoader for a RAM-backed KAIST dataset.

    num_workers=0 is the recommended initial configuration because all expensive
    file loading and window construction already happened in dataset.__init__.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative.")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate_windows,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def _select_primary_imu(imu_streams: dict[str, IMUData], spec: KAISTSequenceSpec) -> tuple[str, IMUData]:
    """Resolve the IMU stream used for model training."""

    if not imu_streams:
        raise ValueError(f"No IMU streams were found for KAIST sequence {spec.name!r}.")

    if spec.imu_stream_key is not None:
        if spec.imu_stream_key not in imu_streams:
            raise KeyError(f"Requested IMU stream {spec.imu_stream_key!r} was not found for {spec.name!r}. Available streams: {sorted(imu_streams)}")

        return spec.imu_stream_key, imu_streams[spec.imu_stream_key]

    if len(imu_streams) == 1:
        imu_stream_key = next(iter(imu_streams))
        return imu_stream_key, imu_streams[imu_stream_key]

    xsens_candidates = [key for key in imu_streams if "xsens" in key.lower()]

    if len(xsens_candidates) == 1:
        imu_stream_key = xsens_candidates[0]
        return imu_stream_key, imu_streams[imu_stream_key]

    raise ValueError(f"Multiple IMU streams were found for {spec.name!r}. Set KAISTSequenceSpec.imu_stream_key explicitly. Available streams: {sorted(imu_streams)}")


def _cast_window_relative_timestamps(window: WindowSample, dtype: torch.dtype) -> WindowSample:
    """
    Cast already window-relative timestamps to the training timestamp dtype.

    Absolute epoch timestamps are kept in float64 until windowing. Once the
    window origin has been subtracted, float32 is sufficient for ordinary
    neural-network training.
    """

    streams: dict[str, SensorStream] = {}

    for stream_key, stream in window.streams.items():
        streams[stream_key] = SensorStream(
            values=stream.values,
            timestamps=stream.timestamps.to(dtype=dtype),
            interval_start_timestamps=None if stream.interval_start_timestamps is None else stream.interval_start_timestamps.to(dtype=dtype),
        )

    return WindowSample(
        streams=streams,
        current_calibration=dict(window.current_calibration),
        metadata=dict(window.metadata),
        targets=None if window.targets is None else dict(window.targets),
    )


def _window_sample_tensor_bytes(window: WindowSample) -> int:
    """Count tensor storage owned by one materialized WindowSample."""

    total_bytes = 0

    for stream in window.streams.values():
        total_bytes += _tensor_bytes(stream.values)
        total_bytes += _tensor_bytes(stream.timestamps)

        if stream.interval_start_timestamps is not None:
            total_bytes += _tensor_bytes(stream.interval_start_timestamps)

    for state in window.current_calibration.values():
        total_bytes += _tensor_bytes(state.transform)
        total_bytes += _tensor_bytes(state.time_offset)

    if window.targets is not None:
        for target in window.targets.values():
            for field_name in ("next_transform", "next_time_offset", "change_label", "change_time"):
                value = getattr(target, field_name, None)

                if isinstance(value, torch.Tensor):
                    total_bytes += _tensor_bytes(value)

    return total_bytes


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()

class KAISTSequenceChunkDataset(Dataset[tuple[WindowSample, ...]]):
    """Expose consecutive windows from one KAISTCalibrationDataset as fixed-length sequence chunks."""

    def __init__(self, dataset: KAISTCalibrationDataset, sequence_length: int, sequence_stride: int | None = None) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")

        if sequence_stride is None:
            sequence_stride = sequence_length

        if sequence_stride <= 0:
            raise ValueError("sequence_stride must be positive.")

        self.dataset = dataset
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride

        indices_by_sequence: dict[str, list[int]] = {}

        for index in range(len(dataset)):
            sequence_name = dataset.sequence_name_for_window(index)

            indices_by_sequence.setdefault(
                sequence_name,
                [],
            ).append(index)

        self._chunk_indices: list[tuple[int, ...]] = []

        for sequence_name in dataset.sequence_names:
            sequence_indices = indices_by_sequence.get(
                sequence_name,
                [],
            )

            for start_index in range(
                0,
                len(sequence_indices) - sequence_length + 1,
                sequence_stride,
            ):
                chunk_indices = tuple(
                    sequence_indices[
                        start_index :
                        start_index + sequence_length
                    ]
                )

                self._chunk_indices.append(
                    chunk_indices
                )

        if not self._chunk_indices:
            raise ValueError("No complete sequence chunks could be constructed.")

    def __len__(self) -> int:
        return len(self._chunk_indices)

    def __getitem__(self, index: int) -> tuple[WindowSample, ...]:
        return tuple(
            self.dataset[window_index]
            for window_index in self._chunk_indices[index]
        )


def collate_kaist_sequence_chunks(samples: list[tuple[WindowSample, ...]]) -> tuple[WindowBatch, ...]:
    """Collate B independent sequence chunks into K ordinary WindowBatch objects."""

    if not samples:
        raise ValueError("Cannot collate an empty sequence batch.")

    sequence_length = len(samples[0])

    if any(len(sample) != sequence_length for sample in samples):
        raise ValueError("All sequence chunks must have the same sequence length.")

    return tuple(
        collate_windows(
            [
                sample[sequence_index]
                for sample in samples
            ]
        )
        for sequence_index in range(sequence_length)
    )