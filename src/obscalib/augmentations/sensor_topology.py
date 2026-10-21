
"""Generate KAIST 1-IMU/2-LiDAR and synthetic 2-IMU/1-LiDAR training topologies.

Synthetic IMU observations are physical, native VECTOR measurements; no LiDAR
measurements are fabricated. One output WindowBatch has one shared topology.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from obscalib.augmentations.config import SensorTopologyAugmentationConfig
from obscalib.calibration.state import CalibrationState
from obscalib.data.structures import GeometryType, MeasurementType, WindowBatch
from obscalib.geometry.lie import so3_exp


class SensorTopologyAugmenter:
    """Replace one genuine LiDAR with a co-located, independently oriented ideal IMU."""

    def __init__(self, config: SensorTopologyAugmentationConfig) -> None:
        self.config = config

    def __call__(self, window: WindowBatch, generator: torch.Generator | None = None) -> tuple[WindowBatch, str]:
        """Return (window, topology), with probabilities (1-p, p/2, p/2) by default.

        Sampling is per homogeneous batch, not independently per batch item.
        For true per-window sampling, call with B=1 before topology-aware batching.
        """
        if not self.config.enabled:
            return window, "original"

        if not window.current_calibration or not window.streams or set(window.streams) != set(window.metadata):
            raise ValueError("Topology augmentation requires valid streams, metadata, and calibration states.")

        device = next(iter(window.current_calibration.values())).transform.device
        if generator is not None and generator.device.type != device.type:
            raise ValueError("Topology RNG device must match the batch device.")

        self._validate_kaist_sources(window)
        if torch.rand((), device=device, generator=generator).item() >= self.config.replacement_probability:
            return window, "original"

        lidar_keys = self.config.lidar_keys
        replace_index = 0 if torch.rand((), device=device, generator=generator).item() < self.config.replace_first_lidar_probability else 1
        removed_lidar = lidar_keys[replace_index]
        return self._replace_lidar(window, removed_lidar, generator=generator), f"replace_{removed_lidar}"

    def _validate_kaist_sources(self, window: WindowBatch) -> None:
        config = self.config
        expected_keys = {config.source_imu_key, *config.lidar_keys}
        if set(window.current_calibration) != expected_keys:
            raise ValueError(f"Expected genuine KAIST calibration keys {expected_keys}, got {set(window.current_calibration)}.")

        expected_types = {
            MeasurementType.IMU_GYROSCOPE: 1,
            MeasurementType.IMU_ACCELEROMETER: 1,
        }
        imu_streams = self._stream_names_for_key(window, config.source_imu_key)
        for measurement_type, count in expected_types.items():
            matching = [name for name in imu_streams if window.metadata[name].measurement_type == measurement_type and window.metadata[name].geometry_type == GeometryType.VECTOR]
            if len(matching) != count:
                raise ValueError(f"Expected one {measurement_type} VECTOR stream for {config.source_imu_key}, found {matching}.")

        for lidar_key in config.lidar_keys:
            streams = self._stream_names_for_key(window, lidar_key)
            if len(streams) != 1 or window.metadata[streams[0]].geometry_type != GeometryType.SE3 or window.metadata[streams[0]].measurement_type != MeasurementType.LIDAR_POSE:
                raise ValueError(f"Expected one real SE3 LiDAR stream for {lidar_key!r}, got {streams}.")

        for name in (config.synthetic_gyro_stream, config.synthetic_accel_stream):
            if name in window.streams:
                raise ValueError(f"Synthetic stream name {name!r} collides with a real stream.")

    @staticmethod
    def _stream_names_for_key(window: WindowBatch, calibration_key: str) -> list[str]:
        return [name for name, metadata in window.metadata.items() if metadata.calibration_key == calibration_key]

    def _replace_lidar(self, window: WindowBatch, removed_lidar: str, generator: torch.Generator | None) -> WindowBatch:
        source_key = self.config.source_imu_key
        source_state = window.current_calibration[source_key]
        T_BI = source_state.transform
        if T_BI.ndim != 3 or T_BI.shape[-2:] != (4, 4):
            raise ValueError("IMU calibration transform must have shape [B,4,4].")
        batch_size, device, dtype = T_BI.shape[0], T_BI.device, T_BI.dtype

        # One independently sampled synthetic mounting orientation per batch item.
        axes = torch.randn((batch_size, 3), device=device, dtype=dtype, generator=generator)
        fallback = torch.zeros_like(axes)
        fallback[:, 0] = 1.0
        axes = torch.where(torch.linalg.vector_norm(axes, dim=-1, keepdim=True) > 1e-12, axes / torch.linalg.vector_norm(axes, dim=-1, keepdim=True).clamp_min(1e-12), fallback)
        radians = torch.rand((batch_size, 1), device=device, dtype=dtype, generator=generator) * (self.config.maximum_mount_rotation_deg * torch.pi / 180.0)
        Q = so3_exp(axes * radians)

        A = torch.eye(4, device=device, dtype=dtype).expand(batch_size, 4, 4).clone()
        A[:, :3, :3] = Q
        synthetic_state = CalibrationState(transform=T_BI @ A, time_offset=source_state.time_offset.clone())
        synthetic_state.validate()

        imu_stream_names = self._stream_names_for_key(window, source_key)
        source_gyro = next(name for name in imu_stream_names if window.metadata[name].measurement_type == MeasurementType.IMU_GYROSCOPE)
        source_accel = next(name for name in imu_stream_names if window.metadata[name].measurement_type == MeasurementType.IMU_ACCELEROMETER)
        removed_stream_name = self._stream_names_for_key(window, removed_lidar)[0]
        calibration = {}
        for key, state in window.current_calibration.items():
            calibration[self.config.synthetic_calibration_key if key == removed_lidar else key] = synthetic_state if key == removed_lidar else state

        # Preserve the other LiDAR and native IMU streams, and drop only the selected real LiDAR.
        streams = {name: stream for name, stream in window.streams.items() if name != removed_stream_name}
        metadata = {name: item for name, item in window.metadata.items() if name != removed_stream_name}

        for source_name, output_name, noise_std in (
            (source_gyro, self.config.synthetic_gyro_stream, self.config.synthetic_gyro_noise_std),
            (source_accel, self.config.synthetic_accel_stream, self.config.synthetic_accel_noise_std),
        ):
            stream = window.streams[source_name]
            # Row-vector form: v_I2 = Q^T v_I1 -> stored values_I2 = values_I1 @ Q.
            values = stream.values @ Q
            if noise_std > 0.0:
                noise = torch.randn(values.shape, device=values.device, dtype=values.dtype, generator=generator) * noise_std
                values = values + noise * stream.sample_mask.unsqueeze(-1)
            streams[output_name] = replace(stream, values=values)
            metadata[output_name] = replace(window.metadata[source_name], calibration_key=self.config.synthetic_calibration_key)

        return WindowBatch(streams=streams, current_calibration=calibration, metadata=metadata, targets=None, dataset_indices=window.dataset_indices)
