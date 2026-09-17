"""Calibration-prior transformation followed by geometry vectorization."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from obscalib.calibration import CalibrationState
from obscalib.data.structures import CanonicalSensorStreamBatch, GeometryType, SensorMetadata, SensorStreamBatch
from obscalib.geometry.maps import SE3LogMap, SO3LogMap, VectorObservationMap


def _inverse_se3(transform: torch.Tensor) -> torch.Tensor:
    """Invert batched rigid transforms with shape [B, 4, 4]."""

    if transform.ndim != 3 or transform.shape[-2:] != (4, 4):
        raise ValueError("transform must have shape [B, 4, 4].")

    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]

    rotation_inverse = rotation.transpose(-1, -2)
    translation_inverse = -(rotation_inverse @ translation.unsqueeze(-1)).squeeze(-1)

    inverse = torch.zeros_like(transform)
    inverse[..., :3, :3] = rotation_inverse
    inverse[..., :3, 3] = translation_inverse
    inverse[..., 3, 3] = 1.0

    return inverse


def transform_raw_measurements_to_world(measurements: torch.Tensor, transform_sensor_in_world: torch.Tensor, geometry_type: GeometryType) -> torch.Tensor:
    """
    Express raw measurements in the world frame before geometry vectorization.

    The calibration transform follows

        T_sensor_in_world = T_WS.

    VECTOR:
        v_W = R_WS v_S.

    SO3:
        delta_R_W = R_WS delta_R_S R_WS^T.

    SE3:
        delta_T_W = T_WS delta_T_S T_WS^-1.

    SO(3) and SE(3) observations remain matrices here. Log().vee() conversion
    deliberately happens only after the frame transformation.
    """

    if transform_sensor_in_world.ndim != 3 or transform_sensor_in_world.shape[-2:] != (4, 4):
        raise ValueError("transform_sensor_in_world must have shape [B, 4, 4].")

    if measurements.shape[0] != transform_sensor_in_world.shape[0]:
        raise ValueError("measurements and transform_sensor_in_world must share batch size.")

    if measurements.device != transform_sensor_in_world.device:
        raise ValueError("measurements and transform_sensor_in_world must be on the same device.")

    if measurements.dtype != transform_sensor_in_world.dtype:
        raise ValueError("measurements and transform_sensor_in_world must have the same dtype.")

    rotation_sensor_in_world = transform_sensor_in_world[..., :3, :3]

    if geometry_type == GeometryType.VECTOR:
        if measurements.ndim != 3 or measurements.shape[-1] != 3:
            raise ValueError("Vector measurements must have shape [B, N, 3].")

        # Row-vector storage of N measurements:
        #
        #     [B, N, 3] @ [B, 3, 3]^T -> [B, N, 3]
        #
        # equivalent to applying R_WS @ v_S to each individual column vector.
        return measurements @ rotation_sensor_in_world.transpose(-1, -2)

    if geometry_type == GeometryType.SO3:
        if measurements.ndim != 4 or measurements.shape[-2:] != (3, 3):
            raise ValueError("SO(3) measurements must have shape [B, N, 3, 3].")

        rotation = rotation_sensor_in_world[:, None, :, :]

        return rotation @ measurements @ rotation.transpose(-1, -2)

    if geometry_type == GeometryType.SE3:
        if measurements.ndim != 4 or measurements.shape[-2:] != (4, 4):
            raise ValueError("SE(3) measurements must have shape [B, N, 4, 4].")

        transform = transform_sensor_in_world[:, None, :, :]
        transform_inverse = _inverse_se3(transform_sensor_in_world)[:, None, :, :]

        return transform @ measurements @ transform_inverse

    raise ValueError(f"Unsupported geometry type: {geometry_type!r}.")


class GeometryProcessor(nn.Module):
    """
    Apply the current calibration priors and convert every measurement to a vector.

    Processing order:

        raw spatial measurement
            -> transform from sensor coordinates to world coordinates
            -> geometry-specific Log().vee() / vector conversion
            -> canonical vector

        raw relative timestamp
            -> add current calibration time offset
            -> corrected relative timestamp

    The resulting canonical streams are ready for zero-padding and token
    construction.
    """

    def __init__(self) -> None:
        super().__init__()

        self.geometry_maps = nn.ModuleDict(
            {
                GeometryType.VECTOR.value: VectorObservationMap(),
                GeometryType.SO3.value: SO3LogMap(),
                GeometryType.SE3.value: SE3LogMap(),
            }
        )

    def forward(self, streams: Mapping[str, SensorStreamBatch], metadata: Mapping[str, SensorMetadata], calibration: Mapping[str, CalibrationState]) -> dict[str, CanonicalSensorStreamBatch]:
        """Transform and vectorize every raw stream using its current calibration prior."""

        if set(streams.keys()) != set(metadata.keys()):
            raise ValueError("streams and metadata must contain the same stream keys.")

        result: dict[str, CanonicalSensorStreamBatch] = {}

        for stream_name, stream in streams.items():
            stream.validate()

            stream_metadata = metadata[stream_name]

            if stream_metadata.calibration_key not in calibration:
                raise ValueError(f"Stream {stream_name!r} requires missing calibration key {stream_metadata.calibration_key!r}.")

            calibration_state = calibration[stream_metadata.calibration_key]
            calibration_state.validate()

            if stream.values.shape[0] != calibration_state.transform.shape[0]:
                raise ValueError(f"Stream {stream_name!r} and calibration {stream_metadata.calibration_key!r} must share batch size.")

            if stream.values.device != calibration_state.transform.device or stream.timestamps.device != calibration_state.time_offset.device:
                raise ValueError(f"Stream {stream_name!r} and its calibration state must be on the same device.")

            if stream.values.dtype != calibration_state.transform.dtype or stream.timestamps.dtype != calibration_state.time_offset.dtype:
                raise ValueError(f"Stream {stream_name!r} and its calibration state must use matching floating-point dtypes.")

            # Spatial calibration is applied to the raw group-valued measurement
            # before SO(3)/SE(3) logarithmic vectorization.
            world_measurements = transform_raw_measurements_to_world(stream.values, calibration_state.transform, stream_metadata.geometry_type)
            canonical_values = self.geometry_maps[stream_metadata.geometry_type.value](world_measurements)

            if canonical_values.ndim != 3:
                raise ValueError(f"Geometry processing for stream {stream_name!r} must produce [B, N, D] vectors.")

            # Temporal calibration convention:
            #
            #     t_corrected = t_measured + tau.
            #
            # [B, N] + [B, 1] broadcasts independently for every minibatch item.
            corrected_timestamps = stream.timestamps + calibration_state.time_offset

            result[stream_name] = CanonicalSensorStreamBatch(values=canonical_values, timestamps=corrected_timestamps, sample_mask=stream.sample_mask, measurement_type=stream_metadata.measurement_type)

        return result