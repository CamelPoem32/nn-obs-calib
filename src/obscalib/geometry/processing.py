'''Geometry processing, trajectory interpolation, and calibration-prior transformation.'''

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from obscalib.calibration.state import CalibrationState
from obscalib.data.structures import CanonicalSensorStreamBatch, GeometryType, SensorMetadata, SensorStreamBatch
from obscalib.geometry.lie import interpolate_se3, se3_adjoint, se3_inverse, se3_log
from obscalib.geometry.maps import SE3LogMap, SO3LogMap, VectorObservationMap


def _inverse_se3(transform: torch.Tensor) -> torch.Tensor:
    '''Invert batched rigid transforms with shape [B, 4, 4].

    This private wrapper is retained for drop-in compatibility. New code should
    use ``obscalib.geometry.lie.se3_inverse`` directly.
    '''

    if transform.ndim != 3 or transform.shape[-2:] != (4, 4):
        raise ValueError("transform must have shape [B, 4, 4].")

    return se3_inverse(transform)


def _validate_single_trajectory(
    timestamps: torch.Tensor,
    transforms: torch.Tensor,
) -> None:
    '''Validate one unbatched SE(3) trajectory used for interpolation.'''

    if timestamps.ndim != 1:
        raise ValueError("timestamps must have shape [N].")

    if transforms.ndim != 3 or transforms.shape[-2:] != (4, 4):
        raise ValueError("transforms must have shape [N, 4, 4].")

    if timestamps.shape[0] != transforms.shape[0]:
        raise ValueError(
            "timestamps and transforms must contain the same number of trajectory states."
        )

    if timestamps.numel() < 2:
        raise ValueError(
            "At least two trajectory states are required for interpolation."
        )

    if timestamps.device != transforms.device:
        raise ValueError(
            "timestamps and transforms must be on the same device."
        )

    if timestamps.dtype != transforms.dtype:
        raise ValueError(
            "timestamps and transforms must have the same dtype."
        )

    if not torch.is_floating_point(timestamps) or not torch.is_floating_point(transforms):
        raise TypeError(
            "timestamps and transforms must use floating-point dtypes."
        )

    if not torch.isfinite(timestamps).all() or not torch.isfinite(transforms).all():
        raise ValueError(
            "timestamps and transforms must contain only finite values."
        )

    if torch.any(
        timestamps[1:]
        <= timestamps[:-1]
    ):
        raise ValueError(
            "timestamps must be strictly increasing."
        )


def _trajectory_interpolation_indices(
    timestamps: torch.Tensor,
    query_timestamps: torch.Tensor,
    *,
    allow_extrapolation: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''Find bracketing trajectory states and interpolation fractions.'''

    if query_timestamps.ndim != 1:
        raise ValueError(
            "query_timestamps must have shape [M]."
        )

    if query_timestamps.device != timestamps.device:
        raise ValueError(
            "query_timestamps and timestamps must be on the same device."
        )

    if query_timestamps.dtype != timestamps.dtype:
        raise ValueError(
            "query_timestamps and timestamps must have the same dtype."
        )

    if not torch.is_floating_point(query_timestamps):
        raise TypeError(
            "query_timestamps must use a floating-point dtype."
        )

    if not torch.isfinite(query_timestamps).all():
        raise ValueError(
            "query_timestamps must contain only finite values."
        )

    if (
        not allow_extrapolation
        and query_timestamps.numel() > 0
    ):
        if (
            torch.any(
                query_timestamps < timestamps[0]
            )
            or torch.any(
                query_timestamps > timestamps[-1]
            )
        ):
            raise ValueError(
                "query_timestamps must lie inside the trajectory interval unless extrapolation is enabled."
            )

    # searchsorted(..., right=True) assigns a query exactly at the final knot
    # to the final interpolation interval. Clamping also selects the first or
    # last interval naturally when explicit extrapolation is requested.
    upper_indices = torch.searchsorted(
        timestamps,
        query_timestamps,
        right=True,
    )

    upper_indices = upper_indices.clamp(
        min=1,
        max=timestamps.numel() - 1,
    )

    lower_indices = upper_indices - 1

    interval_start = timestamps[
        lower_indices
    ]
    interval_end = timestamps[
        upper_indices
    ]

    alpha = (
        query_timestamps
        - interval_start
    ) / (
        interval_end
        - interval_start
    )

    return (
        lower_indices,
        upper_indices,
        alpha,
    )


def interpolate_se3_trajectory(
    timestamps: torch.Tensor,
    transforms: torch.Tensor,
    query_timestamps: torch.Tensor,
    *,
    allow_extrapolation: bool = False,
) -> torch.Tensor:
    '''Interpolate one SE(3) trajectory at arbitrary query timestamps.

    Parameters
    ----------
    timestamps:
        Strictly increasing trajectory timestamps with shape [N].
    transforms:
        Trajectory transforms with shape [N, 4, 4].
    query_timestamps:
        Requested timestamps with shape [M].
    allow_extrapolation:
        Whether queries outside the trajectory interval may use the first or
        last trajectory segment for geodesic extrapolation.

    Returns
    -------
    torch.Tensor
        Interpolated transforms with shape [M, 4, 4].
    '''

    _validate_single_trajectory(
        timestamps,
        transforms,
    )

    (
        lower_indices,
        upper_indices,
        alpha,
    ) = _trajectory_interpolation_indices(
        timestamps,
        query_timestamps,
        allow_extrapolation=allow_extrapolation,
    )

    transform_start = transforms[
        lower_indices
    ]
    transform_end = transforms[
        upper_indices
    ]

    return interpolate_se3(
        transform_start,
        transform_end,
        alpha,
        allow_extrapolation=allow_extrapolation,
    )


def interpolate_se3_trajectory_with_twist(
    timestamps: torch.Tensor,
    transforms: torch.Tensor,
    query_timestamps: torch.Tensor,
    *,
    allow_extrapolation: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    '''Interpolate one SE(3) trajectory and return body and spatial twists.

    Piecewise interpolation follows

        T(t) = T_k @ Exp(alpha * xi_k),

        xi_k = Log(inv(T_k) @ T_{k+1}).

    The body twist is constant inside each interpolation segment. The spatial
    twist is obtained with the SE(3) group adjoint at the interpolated pose.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(interpolated_transforms, body_twists, spatial_twists)`` with shapes
        [M, 4, 4], [M, 6], and [M, 6].
    '''

    _validate_single_trajectory(
        timestamps,
        transforms,
    )

    (
        lower_indices,
        upper_indices,
        alpha,
    ) = _trajectory_interpolation_indices(
        timestamps,
        query_timestamps,
        allow_extrapolation=allow_extrapolation,
    )

    transform_start = transforms[
        lower_indices
    ]
    transform_end = transforms[
        upper_indices
    ]

    interval_duration = (
        timestamps[upper_indices]
        - timestamps[lower_indices]
    )

    relative_transform = (
        se3_inverse(
            transform_start
        )
        @ transform_end
    )

    relative_tangent = se3_log(
        relative_transform
    )

    interpolated_transforms = interpolate_se3(
        transform_start,
        transform_end,
        alpha,
        allow_extrapolation=allow_extrapolation,
    )

    body_twists = (
        relative_tangent
        / interval_duration[..., None]
    )

    spatial_twists = (
        se3_adjoint(
            interpolated_transforms
        )
        @ body_twists.unsqueeze(-1)
    ).squeeze(-1)

    return (
        interpolated_transforms,
        body_twists,
        spatial_twists,
    )


def transform_raw_measurements_to_world(
    measurements: torch.Tensor,
    transform_sensor_in_world: torch.Tensor,
    geometry_type: GeometryType,
) -> torch.Tensor:
    '''
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
    '''

    if transform_sensor_in_world.ndim != 3 or transform_sensor_in_world.shape[-2:] != (4, 4):
        raise ValueError(
            "transform_sensor_in_world must have shape [B, 4, 4]."
        )

    if measurements.shape[0] != transform_sensor_in_world.shape[0]:
        raise ValueError(
            "measurements and transform_sensor_in_world must share batch size."
        )

    if measurements.device != transform_sensor_in_world.device:
        raise ValueError(
            "measurements and transform_sensor_in_world must be on the same device."
        )

    if measurements.dtype != transform_sensor_in_world.dtype:
        raise ValueError(
            "measurements and transform_sensor_in_world must have the same dtype."
        )

    rotation_sensor_in_world = transform_sensor_in_world[
        ...,
        :3,
        :3,
    ]

    if geometry_type == GeometryType.VECTOR:
        if measurements.ndim != 3 or measurements.shape[-1] != 3:
            raise ValueError(
                "Vector measurements must have shape [B, N, 3]."
            )

        # Row-vector storage of N measurements:
        #
        #     [B, N, 3] @ [B, 3, 3]^T -> [B, N, 3]
        #
        # equivalent to applying R_WS @ v_S to each individual column vector.
        return (
            measurements
            @ rotation_sensor_in_world.transpose(
                -1,
                -2,
            )
        )

    if geometry_type == GeometryType.SO3:
        if measurements.ndim != 4 or measurements.shape[-2:] != (3, 3):
            raise ValueError(
                "SO(3) measurements must have shape [B, N, 3, 3]."
            )

        rotation = rotation_sensor_in_world[
            :,
            None,
            :,
            :,
        ]

        return (
            rotation
            @ measurements
            @ rotation.transpose(
                -1,
                -2,
            )
        )

    if geometry_type == GeometryType.SE3:
        if measurements.ndim != 4 or measurements.shape[-2:] != (4, 4):
            raise ValueError(
                "SE(3) measurements must have shape [B, N, 4, 4]."
            )

        transform = transform_sensor_in_world[
            :,
            None,
            :,
            :,
        ]

        transform_inverse = se3_inverse(
            transform_sensor_in_world
        )[
            :,
            None,
            :,
            :,
        ]

        return (
            transform
            @ measurements
            @ transform_inverse
        )

    raise ValueError(
        f"Unsupported geometry type: {geometry_type!r}."
    )


class GeometryProcessor(nn.Module):
    '''
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
    '''

    def __init__(self) -> None:
        super().__init__()

        self.geometry_maps = nn.ModuleDict(
            {
                GeometryType.VECTOR.value: VectorObservationMap(),
                GeometryType.SO3.value: SO3LogMap(),
                GeometryType.SE3.value: SE3LogMap(),
            }
        )

    def forward(
        self,
        streams: Mapping[
            str,
            SensorStreamBatch,
        ],
        metadata: Mapping[
            str,
            SensorMetadata,
        ],
        calibration: Mapping[
            str,
            CalibrationState,
        ],
    ) -> dict[
        str,
        CanonicalSensorStreamBatch,
    ]:
        '''Transform and vectorize every raw stream using its current calibration prior.'''

        if set(streams.keys()) != set(metadata.keys()):
            raise ValueError(
                "streams and metadata must contain the same stream keys."
            )

        result: dict[
            str,
            CanonicalSensorStreamBatch,
        ] = {}

        for stream_name, stream in streams.items():
            stream.validate()

            stream_metadata = metadata[
                stream_name
            ]

            if stream_metadata.calibration_key not in calibration:
                raise ValueError(
                    f"Stream {stream_name!r} requires missing calibration key {stream_metadata.calibration_key!r}."
                )

            calibration_state = calibration[
                stream_metadata.calibration_key
            ]
            calibration_state.validate()

            if stream.values.shape[0] != calibration_state.transform.shape[0]:
                raise ValueError(
                    f"Stream {stream_name!r} and calibration {stream_metadata.calibration_key!r} must share batch size."
                )

            if stream.values.device != calibration_state.transform.device or stream.timestamps.device != calibration_state.time_offset.device:
                raise ValueError(
                    f"Stream {stream_name!r} and its calibration state must be on the same device."
                )

            if stream.values.dtype != calibration_state.transform.dtype or stream.timestamps.dtype != calibration_state.time_offset.dtype:
                raise ValueError(
                    f"Stream {stream_name!r} and its calibration state must use matching floating-point dtypes."
                )

            # Spatial calibration is applied to the raw group-valued measurement
            # before SO(3)/SE(3) logarithmic vectorization.
            world_measurements = transform_raw_measurements_to_world(
                stream.values,
                calibration_state.transform,
                stream_metadata.geometry_type,
            )

            canonical_values = self.geometry_maps[
                stream_metadata.geometry_type.value
            ](
                world_measurements
            )

            if canonical_values.ndim != 3:
                raise ValueError(
                    f"Geometry processing for stream {stream_name!r} must produce [B, N, D] vectors."
                )

            # Temporal calibration convention:
            #
            #     t_corrected = t_measured + tau.
            #
            # [B, N] + [B, 1] broadcasts independently for every minibatch item.
            corrected_timestamps = (
                stream.timestamps
                + calibration_state.time_offset
            )

            result[
                stream_name
            ] = CanonicalSensorStreamBatch(
                values=canonical_values,
                timestamps=corrected_timestamps,
                sample_mask=stream.sample_mask,
                measurement_type=stream_metadata.measurement_type,
            )

        return result