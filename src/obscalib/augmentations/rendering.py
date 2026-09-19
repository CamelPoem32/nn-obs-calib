"""Render synthetic calibration trajectories into raw sensor streams."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from obscalib.augmentations.structures import CalibrationTrajectory
from obscalib.augmentations.trajectory import evaluate_calibration_trajectory
from obscalib.data.structures import GeometryType, SensorMetadata, SensorStreamBatch, WindowBatch


class CalibrationEventRenderer:
    """
    Render true calibration-change trajectories into raw sensor measurements.

    The input raw window is assumed to be physically consistent with each
    calibration trajectory's pre-event state.

    A time-varying calibration trajectory

        T_WS(t)

    is converted into the corresponding change of sensor coordinates

        A(t) = T_WS_pre^-1 @ T_WS(t),

    so that

        T_WS(t) = T_WS_pre @ A(t).

    Raw measurements are then re-expressed in the changing sensor frame.

    VECTOR:
        v_new(t) = R_A(t)^T v_old(t)

    With row-vector tensor storage [B, N, 3], this becomes

        v_new = v_old @ R_A(t).

    Relative SO3:
        delta_R_new(i, j)
            = R_A(t_i)^T @ delta_R_old(i, j) @ R_A(t_j)

    Relative SE3:
        delta_T_new(i, j)
            = A(t_i)^-1 @ delta_T_old(i, j) @ A(t_j)

    Temporal calibration is rendered by shifting measured timestamps according
    to the change from the pre-event offset:

        t_measured_new
            = t_measured_old - (tau(t) - tau_pre).

    Therefore applying the true time-varying offset recovers the same underlying
    time reference:

        t_measured_new + tau(t)
            = t_measured_old + tau_pre.

    For relative SO3 and SE3 streams, each measurement timestamp is interpreted
    as the end time of the relative update. Its start time is the previous valid
    timestamp in that stream. The first valid relative measurement starts at
    window-relative time zero.

    This first version intentionally models only coordinate-frame
    re-expression for VECTOR streams. Additional IMU effects caused by a
    physically moving sensor frame, such as angular-velocity and lever-arm
    terms, are outside the current augmentation model.
    """

    def __call__(
        self,
        window: WindowBatch,
        calibration_truth: Mapping[str, CalibrationTrajectory],
    ) -> WindowBatch:
        """Render all calibration trajectories into their associated streams."""

        _validate_window_and_truth(
            window=window,
            calibration_truth=calibration_truth,
        )

        rendered_streams: dict[str, SensorStreamBatch] = {}

        for stream_key, stream in window.streams.items():
            metadata = window.metadata[stream_key]
            trajectory = calibration_truth[metadata.calibration_key]

            rendered_streams[stream_key] = _render_stream(
                stream=stream,
                metadata=metadata,
                trajectory=trajectory,
            )

        return WindowBatch(
            streams=rendered_streams,
            current_calibration=dict(window.current_calibration),
            metadata=dict(window.metadata),
            targets=None if window.targets is None else dict(window.targets),
        )


def _render_stream(stream: SensorStreamBatch, metadata: SensorMetadata, trajectory: CalibrationTrajectory) -> SensorStreamBatch:
    """Render one calibration trajectory into one raw sensor stream."""

    stream.validate()
    trajectory.validate()

    end_transforms, end_time_offsets = evaluate_calibration_trajectory(pre_event=trajectory.pre_event, event=trajectory.event, timestamps_s=stream.timestamps)

    pre_event_inverse = _inverse_se3(trajectory.pre_event.transform)
    A_end = pre_event_inverse[:, None, :, :] @ end_transforms

    rendered_end_timestamps = _render_timestamps(stream.timestamps, end_time_offsets, trajectory.pre_event.time_offset, stream.sample_mask)

    if metadata.geometry_type == GeometryType.VECTOR:
        if stream.interval_start_timestamps is not None:
            raise ValueError("VECTOR streams must not contain interval_start_timestamps.")

        rendered_values = _render_vector_measurements(stream.values, A_end, stream.sample_mask)

        return SensorStreamBatch(values=rendered_values, timestamps=rendered_end_timestamps, sample_mask=stream.sample_mask, interval_start_timestamps=None)

    if stream.interval_start_timestamps is None:
        raise ValueError(f"{metadata.geometry_type.value.upper()} relative measurements require interval_start_timestamps.")

    start_transforms, start_time_offsets = evaluate_calibration_trajectory(pre_event=trajectory.pre_event, event=trajectory.event, timestamps_s=stream.interval_start_timestamps)
    A_start = pre_event_inverse[:, None, :, :] @ start_transforms

    rendered_start_timestamps = _render_timestamps(stream.interval_start_timestamps, start_time_offsets, trajectory.pre_event.time_offset, stream.sample_mask)

    if metadata.geometry_type == GeometryType.SO3:
        rendered_values = _render_relative_so3_measurements(stream.values, A_start, A_end, stream.sample_mask)

    elif metadata.geometry_type == GeometryType.SE3:
        rendered_values = _render_relative_se3_measurements(stream.values, A_start, A_end, stream.sample_mask)

    else:
        raise ValueError(f"Unsupported geometry type: {metadata.geometry_type!r}")

    return SensorStreamBatch(values=rendered_values, timestamps=rendered_end_timestamps, sample_mask=stream.sample_mask, interval_start_timestamps=rendered_start_timestamps)


def _render_vector_measurements(values: torch.Tensor, A_end: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
    """Re-express vector measurements in the time-varying sensor frame."""

    R_end = A_end[..., :3, :3]
    rendered = (values.unsqueeze(-2) @ R_end).squeeze(-2)

    mask = sample_mask[..., None]
    return torch.where(mask, rendered, values)


def _render_relative_so3_measurements(values: torch.Tensor, A_start: torch.Tensor, A_end: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
    """Render relative SO3 measurements using their explicit start and end frames."""

    R_start = A_start[..., :3, :3]
    R_end = A_end[..., :3, :3]
    rendered = R_start.transpose(-1, -2) @ values @ R_end

    mask = sample_mask[..., None, None]
    return torch.where(mask, rendered, values)


def _render_relative_se3_measurements(values: torch.Tensor, A_start: torch.Tensor, A_end: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
    """Render relative SE3 measurements using their explicit start and end frames."""

    rendered = _inverse_se3(A_start) @ values @ A_end

    mask = sample_mask[..., None, None]
    return torch.where(mask, rendered, values)


def _render_timestamps(timestamps: torch.Tensor, time_offsets: torch.Tensor, pre_event_time_offset: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
    """Synthesize measured timestamps while preserving the corresponding true-time trajectory."""

    delta_tau = time_offsets - pre_event_time_offset[:, None, :]
    rendered = timestamps - delta_tau.squeeze(-1)

    return torch.where(sample_mask, rendered, timestamps)


def _inverse_se3(
    transform: torch.Tensor,
) -> torch.Tensor:
    """Invert SE3 matrices with arbitrary leading batch dimensions."""

    if transform.ndim < 2 or transform.shape[-2:] != (4, 4):
        raise ValueError(
            "transform must have shape [..., 4, 4]."
        )

    R = transform[..., :3, :3]
    t = transform[..., :3, 3]

    R_inverse = R.transpose(-1, -2)
    t_inverse = -(
        R_inverse @ t.unsqueeze(-1)
    ).squeeze(-1)

    inverse = torch.zeros_like(transform)

    inverse[..., :3, :3] = R_inverse
    inverse[..., :3, 3] = t_inverse
    inverse[..., 3, 3] = 1.0

    return inverse


def _validate_window_and_truth(
    window: WindowBatch,
    calibration_truth: Mapping[str, CalibrationTrajectory],
) -> None:
    """Validate compatibility between raw streams and calibration trajectories."""

    if not window.streams:
        raise ValueError(
            "Calibration-event rendering requires at least one sensor stream."
        )

    if set(window.streams) != set(window.metadata):
        raise ValueError(
            "Window streams and metadata must have identical keys."
        )

    if not calibration_truth:
        raise ValueError(
            "Calibration-event rendering requires calibration truth."
        )

    required_calibration_keys = {
        metadata.calibration_key
        for metadata in window.metadata.values()
    }

    missing_calibration_keys = (
        required_calibration_keys
        - set(calibration_truth)
    )

    if missing_calibration_keys:
        raise KeyError(
            "Missing calibration trajectories for stream calibration keys: "
            f"{sorted(missing_calibration_keys)}"
        )

    for calibration_key, trajectory in calibration_truth.items():
        try:
            trajectory.validate()
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid calibration trajectory {calibration_key!r}."
            ) from error

    for stream_key, stream in window.streams.items():
        stream.validate()

        metadata = window.metadata[stream_key]
        trajectory = calibration_truth[
            metadata.calibration_key
        ]

        batch_size = (
            trajectory.pre_event.transform.shape[0]
        )

        if stream.values.shape[0] != batch_size:
            raise ValueError(
                f"Stream {stream_key!r} and calibration trajectory "
                f"{metadata.calibration_key!r} must share batch size."
            )

        if (
            stream.values.device
            != trajectory.pre_event.transform.device
        ):
            raise ValueError(
                f"Stream {stream_key!r} and calibration trajectory "
                "must be on the same device."
            )

        if (
            stream.timestamps.device
            != trajectory.pre_event.transform.device
        ):
            raise ValueError(
                f"Stream {stream_key!r} timestamps and calibration trajectory "
                "must be on the same device."
            )

        if (
            stream.values.dtype
            != trajectory.pre_event.transform.dtype
        ):
            raise TypeError(
                f"Stream {stream_key!r} values and calibration trajectory "
                "must have the same dtype."
            )

        if (
            stream.timestamps.dtype
            != trajectory.pre_event.transform.dtype
        ):
            raise TypeError(
                f"Stream {stream_key!r} timestamps and calibration trajectory "
                "must have the same dtype."
            )

        if torch.any(
            stream.sample_mask
            & (stream.timestamps < 0.0)
        ):
            raise ValueError(
                f"Valid timestamps in stream {stream_key!r} must be "
                "nonnegative and relative to the window start."
            )