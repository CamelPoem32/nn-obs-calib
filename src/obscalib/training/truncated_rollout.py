"""Helpers for detached sequential calibration training."""

from __future__ import annotations

import torch

from obscalib.calibration.state import CalibrationState
from obscalib.data.structures import GeometryType, SensorStreamBatch, WindowBatch


def detach_calibration(calibration: dict[str, CalibrationState]) -> dict[str, CalibrationState]:
    """Detach a carried calibration state from all previous computational graphs."""

    return {
        calibration_key: CalibrationState(
            transform=state.transform.detach(),
            time_offset=state.time_offset.detach(),
        )
        for calibration_key, state in calibration.items()
    }


def apply_fixed_frame_randomization(window: WindowBatch, frame_randomization_by_key: dict[str, torch.Tensor]) -> WindowBatch:
    """
    Apply one already-sampled sensor-frame randomization to a new temporal window.

    The same randomization can therefore be kept fixed across all windows of a
    truncated rollout sequence.
    """

    randomized_streams: dict[str, SensorStreamBatch] = {}

    for stream_key, stream in window.streams.items():
        metadata = window.metadata[stream_key]
        calibration_key = metadata.calibration_key

        if calibration_key not in frame_randomization_by_key:
            raise KeyError(f"Missing frame randomization for calibration key {calibration_key!r}.")

        A = frame_randomization_by_key[calibration_key]
        R_A = A[:, :3, :3]

        if metadata.geometry_type == GeometryType.VECTOR:
            randomized_values = torch.matmul(
                stream.values,
                R_A,
            )

        elif metadata.geometry_type == GeometryType.SO3:
            randomized_values = (
                R_A.transpose(-1, -2).unsqueeze(1)
                @ stream.values
                @ R_A.unsqueeze(1)
            )

        elif metadata.geometry_type == GeometryType.SE3:
            A_inv = torch.linalg.inv(
                A
            )

            randomized_values = (
                A_inv.unsqueeze(1)
                @ stream.values
                @ A.unsqueeze(1)
            )

        else:
            raise ValueError(f"Unsupported geometry type {metadata.geometry_type!r}.")

        randomized_streams[stream_key] = SensorStreamBatch(
            values=randomized_values,
            timestamps=stream.timestamps,
            sample_mask=stream.sample_mask,
            interval_start_timestamps=stream.interval_start_timestamps,
        )

    randomized_calibration = {
        calibration_key: CalibrationState(
            transform=(
                state.transform
                @ frame_randomization_by_key[calibration_key]
            ),
            time_offset=state.time_offset,
        )
        for calibration_key, state in window.current_calibration.items()
    }

    return WindowBatch(
        streams=randomized_streams,
        current_calibration=randomized_calibration,
        metadata=dict(window.metadata),
        targets=None,
    )