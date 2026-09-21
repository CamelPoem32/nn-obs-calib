"""Device-transfer helpers for obscalib data structures."""

from __future__ import annotations

import torch

from obscalib.calibration.state import CalibrationState
from obscalib.data.structures import CalibrationTargetBatch, SensorStreamBatch, WindowBatch


def window_batch_to(window: WindowBatch, device: torch.device | str, *, dtype: torch.dtype | None = None, non_blocking: bool = False) -> WindowBatch:
    """
    Move all tensor-valued fields of a WindowBatch to one device.

    If dtype is provided, floating-point tensors are cast to that dtype while
    boolean/integer tensors retain their original dtype.
    """

    streams = {
        stream_key: SensorStreamBatch(
            values=_tensor_to(stream.values, device=device, dtype=dtype, non_blocking=non_blocking),
            timestamps=_tensor_to(stream.timestamps, device=device, dtype=dtype, non_blocking=non_blocking),
            sample_mask=stream.sample_mask.to(device=device, non_blocking=non_blocking),
            interval_start_timestamps=None if stream.interval_start_timestamps is None else _tensor_to(stream.interval_start_timestamps, device=device, dtype=dtype, non_blocking=non_blocking),
        )
        for stream_key, stream in window.streams.items()
    }

    current_calibration = {
        calibration_key: CalibrationState(
            transform=_tensor_to(state.transform, device=device, dtype=dtype, non_blocking=non_blocking),
            time_offset=_tensor_to(state.time_offset, device=device, dtype=dtype, non_blocking=non_blocking),
        )
        for calibration_key, state in window.current_calibration.items()
    }

    targets = None if window.targets is None else {
        calibration_key: _calibration_target_batch_to(target, device=device, dtype=dtype, non_blocking=non_blocking)
        for calibration_key, target in window.targets.items()
    }

    return WindowBatch(
        streams=streams,
        current_calibration=current_calibration,
        metadata=dict(window.metadata),
        targets=targets,
    )


def _calibration_target_batch_to(target: CalibrationTargetBatch, *, device: torch.device | str, dtype: torch.dtype | None, non_blocking: bool) -> CalibrationTargetBatch:
    """Move one calibration supervision batch to a device."""

    return CalibrationTargetBatch(
        next_transform=None if target.next_transform is None else _tensor_to(target.next_transform, device=device, dtype=dtype, non_blocking=non_blocking),
        next_time_offset=None if target.next_time_offset is None else _tensor_to(target.next_time_offset, device=device, dtype=dtype, non_blocking=non_blocking),
        change_label=None if target.change_label is None else _tensor_to(target.change_label, device=device, dtype=dtype, non_blocking=non_blocking),
        change_time=None if target.change_time is None else _tensor_to(target.change_time, device=device, dtype=dtype, non_blocking=non_blocking),
    )


def _tensor_to(tensor: torch.Tensor, *, device: torch.device | str, dtype: torch.dtype | None, non_blocking: bool) -> torch.Tensor:
    """Move one tensor while casting only floating-point data."""

    target_dtype = dtype if dtype is not None and torch.is_floating_point(tensor) else tensor.dtype

    return tensor.to(device=device, dtype=target_dtype, non_blocking=non_blocking)