"""Minibatch collation for variable-length temporal windows."""

from collections.abc import Sequence

import torch

from obscalib.calibration import CalibrationState
from obscalib.data.structures import (
    CalibrationTarget,
    CalibrationTargetBatch,
    SensorMetadata,
    SensorStream,
    SensorStreamBatch,
    WindowBatch,
    WindowSample,
)


def _require_same_keys(
    dictionaries: Sequence[dict[str, object]],
    name: str,
) -> tuple[str, ...]:
    """Require every sample in a minibatch to expose the same named entries."""

    reference_keys = tuple(dictionaries[0].keys())
    reference_set = set(reference_keys)

    for index, dictionary in enumerate(dictionaries[1:], start=1):
        if set(dictionary.keys()) != reference_set:
            raise ValueError(
                f"All windows in a minibatch must have the same {name}; "
                f"window 0 has {sorted(reference_set)}, while window {index} "
                f"has {sorted(dictionary.keys())}."
            )

    return reference_keys


def _collate_sensor_streams(streams: Sequence[SensorStream]) -> SensorStreamBatch:
    """Pad one sensor independently to the longest stream in this minibatch."""

    if not streams:
        raise ValueError("Cannot collate an empty sensor-stream sequence.")

    for stream in streams:
        stream.validate()

    for window_index, stream in enumerate(streams):
        num_samples = int(
            stream.timestamps.numel()
        )

        if num_samples < 2:
            raise ValueError(
                "Every required sensor stream in a minibatch must contain at least "
                f"two valid measurements, but window {window_index} contains "
                f"{num_samples}."
            )

    sample_shape = streams[0].values.shape[1:]
    values_dtype = streams[0].values.dtype
    timestamp_dtype = streams[0].timestamps.dtype
    device = streams[0].values.device
    has_interval_starts = streams[0].interval_start_timestamps is not None

    for stream in streams[1:]:
        if stream.values.shape[1:] != sample_shape:
            raise ValueError("The same sensor stream must have one measurement shape throughout a minibatch.")

        if stream.values.dtype != values_dtype:
            raise ValueError("The same sensor stream must have one values dtype throughout a minibatch.")

        if stream.timestamps.dtype != timestamp_dtype:
            raise ValueError("The same sensor stream must have one timestamp dtype throughout a minibatch.")

        if stream.values.device != device or stream.timestamps.device != device:
            raise ValueError("The same sensor stream must be on one device throughout a minibatch.")

        if (stream.interval_start_timestamps is not None) != has_interval_starts:
            raise ValueError("interval_start_timestamps must be present for either all or none of the samples of one sensor stream in a minibatch.")

    batch_size = len(streams)
    max_samples = max(stream.values.shape[0] for stream in streams)

    # Preserve the complete per-measurement representation, including matrix-valued SO3/SE3 measurements.
    values = torch.zeros((batch_size, max_samples, *sample_shape), dtype=values_dtype, device=device)
    timestamps = torch.zeros((batch_size, max_samples), dtype=timestamp_dtype, device=device)
    sample_mask = torch.zeros((batch_size, max_samples), dtype=torch.bool, device=device)
    interval_start_timestamps = torch.zeros((batch_size, max_samples), dtype=timestamp_dtype, device=device) if has_interval_starts else None

    for batch_index, stream in enumerate(streams):
        num_samples = stream.values.shape[0]

        values[batch_index, :num_samples] = stream.values
        timestamps[batch_index, :num_samples] = stream.timestamps
        sample_mask[batch_index, :num_samples] = True

        if interval_start_timestamps is not None:
            interval_start_timestamps[batch_index, :num_samples] = stream.interval_start_timestamps

    return SensorStreamBatch(values=values, timestamps=timestamps, sample_mask=sample_mask, interval_start_timestamps=interval_start_timestamps)


def _collate_calibration_states(
    states: Sequence[CalibrationState],
) -> CalibrationState:
    """
    Concatenate singleton-batch calibration states into one minibatch.

    WindowSample currently stores the canonical batched CalibrationState with
    B=1 so there remains only one calibration-state representation in the
    package.
    """

    transforms: list[torch.Tensor] = []
    time_offsets: list[torch.Tensor] = []

    for state in states:
        state.validate()

        if state.transform.shape[0] != 1:
            raise ValueError(
                "CalibrationState inside WindowSample must have batch size 1."
            )

        transforms.append(state.transform)
        time_offsets.append(state.time_offset)

    return CalibrationState(
        transform=torch.cat(transforms, dim=0),
        time_offset=torch.cat(time_offsets, dim=0),
    )


def _stack_optional_target_field(
    targets: Sequence[CalibrationTarget],
    field_name: str,
) -> torch.Tensor | None:
    """Stack one optional target field while rejecting mixed availability."""

    values = [
        getattr(target, field_name)
        for target in targets
    ]

    if all(value is None for value in values):
        return None

    if any(value is None for value in values):
        raise ValueError(
            f"Target field {field_name!r} must be present for either all "
            "or none of the windows in one minibatch."
        )

    return torch.stack(values, dim=0)


def _collate_targets(
    targets: Sequence[CalibrationTarget],
) -> CalibrationTargetBatch:
    """Stack one calibration target from every sample."""

    return CalibrationTargetBatch(
        next_transform=_stack_optional_target_field(
            targets,
            "next_transform",
        ),
        next_time_offset=_stack_optional_target_field(
            targets,
            "next_time_offset",
        ),
        change_label=_stack_optional_target_field(
            targets,
            "change_label",
        ),
        change_time=_stack_optional_target_field(
            targets,
            "change_time",
        ),
    )


def _validate_metadata(
    metadata: Sequence[dict[str, SensorMetadata]],
    stream_keys: tuple[str, ...],
) -> dict[str, SensorMetadata]:
    """Require one stable sensor definition throughout a minibatch."""

    _require_same_keys(
        metadata,
        "metadata keys",
    )

    reference = metadata[0]

    for sample_index, sample_metadata in enumerate(
        metadata[1:],
        start=1,
    ):
        for stream_key in stream_keys:
            if sample_metadata[stream_key] != reference[stream_key]:
                raise ValueError(
                    f"Metadata for stream {stream_key!r} differs between "
                    f"window 0 and window {sample_index}."
                )

    return dict(reference)


def collate_windows(
    windows: Sequence[WindowSample],
) -> WindowBatch:
    """
    Collate variable-length windows using independent per-sensor padding.

    All windows in a minibatch must contain the same sensor set. Dataset
    preparation should discard windows with missing required sensors before
    they reach this function.
    """

    if not windows:
        raise ValueError("Cannot collate an empty sequence of windows.")

    stream_keys = _require_same_keys(
        [window.streams for window in windows],
        "sensor streams",
    )
    calibration_keys = _require_same_keys(
        [window.current_calibration for window in windows],
        "calibration states",
    )

    # Pad every sensor independently because their sample rates may differ by
    # orders of magnitude.
    streams = {
        stream_key: _collate_sensor_streams(
            [
                window.streams[stream_key]
                for window in windows
            ]
        )
        for stream_key in stream_keys
    }

    # Current calibration is teacher-forced during normal training, but the
    # batching representation is identical for GT and predicted states.
    current_calibration = {
        calibration_key: _collate_calibration_states(
            [
                window.current_calibration[calibration_key]
                for window in windows
            ]
        )
        for calibration_key in calibration_keys
    }

    metadata = _validate_metadata(
        [window.metadata for window in windows],
        stream_keys,
    )

    target_presence = [
        window.targets is not None
        for window in windows
    ]

    if any(target_presence) and not all(target_presence):
        raise ValueError(
            "Targets must be present for either all or none of the windows "
            "in one minibatch."
        )

    targets = None

    if all(target_presence):
        target_dicts = [
            window.targets
            for window in windows
            if window.targets is not None
        ]

        target_keys = _require_same_keys(
            target_dicts,
            "target keys",
        )

        targets = {
            target_key: _collate_targets(
                [
                    target_dict[target_key]
                    for target_dict in target_dicts
                ]
            )
            for target_key in target_keys
        }

    return WindowBatch(
        streams=streams,
        current_calibration=current_calibration,
        metadata=metadata,
        targets=targets,
    )