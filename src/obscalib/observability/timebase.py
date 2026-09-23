'''Reference timestamp selection for one-window observability.'''

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import torch

from obscalib.data.structures import MeasurementType, SensorMetadata, SensorStream


class ReferenceTimePolicy(str, Enum):
    '''Supported deterministic reference-stream policies.'''

    EXPLICIT_STREAM = "explicit_stream"
    LOWEST_RATE_LIDAR = "lowest_rate_lidar"


@dataclass(frozen=True)
class ReferenceTimeConfig:
    '''Configuration for choosing trajectory/reference timestamps.'''

    policy: ReferenceTimePolicy = ReferenceTimePolicy.LOWEST_RATE_LIDAR
    explicit_stream_key: str | None = None

    def __post_init__(self) -> None:
        if self.policy == ReferenceTimePolicy.EXPLICIT_STREAM:
            if not self.explicit_stream_key:
                raise ValueError(
                    "EXPLICIT_STREAM requires explicit_stream_key."
                )
        elif self.explicit_stream_key is not None:
            raise ValueError(
                "explicit_stream_key is only valid with EXPLICIT_STREAM."
            )


@dataclass(frozen=True)
class ReferenceTimebase:
    '''Selected single-window trajectory timestamps and their source stream.'''

    stream_key: str
    timestamps: torch.Tensor

    def validate(self) -> None:
        if not self.stream_key:
            raise ValueError(
                "stream_key must be non-empty."
            )

        if self.timestamps.ndim != 1:
            raise ValueError(
                "Reference timestamps must have shape [N]."
            )

        if self.timestamps.numel() == 0:
            raise ValueError(
                "Reference timestamps must not be empty."
            )

        if self.timestamps.device.type != "cpu":
            raise ValueError(
                "Reference timestamps must be stored on CPU."
            )

        if self.timestamps.requires_grad:
            raise ValueError(
                "Reference timestamps must be detached from autograd."
            )

        if not torch.is_floating_point(
            self.timestamps
        ):
            raise TypeError(
                "Reference timestamps must have floating-point dtype."
            )

        if not torch.isfinite(
            self.timestamps
        ).all():
            raise ValueError(
                "Reference timestamps must be finite."
            )

        if (
            self.timestamps.numel() > 1
            and torch.any(
                self.timestamps[1:]
                <= self.timestamps[:-1]
            )
        ):
            raise ValueError(
                "Reference timestamps must be strictly increasing."
            )


def _estimate_sampling_rate_hz(
    stream: SensorStream,
) -> float:
    '''Estimate one stream's sampling rate from its actual timestamp spacing.

    The median interval is used instead of sample count so streams observed over
    slightly different window spans are compared by cadence rather than by the
    number of samples that happened to fall inside the window.
    '''

    timestamps = stream.timestamps

    if timestamps.numel() < 2:
        return math.inf

    intervals = (
        timestamps[1:]
        - timestamps[:-1]
    )

    if torch.any(
        intervals <= 0.0
    ):
        raise ValueError(
            "Sampling-rate estimation requires strictly increasing timestamps."
        )

    median_interval = float(
        torch.median(
            intervals
        ).item()
    )

    if (
        not math.isfinite(
            median_interval
        )
        or median_interval <= 0.0
    ):
        raise ValueError(
            "Sampling-rate estimation produced an invalid timestamp interval."
        )

    return (
        1.0
        / median_interval
    )


def _select_lowest_rate_lidar(
    streams: Mapping[
        str,
        SensorStream,
    ],
    metadata: Mapping[
        str,
        SensorMetadata,
    ],
) -> str:
    '''Select the LiDAR pose stream with the lowest timestamp-derived rate.'''

    candidates: list[
        tuple[
            float,
            str,
        ]
    ] = []

    for stream_key, stream_metadata in metadata.items():
        if stream_metadata.measurement_type != MeasurementType.LIDAR_POSE:
            continue

        stream = streams[
            stream_key
        ]
        stream.validate()

        sampling_rate_hz = _estimate_sampling_rate_hz(
            stream
        )

        if math.isfinite(
            sampling_rate_hz
        ):
            candidates.append(
                (
                    sampling_rate_hz,
                    stream_key,
                )
            )

    if not candidates:
        raise ValueError(
            "LOWEST_RATE_LIDAR requires at least one LiDAR pose stream with at least two timestamps."
        )

    # Sort first by physical sampling rate and then by stream key so equal-rate
    # sensors have a deterministic, data-order-independent tie break.
    _, stream_key = min(
        candidates,
        key=lambda item: (
            item[0],
            item[1],
        ),
    )

    return stream_key


def select_reference_timebase_single_window(
    streams: Mapping[
        str,
        SensorStream,
    ],
    metadata: Mapping[
        str,
        SensorMetadata,
    ],
    config: ReferenceTimeConfig,
) -> ReferenceTimebase:
    '''Select one reference timebase without relying on hard-coded sensor names.

    Relative-pose streams contribute their end timestamps. Their interval-start
    timestamps remain available on SensorStream for later factor construction.

    ``LOWEST_RATE_LIDAR`` compares the median timestamp cadence of all eligible
    LiDAR pose streams. It therefore selects by measured rate instead of using
    sample count as a proxy.
    '''

    if not streams:
        raise ValueError(
            "At least one sensor stream is required."
        )

    if set(streams) != set(metadata):
        raise ValueError(
            "streams and metadata must contain identical keys."
        )

    if config.policy == ReferenceTimePolicy.EXPLICIT_STREAM:
        stream_key = config.explicit_stream_key

        if stream_key not in streams:
            raise KeyError(
                f"Unknown explicit reference stream {stream_key!r}."
            )

    elif config.policy == ReferenceTimePolicy.LOWEST_RATE_LIDAR:
        stream_key = _select_lowest_rate_lidar(
            streams,
            metadata,
        )

    else:
        raise ValueError(
            f"Unsupported reference-time policy {config.policy!r}."
        )

    stream = streams[
        stream_key
    ]
    stream.validate()

    timebase = ReferenceTimebase(
        stream_key=stream_key,
        timestamps=stream.timestamps,
    )
    timebase.validate()

    return timebase