"""Randomization of physical sensor coordinate frames."""

from __future__ import annotations

import torch

from obscalib.augmentations.config import FrameRandomizationConfig
from obscalib.augmentations.sampling import sample_uniform_so3
from obscalib.calibration.state import CalibrationState
from obscalib.data.structures import GeometryType, SensorStreamBatch, WindowBatch


class SensorFrameRandomizer:
    """
    Randomize one coordinate frame per physical calibration key.

    The sampled frame randomization A uses the convention

        T_WS_randomized = T_WS @ A,

    where T_WS is the original sensor-to-world calibration.

    In the current version, A contains an arbitrary Haar-uniform SO3 rotation
    and zero translation.

    Every raw stream sharing one SensorMetadata.calibration_key is re-expressed
    using the same A, so the physical world-frame measurement represented by the
    stream remains unchanged.
    """

    def __init__(self, config: FrameRandomizationConfig) -> None:
        self.config = config

    def __call__(
        self,
        window: WindowBatch,
        generator: torch.Generator | None = None,
    ) -> tuple[WindowBatch, dict[str, CalibrationState], dict[str, torch.Tensor]]:
        """
        Randomize sensor frames and return the corresponding true calibration.

        Returns:
            randomized_window:
                Window with raw measurements re-expressed in the randomized
                sensor frames and current_calibration updated consistently.

            randomized_calibration:
                True calibration after coordinate-frame randomization.

            frame_randomization_by_key:
                Sampled SE3 matrices A with shape [B, 4, 4], keyed by physical
                calibration key.
        """

        _validate_generator(generator)
        _validate_window(window)

        if not self.config.enabled:
            return _copy_window(window), dict(window.current_calibration), {}

        frame_randomization_by_key = _sample_frame_randomizations(
            window=window,
            generator=generator,
        )

        randomized_calibration = _randomize_calibration(
            calibration=window.current_calibration,
            frame_randomization_by_key=frame_randomization_by_key,
        )

        randomized_streams = {
            stream_key: _randomize_stream(
                stream=stream,
                geometry_type=window.metadata[stream_key].geometry_type,
                frame_randomization=frame_randomization_by_key[window.metadata[stream_key].calibration_key],
            )
            for stream_key, stream in window.streams.items()
        }

        randomized_window = WindowBatch(
            streams=randomized_streams,
            current_calibration=randomized_calibration,
            metadata=dict(window.metadata),
            targets=None if window.targets is None else dict(window.targets),
        )

        return randomized_window, randomized_calibration, frame_randomization_by_key


def _sample_frame_randomizations(
    window: WindowBatch,
    generator: torch.Generator | None,
) -> dict[str, torch.Tensor]:
    """Sample one independent orientation randomization per used calibration key."""

    calibration_keys = {
        metadata.calibration_key
        for metadata in window.metadata.values()
    }

    frame_randomization_by_key: dict[str, torch.Tensor] = {}

    for calibration_key in calibration_keys:
        state = window.current_calibration[calibration_key]

        batch_size = state.transform.shape[0]
        device = state.transform.device
        dtype = state.transform.dtype

        R = sample_uniform_so3(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        A = torch.zeros(
            batch_size,
            4,
            4,
            device=device,
            dtype=dtype,
        )

        A[..., :3, :3] = R
        A[..., 3, 3] = 1.0

        frame_randomization_by_key[calibration_key] = A

    return frame_randomization_by_key


def _randomize_calibration(
    calibration: dict[str, CalibrationState],
    frame_randomization_by_key: dict[str, torch.Tensor],
) -> dict[str, CalibrationState]:
    """Apply T_WS_randomized = T_WS @ A for every randomized calibration key."""

    randomized_calibration: dict[str, CalibrationState] = {}

    for calibration_key, state in calibration.items():
        if calibration_key not in frame_randomization_by_key:
            randomized_calibration[calibration_key] = state
            continue

        A = frame_randomization_by_key[calibration_key]

        randomized_calibration[calibration_key] = CalibrationState(
            transform=state.transform @ A,
            time_offset=state.time_offset,
        )

    return randomized_calibration


def _randomize_stream(
    stream: SensorStreamBatch,
    geometry_type: GeometryType,
    frame_randomization: torch.Tensor,
) -> SensorStreamBatch:
    """Re-express one raw stream in its randomized sensor coordinate frame."""

    values = _reexpress_measurements(
        measurements=stream.values,
        geometry_type=geometry_type,
        frame_randomization=frame_randomization,
    )

    return SensorStreamBatch(
        values=values,
        timestamps=stream.timestamps,
        sample_mask=stream.sample_mask,
        interval_start_timestamps=stream.interval_start_timestamps,
    )


def _reexpress_measurements(
    measurements: torch.Tensor,
    geometry_type: GeometryType,
    frame_randomization: torch.Tensor,
) -> torch.Tensor:
    """
    Re-express raw measurements under T_WS_randomized = T_WS @ A.

    For the current rotation-only A:

        VECTOR:
            v_S' = R_A^T v_S

        SO3 relative update:
            delta_R_S' = R_A^T delta_R_S R_A

        SE3 relative update:
            delta_T_S' = A^-1 delta_T_S A

    VECTOR measurements are stored as row vectors [B, N, 3], so the numerical
    implementation of v_S' = R_A^T v_S is

        values_S' = values_S @ R_A.
    """

    if frame_randomization.ndim != 3 or frame_randomization.shape[-2:] != (4, 4):
        raise ValueError("frame_randomization must have shape [B, 4, 4].")

    if measurements.shape[0] != frame_randomization.shape[0]:
        raise ValueError("Measurements and frame randomization must share batch size.")

    R = frame_randomization[..., :3, :3]
    R_inverse = R.transpose(-1, -2)

    if geometry_type == GeometryType.VECTOR:
        if measurements.ndim != 3 or measurements.shape[-1] != 3:
            raise ValueError("VECTOR measurements must have shape [B, N, 3].")

        return measurements @ R

    if geometry_type == GeometryType.SO3:
        if measurements.ndim != 4 or measurements.shape[-2:] != (3, 3):
            raise ValueError("SO3 measurements must have shape [B, N, 3, 3].")

        R = R[:, None]
        R_inverse = R_inverse[:, None]

        return R_inverse @ measurements @ R

    if geometry_type == GeometryType.SE3:
        if measurements.ndim != 4 or measurements.shape[-2:] != (4, 4):
            raise ValueError("SE3 measurements must have shape [B, N, 4, 4].")

        A = frame_randomization[:, None]
        A_inverse = _inverse_rotation_only_se3(frame_randomization)[:, None]

        return A_inverse @ measurements @ A

    raise ValueError(f"Unsupported geometry type: {geometry_type!r}.")


def _inverse_rotation_only_se3(transform: torch.Tensor) -> torch.Tensor:
    """Invert an SE3 transform known to contain zero translation."""

    inverse = torch.zeros_like(transform)

    inverse[..., :3, :3] = transform[..., :3, :3].transpose(-1, -2)
    inverse[..., 3, 3] = 1.0

    return inverse


def _copy_window(window: WindowBatch) -> WindowBatch:
    """Copy the window containers without cloning unchanged tensors."""

    return WindowBatch(
        streams=dict(window.streams),
        current_calibration=dict(window.current_calibration),
        metadata=dict(window.metadata),
        targets=None if window.targets is None else dict(window.targets),
    )


def _validate_window(window: WindowBatch) -> None:
    """Validate the window fields required by sensor-frame randomization."""

    if not window.streams:
        raise ValueError("Sensor-frame randomization requires at least one sensor stream.")

    if set(window.streams) != set(window.metadata):
        raise ValueError("Window streams and metadata must have identical keys.")

    if not window.current_calibration:
        raise ValueError("Sensor-frame randomization requires at least one calibration state.")

    calibration_keys = {
        metadata.calibration_key
        for metadata in window.metadata.values()
    }

    missing_calibration_keys = calibration_keys - set(window.current_calibration)

    if missing_calibration_keys:
        raise KeyError(f"Missing calibration states for sensor streams: {sorted(missing_calibration_keys)}")

    for calibration_key, state in window.current_calibration.items():
        state.validate()

    for stream_key, stream in window.streams.items():
        stream.validate()

        calibration_key = window.metadata[stream_key].calibration_key
        calibration_state = window.current_calibration[calibration_key]

        if stream.values.shape[0] != calibration_state.transform.shape[0]:
            raise ValueError(f"Stream {stream_key!r} and calibration {calibration_key!r} must share batch size.")


def _validate_generator(generator: torch.Generator | None) -> None:
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or None.")