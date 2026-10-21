"""Randomization of physical sensor coordinate frames."""

from __future__ import annotations

import torch

from obscalib.augmentations.config import FrameRandomizationConfig
from obscalib.augmentations.sampling import sample_uniform_so3
from obscalib.calibration.state import CalibrationState
from obscalib.data.structures import GeometryType, SensorStreamBatch, WindowBatch
from obscalib.augmentations.sampling import sample_isotropic_perturbation
from obscalib.geometry.lie import so3_exp

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

        
        frame_randomization_by_key = _sample_frame_randomizations(window=window, config=self.config, generator=generator)


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
            dataset_indices=window.dataset_indices,
        )

        return randomized_window, randomized_calibration, frame_randomization_by_key



def _sample_frame_randomizations(window: WindowBatch, config: FrameRandomizationConfig, generator: torch.Generator | None) -> dict[str, torch.Tensor]:
    """
    Sample independent frame transformations for exactly N sensor keys per active window.

    A sensor key is selected once, even if it has several raw streams
    such as gyroscope and accelerometer.

    Returns dense [B, 4, 4] matrices for every sensor key. Unselected
    sensors receive identity transformations.
    """

    # Sort physical calibration keys to make random selection reproducible.
    calibration_keys = tuple(sorted({metadata.calibration_key for metadata in window.metadata.values()}))
    reference = window.current_calibration[calibration_keys[0]].transform
    batch_size, device, dtype = reference.shape[0], reference.device, reference.dtype

    # By default, every physical sensor is independently randomized.
    sensor_count = len(calibration_keys) if config.sensors_per_window is None else config.sensors_per_window

    if sensor_count > len(calibration_keys):
        raise ValueError(f"sensors_per_window={sensor_count} exceeds the {len(calibration_keys)} physical sensors in this window.")

    # Every calibration key must share the same batch representation.
    for key in calibration_keys:
        transform = window.current_calibration[key].transform
        if transform.shape[0] != batch_size or transform.device != device or transform.dtype != dtype:
            raise ValueError("All calibration states must share batch size, device and dtype.")

    # A separate random permutation for each batch item.
    # The first sensor_count positions are selected without replacement.
    selection_ranks = torch.rand(batch_size, len(calibration_keys), device=device, generator=generator).argsort(dim=1).argsort(dim=1)
    window_enabled = torch.rand(batch_size, 1, device=device, generator=generator) < config.probability
    selected = (selection_ranks < sensor_count) & window_enabled

    frame_randomization_by_key: dict[str, torch.Tensor] = {}

    for sensor_index, calibration_key in enumerate(calibration_keys):
        # Begin with an identity rotation for every batch item.
        R = torch.eye(3, device=device, dtype=dtype).expand(batch_size, 3, 3).clone()

        if config.rotation is None and config.translation is None:
            # Backward-compatible case: unrestricted Haar-uniform SO(3).
            R = sample_uniform_so3(batch_size=batch_size, device=device, dtype=dtype, generator=generator)

        elif config.rotation is not None:
            # Sample an isotropic axis and a bounded rotation-vector magnitude.
            delta_phi = sample_isotropic_perturbation(config=config.rotation, batch_size=batch_size, dimension=3, device=device, dtype=dtype, generator=generator)
            R = so3_exp(delta_phi)

        # Translation is optional and is expressed in the original sensor frame.
        t = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        translation_enabled = config.translation is not None and (config.translation_keys is None or calibration_key in config.translation_keys)

        if translation_enabled:
            # Translating an IMU coordinate origin requires additional acceleration terms.
            # Reject that operation rather than creating inconsistent IMU measurements.
            has_vector_stream = any(window.metadata[stream_key].calibration_key == calibration_key and window.metadata[stream_key].geometry_type == GeometryType.VECTOR for stream_key in window.streams)

            if has_vector_stream and bool(selected[:, sensor_index].any()):
                raise ValueError(f"Cannot translate IMU/VECTOR frame {calibration_key!r} without lever-arm acceleration compensation.")

            t = sample_isotropic_perturbation(config=config.translation, batch_size=batch_size, dimension=3, device=device, dtype=dtype, generator=generator)

        # Construct A_S = [R, t; 0, 1].
        A = torch.eye(4, device=device, dtype=dtype).expand(batch_size, 4, 4).clone()
        A[:, :3, :3] = R
        A[:, :3, 3] = t

        # Only selected sensor keys receive the sampled transformation.
        # All other keys get an exact identity, preserving raw data unchanged.
        frame_randomization_by_key[calibration_key] = torch.where(selected[:, sensor_index, None, None], A, torch.eye(4, device=device, dtype=dtype))

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

        if bool((torch.linalg.vector_norm(frame_randomization[:, :3, 3], dim=-1) > 1e-10).any()):
            raise ValueError("Translating an IMU measurement frame requires lever-arm compensation.")

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
        A_inverse = torch.linalg.inv(frame_randomization)[:, None]

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
        dataset_indices=window.dataset_indices,
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