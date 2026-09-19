"""Measurement-noise augmentation for heterogeneous raw sensor streams."""

from __future__ import annotations

import torch

from obscalib.augmentations.config import NoiseAugmentationConfig, SE3NoiseConfig, SO3NoiseConfig, VectorNoiseConfig
from obscalib.data.structures import GeometryType, SensorMetadata, SensorStreamBatch, WindowBatch
from obscalib.geometry.lie import se3_exp, so3_exp


class MeasurementNoiseAugmenter:
    """
    Apply modest measurement noise while preserving the raw stream geometry.

    VECTOR:
        Add independent per-sample Gaussian noise and an optional constant
        per-window bias.

    SO3:
        Apply a small left-multiplicative tangent-space rotation perturbation.

    SE3:
        Apply a small left-multiplicative tangent-space SE3 perturbation with
        independently configured rotational and translational standard
        deviations.

    No random-walk bias, timestamp noise, interpolation, or resampling is
    performed here.
    """

    def __init__(self, config: NoiseAugmentationConfig) -> None:
        self.config = config

    def __call__(
        self,
        window: WindowBatch,
        generator: torch.Generator | None = None,
    ) -> tuple[WindowBatch, dict[str, torch.Tensor]]:
        """
        Add configured measurement noise to every matching raw stream.

        Returns:
            augmented_window:
                New WindowBatch containing noisy streams.

            noise_bias_by_stream:
                Constant VECTOR bias sampled for each stream, with shape
                [B, 3]. Streams without a configured VECTOR bias are omitted.
        """

        _validate_generator(generator)
        _validate_window(window)

        if not self.config.enabled:
            return _copy_window(window), {}

        augmented_streams: dict[str, SensorStreamBatch] = {}
        noise_bias_by_stream: dict[str, torch.Tensor] = {}

        for stream_key, stream in window.streams.items():
            metadata = window.metadata[stream_key]

            augmented_stream, bias = self._augment_stream(
                stream=stream,
                metadata=metadata,
                generator=generator,
            )

            augmented_streams[stream_key] = augmented_stream

            if bias is not None:
                noise_bias_by_stream[stream_key] = bias

        augmented_window = WindowBatch(
            streams=augmented_streams,
            current_calibration=dict(window.current_calibration),
            metadata=dict(window.metadata),
            targets=None if window.targets is None else dict(window.targets),
        )

        return augmented_window, noise_bias_by_stream

    def _augment_stream(
        self,
        stream: SensorStreamBatch,
        metadata: SensorMetadata,
        generator: torch.Generator | None,
    ) -> tuple[SensorStreamBatch, torch.Tensor | None]:
        """Apply the configuration corresponding to one measurement type."""

        measurement_type = metadata.measurement_type

        if measurement_type in self.config.vector_by_type:
            if metadata.geometry_type != GeometryType.VECTOR:
                raise ValueError(f"Measurement type {measurement_type!r} is configured as VECTOR noise but stream geometry is {metadata.geometry_type!r}.")

            values, bias = _augment_vector(
                stream=stream,
                config=self.config.vector_by_type[measurement_type],
                generator=generator,
            )

            return _copy_stream_with_values(stream, values), bias

        if measurement_type in self.config.so3_by_type:
            if metadata.geometry_type != GeometryType.SO3:
                raise ValueError(f"Measurement type {measurement_type!r} is configured as SO3 noise but stream geometry is {metadata.geometry_type!r}.")

            values = _augment_so3(
                stream=stream,
                config=self.config.so3_by_type[measurement_type],
                generator=generator,
            )

            return _copy_stream_with_values(stream, values), None

        if measurement_type in self.config.se3_by_type:
            if metadata.geometry_type != GeometryType.SE3:
                raise ValueError(f"Measurement type {measurement_type!r} is configured as SE3 noise but stream geometry is {metadata.geometry_type!r}.")

            values = _augment_se3(
                stream=stream,
                config=self.config.se3_by_type[measurement_type],
                generator=generator,
            )

            return _copy_stream_with_values(stream, values), None

        # Measurement types without a noise configuration pass through unchanged.
        return _copy_stream_with_values(stream, stream.values), None


def _augment_vector(
    stream: SensorStreamBatch,
    config: VectorNoiseConfig,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Add Gaussian sample noise and one constant bias per batch/window.

    Measurements must have shape [B, N, 3].
    Bias has shape [B, 3] and is shared across all valid samples of the
    corresponding stream within one window.
    """

    values = stream.values

    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("VECTOR measurements must have shape [B, N, 3].")

    batch_size, num_samples, _ = values.shape
    device = values.device
    dtype = values.dtype

    if config.gaussian_std > 0.0:
        gaussian_noise = config.gaussian_std * torch.randn(
            batch_size,
            num_samples,
            3,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    else:
        gaussian_noise = torch.zeros_like(values)

    if config.window_bias_std > 0.0:
        bias = config.window_bias_std * torch.randn(
            batch_size,
            3,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    else:
        bias = torch.zeros(
            batch_size,
            3,
            device=device,
            dtype=dtype,
        )

    perturbation = gaussian_noise + bias[:, None, :]

    # Padding values remain exactly unchanged.
    valid_mask = stream.sample_mask[..., None]

    noisy_values = torch.where(
        valid_mask,
        values + perturbation,
        values,
    )

    return noisy_values, bias


def _augment_so3(
    stream: SensorStreamBatch,
    config: SO3NoiseConfig,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """
    Apply independent tangent-space rotation noise to SO3 measurements.

    Noise follows

        R_noisy = Exp(epsilon_phi) @ R,

    with

        epsilon_phi ~ N(0, rotation_std^2 I).
    """

    values = stream.values

    if values.ndim != 4 or values.shape[-2:] != (3, 3):
        raise ValueError("SO3 measurements must have shape [B, N, 3, 3].")

    if config.rotation_std == 0.0:
        return values

    batch_size, num_samples = values.shape[:2]

    epsilon_phi = config.rotation_std * torch.randn(
        batch_size,
        num_samples,
        3,
        device=values.device,
        dtype=values.dtype,
        generator=generator,
    )

    delta_R = so3_exp(epsilon_phi)
    noisy_values = delta_R @ values

    # Padding matrices may not represent valid SO3 values and must remain
    # untouched by augmentation.
    valid_mask = stream.sample_mask[..., None, None]

    return torch.where(
        valid_mask,
        noisy_values,
        values,
    )


def _augment_se3(
    stream: SensorStreamBatch,
    config: SE3NoiseConfig,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """
    Apply independent tangent-space SE3 noise.

    Noise follows

        T_noisy = Exp(epsilon_xi) @ T,

    where

        epsilon_xi = [epsilon_phi, epsilon_rho].

    epsilon_phi and epsilon_rho use their separately configured standard
    deviations.
    """

    values = stream.values

    if values.ndim != 4 or values.shape[-2:] != (4, 4):
        raise ValueError("SE3 measurements must have shape [B, N, 4, 4].")

    if config.rotation_std == 0.0 and config.translation_std == 0.0:
        return values

    batch_size, num_samples = values.shape[:2]
    device = values.device
    dtype = values.dtype

    epsilon_phi = config.rotation_std * torch.randn(
        batch_size,
        num_samples,
        3,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    epsilon_rho = config.translation_std * torch.randn(
        batch_size,
        num_samples,
        3,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    epsilon_xi = torch.cat(
        (epsilon_phi, epsilon_rho),
        dim=-1,
    )

    delta_T = se3_exp(epsilon_xi)
    noisy_values = delta_T @ values

    valid_mask = stream.sample_mask[..., None, None]

    return torch.where(
        valid_mask,
        noisy_values,
        values,
    )


def _copy_stream_with_values(
    stream: SensorStreamBatch,
    values: torch.Tensor,
) -> SensorStreamBatch:
    """Construct a new stream while preserving timestamps and padding mask."""

    return SensorStreamBatch(
        values=values,
        timestamps=stream.timestamps,
        sample_mask=stream.sample_mask,
        interval_start_timestamps=stream.interval_start_timestamps,
    )


def _copy_window(window: WindowBatch) -> WindowBatch:
    """Copy window containers without cloning unchanged tensors."""

    return WindowBatch(
        streams=dict(window.streams),
        current_calibration=dict(window.current_calibration),
        metadata=dict(window.metadata),
        targets=None if window.targets is None else dict(window.targets),
    )


def _validate_window(window: WindowBatch) -> None:
    """Validate fields required by measurement-noise augmentation."""

    if not window.streams:
        raise ValueError("Measurement-noise augmentation requires at least one sensor stream.")

    if set(window.streams) != set(window.metadata):
        raise ValueError("Window streams and metadata must have identical keys.")

    for stream in window.streams.values():
        stream.validate()


def _validate_generator(generator: torch.Generator | None) -> None:
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or None.")