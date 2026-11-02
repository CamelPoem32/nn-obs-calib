"""Measurement-noise augmentation for heterogeneous raw sensor streams."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math

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
    performed by this class. See OdometryCorruptionAugmenter below for
    time-correlated SE3 odometry noise.
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
            dataset_indices=window.dataset_indices,
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
        dataset_indices=window.dataset_indices,
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

@dataclass(frozen=True)
class SE3OdometryCorruptionConfig:
    """Corrupt a relative-SE3 odometry stream, using its recorded edge intervals.

    All angle parameters are expressed in degrees to make LiDAR ablations easy
    to configure. Translation scaling is dimensionless and constant per window.
    The bias rate is a zero-mean stationary Ornstein-Uhlenbeck process in deg/s,
    not a fixed extra extrinsic rotation. Its integral creates cumulative drift.

    ``probability`` is sampled independently for each batch item and stream.
    Setting it below one retains completely clean windows.
    """

    probability: float = 1.0
    rotation_noise_std_deg: float = 0.0
    rotation_bias_rate_std_deg_s: float = 0.0
    rotation_bias_correlation_time_s: float = 2.0
    translation_scale_min: float = 1.0
    translation_scale_max: float = 1.0

    def __post_init__(self) -> None:
        for name in ("probability", "rotation_noise_std_deg", "rotation_bias_rate_std_deg_s", "rotation_bias_correlation_time_s", "translation_scale_min", "translation_scale_max"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite.")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be between 0 and 1.")
        if self.rotation_noise_std_deg < 0.0 or self.rotation_bias_rate_std_deg_s < 0.0:
            raise ValueError("Rotation noise and bias standard deviations must be nonnegative.")
        if self.rotation_bias_correlation_time_s <= 0.0:
            raise ValueError("rotation_bias_correlation_time_s must be positive.")
        if self.translation_scale_min <= 0.0 or self.translation_scale_max < self.translation_scale_min:
            raise ValueError("Translation scale bounds must satisfy 0 < min <= max.")


@dataclass(frozen=True)
class OdometryCorruptionAugmentationConfig:
    """Select exactly which SE3 measurement types receive odometry corruption.

    Keys must be the actual values of ``SensorMetadata.measurement_type``.
    This prevents accidental augmentation of unrelated SE3 sensor streams.
    """

    enabled: bool = False
    se3_by_type: Mapping[object, SE3OdometryCorruptionConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(not isinstance(value, SE3OdometryCorruptionConfig) for value in self.se3_by_type.values()):
            raise TypeError("Every se3_by_type value must be SE3OdometryCorruptionConfig.")


class OdometryCorruptionAugmenter:
    """Apply independent/temporally-correlated errors to relative LiDAR SE3 edges.

    This class is intentionally separate from MeasurementNoiseAugmenter, so
    existing configs, call signatures and pipelines preserve their behavior.
    It never changes extrinsic calibration states, supervision targets,
    timestamps, interval starts, or padding. No resampling is performed.

    The correlated bias is sampled afresh per batch window: applying this
    augmenter to independently batched physical windows does not preserve
    bias continuity from one window to the next.
    """

    def __init__(self, config: OdometryCorruptionAugmentationConfig) -> None:
        self.config = config

    def __call__(
        self,
        window: WindowBatch,
        generator: torch.Generator | None = None,
    ) -> tuple[WindowBatch, dict[str, dict[str, torch.Tensor]]]:
        """Return (augmented_window, diagnostics_by_stream).

        Diagnostics contain ``applied`` [B], ``translation_scale`` [B],
        ``rotation_bias_rate_rad_s`` [B,N,3], and
        ``rotation_error_rotvec_rad`` [B,N,3]. No diagnostic tensors have
        gradients attached to the sensor model.
        """

        _validate_generator(generator)
        _validate_window(window)
        if not self.config.enabled:
            return _copy_window(window), {}

        streams: dict[str, SensorStreamBatch] = {}
        diagnostics: dict[str, dict[str, torch.Tensor]] = {}
        for stream_key, stream in window.streams.items():
            metadata = window.metadata[stream_key]
            config = self.config.se3_by_type.get(metadata.measurement_type)
            if config is None:
                streams[stream_key] = _copy_stream_with_values(stream, stream.values)
                continue
            if metadata.geometry_type != GeometryType.SE3:
                raise ValueError(f"Odometry corruption requires SE3 geometry for {stream_key!r}, got {metadata.geometry_type!r}.")
            values, diagnostics[stream_key] = _corrupt_se3_odometry(stream, config, generator)
            streams[stream_key] = _copy_stream_with_values(stream, values)

        return WindowBatch(
            streams=streams,
            current_calibration=dict(window.current_calibration),
            metadata=dict(window.metadata),
            targets=None if window.targets is None else dict(window.targets),
            dataset_indices=window.dataset_indices,
        ), diagnostics


def _corrupt_se3_odometry(
    stream: SensorStreamBatch,
    config: SE3OdometryCorruptionConfig,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply exp(rotvec) to SE3 relative edges and scale their translation.

    For edge k, the rotation error is

        phi_k = b_k * (t_end_k - t_start_k) + sigma_white * epsilon_k,
        b_k = exp(-dt_k / tau) * b_{k-1}
              + sqrt(1 - exp(-2*dt_k/tau)) * sigma_bias * eta_k.

    The OU step dt_k is measured between *end timestamps* of consecutive
    valid edges, whereas the angle accumulated on each edge uses its own
    [start, end] duration. Every batch member has an independent bias path.

    The output relative edge is ``Exp([phi_k, 0]) @ T_k_scaled``; scaling
    translation before the left rotation keeps this a valid SE3 transform.
    This corrupts relative odometry, not a fixed frame-to-frame extrinsic.
    """

    values = stream.values
    if values.ndim != 4 or values.shape[-2:] != (4, 4):
        raise ValueError("SE3 odometry measurements must have shape [B, N, 4, 4].")

    mask = stream.sample_mask
    if mask.shape != values.shape[:2]:
        raise ValueError("SE3 odometry sample_mask must have shape [B, N].")
    batch_size, num_samples = values.shape[:2]
    shape = (batch_size, num_samples, 3)
    dtype, device = values.dtype, values.device

    zero_path = torch.zeros(shape, dtype=dtype, device=device)
    applied = torch.zeros(batch_size, device=device, dtype=torch.bool)
    scales = torch.ones(batch_size, dtype=dtype, device=device)
    diagnostics = {
        "applied": applied,
        "translation_scale": scales,
        "rotation_bias_rate_rad_s": zero_path,
        "rotation_error_rotvec_rad": zero_path,
    }
    active_effect = (config.rotation_noise_std_deg > 0.0 or config.rotation_bias_rate_std_deg_s > 0.0
                     or config.translation_scale_min != 1.0 or config.translation_scale_max != 1.0)
    if config.probability == 0.0 or not active_effect or not mask.any():
        return values, diagnostics

    if config.probability == 1.0:
        applied = torch.ones(batch_size, device=device, dtype=torch.bool)
    else:
        applied = torch.rand(batch_size, device=device, generator=generator) < config.probability
    if not applied.any():
        diagnostics["applied"] = applied
        return values, diagnostics

    if config.translation_scale_min != config.translation_scale_max:
        candidate_scales = config.translation_scale_min + (config.translation_scale_max - config.translation_scale_min) * torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
    else:
        candidate_scales = torch.full((batch_size,), config.translation_scale_min, dtype=dtype, device=device)
    scales = torch.where(applied, candidate_scales, scales)

    rotation_noise = zero_path
    if config.rotation_noise_std_deg > 0.0:
        rotation_noise = math.radians(config.rotation_noise_std_deg) * torch.randn(shape, device=device, dtype=dtype, generator=generator)

    bias_rate = zero_path
    if config.rotation_bias_rate_std_deg_s > 0.0:
        ends = stream.timestamps
        starts = stream.interval_start_timestamps
        if starts is None or ends.shape != mask.shape or starts.shape != mask.shape:
            raise ValueError("Correlated rotational bias requires interval_start_timestamps and timestamps with shape [B, N].")
        durations = ends - starts
        if (durations[mask] <= 0.0).any() or not torch.isfinite(durations[mask]).all():
            raise ValueError("Valid LiDAR odometry intervals must have finite, strictly positive durations in seconds.")

        sigma = math.radians(config.rotation_bias_rate_std_deg_s)
        b = sigma * torch.randn((batch_size, 3), device=device, dtype=dtype, generator=generator)
        prev_end = torch.zeros(batch_size, device=device, dtype=ends.dtype)
        has_prev = torch.zeros(batch_size, device=device, dtype=torch.bool)
        bias_values = []
        for k in range(num_samples):
            valid = mask[:, k]
            gap = ends[:, k] - prev_end
            if ((gap < 0.0) & has_prev & valid).any():
                raise ValueError("Valid LiDAR odometry end timestamps must be monotonically nondecreasing.")
            gap = torch.where(has_prev, gap.clamp_min(0.0), torch.zeros_like(gap))
            decay = torch.exp(-gap.to(dtype) / config.rotation_bias_correlation_time_s)
            innovation = sigma * torch.sqrt((1.0 - decay.square()).clamp_min(0.0))
            candidate_b = decay[:, None] * b + innovation[:, None] * torch.randn((batch_size, 3), device=device, dtype=dtype, generator=generator)
            b = torch.where(valid[:, None], candidate_b, b)
            bias_values.append(torch.where(valid[:, None], b, torch.zeros_like(b)))
            prev_end = torch.where(valid, ends[:, k], prev_end)
            has_prev |= valid
        bias_rate = torch.stack(bias_values, dim=1) if num_samples else zero_path
        rotation_noise = rotation_noise + bias_rate * durations.to(dtype)[..., None]

    rotation_noise = torch.where((mask & applied[:, None])[..., None], rotation_noise, zero_path)
    if config.rotation_noise_std_deg > 0.0 or config.rotation_bias_rate_std_deg_s > 0.0:
        dR = so3_exp(rotation_noise)
    else:
        dR = torch.eye(3, dtype=dtype, device=device).expand(batch_size, num_samples, 3, 3)

    result = values.clone()
    noisy_rotation = dR @ values[..., :3, :3]
    noisy_translation = (dR @ (values[..., :3, 3] * scales[:, None, None])[..., None]).squeeze(-1)
    result[..., :3, :3] = torch.where(mask[..., None, None], noisy_rotation, values[..., :3, :3])
    result[..., :3, 3] = torch.where(mask[..., None], noisy_translation, values[..., :3, 3])

    diagnostics["applied"] = applied
    diagnostics["translation_scale"] = scales
    diagnostics["rotation_bias_rate_rad_s"] = torch.where((mask & applied[:, None])[..., None], bias_rate, zero_path)
    diagnostics["rotation_error_rotvec_rad"] = rotation_noise
    return result, diagnostics
