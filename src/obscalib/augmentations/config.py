"""Composable configuration for synthetic sensor-calibration augmentation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from obscalib.augmentations.profiles import TransitionProfile
from obscalib.data.structures import MeasurementType


class PerturbationDistribution(str, Enum):
    """Distribution used to sample a nonnegative perturbation magnitude."""

    # Sample the magnitude uniformly from [0, maximum].
    UNIFORM = "uniform"

    # Sample the absolute value of a zero-mean Gaussian and truncate it at maximum.
    TRUNCATED_NORMAL = "truncated_normal"

    # Sample the absolute value of a Student-t variable and truncate it at maximum.
    TRUNCATED_STUDENT_T = "truncated_student_t"


@dataclass(frozen=True)
class PerturbationMagnitudeConfig:
    """Configuration for a bounded perturbation magnitude.

    UNIFORM samples a magnitude directly from [0, maximum].

    TRUNCATED_NORMAL and TRUNCATED_STUDENT_T use `scale` as the typical
    perturbation magnitude and truncate the sampled magnitude at `maximum`.

    `degrees_of_freedom` is used only by TRUNCATED_STUDENT_T.
    """

    # Hard upper bound on the sampled nonnegative perturbation magnitude.
    maximum: float

    # Probability distribution used to sample the perturbation magnitude.
    distribution: PerturbationDistribution = PerturbationDistribution.UNIFORM

    # Characteristic scale of the normal or Student-t distribution.
    # It is unused by UNIFORM.
    scale: float | None = None

    # Degrees of freedom of TRUNCATED_STUDENT_T.
    # Smaller values produce heavier tails. It is unused by other distributions.
    degrees_of_freedom: int = 3

    def __post_init__(self) -> None:
        if not math.isfinite(self.maximum) or self.maximum <= 0.0:
            raise ValueError("maximum must be finite and positive.")

        if self.scale is not None and (not math.isfinite(self.scale) or self.scale <= 0.0):
            raise ValueError("scale must be finite and positive when provided.")

        if self.distribution in (PerturbationDistribution.TRUNCATED_NORMAL, PerturbationDistribution.TRUNCATED_STUDENT_T) and self.scale is None:
            raise ValueError(f"{self.distribution.value} requires scale.")

        if not isinstance(self.degrees_of_freedom, int) or self.degrees_of_freedom <= 0:
            raise ValueError("degrees_of_freedom must be a positive integer.")


@dataclass(frozen=True)
class FrameRandomizationConfig:
    """Configuration for arbitrary physical sensor-frame orientation randomization.

    When enabled, one arbitrary SO3 rotation will be sampled independently for
    each physical calibration key and batch item. The sampled transformation
    will be represented and recorded as an SE3 matrix with zero translation.
    """

    # Enable or disable arbitrary orientation randomization of physical sensor frames.
    enabled: bool = False


@dataclass(frozen=True)
class PriorPerturbationConfig:
    """Configuration for perturbing the calibration prior supplied to the model.

    Prior perturbation is independent of a true calibration-change event and
    may therefore occur in both static and dynamic windows.
    """

    # Enable or disable calibration-prior perturbation.
    enabled: bool = False

    # Default probability that the supplied prior is perturbed for one
    # calibration key and one batch item.
    probability: float = 0.5

    # Optional per-calibration-key overrides of `probability`.
    # Example: {"imu": 0.7, "lidar": 0.3}.
    probability_by_key: dict[str, float] = field(default_factory=dict)

    # Magnitude distribution for rotational perturbation phi [rad].
    # None disables rotational prior perturbation.
    rotation: PerturbationMagnitudeConfig | None = None

    # Magnitude distribution for the SE3 tangent translation component rho [m].
    # None disables translational prior perturbation.
    translation: PerturbationMagnitudeConfig | None = None

    # Magnitude distribution for additive temporal perturbation delta_tau [s].
    # None disables temporal prior perturbation.
    time_offset: PerturbationMagnitudeConfig | None = None

    def __post_init__(self) -> None:
        _validate_probability("probability", self.probability)
        _validate_probability_map("probability_by_key", self.probability_by_key)

        if self.enabled and self.rotation is None and self.translation is None and self.time_offset is None:
            raise ValueError("Enabled prior perturbation requires at least one perturbation magnitude config.")


@dataclass(frozen=True)
class CalibrationEventConfig:
    """Configuration for independent true calibration-change events.

    Event occurrence is sampled independently for each physical calibration key.

    Conditional on an event occurring, rotation, translation, and temporal
    components are selected independently according to their corresponding
    probabilities. The sampler must ensure that at least one component is
    selected for every positive event.
    """

    # Enable or disable true synthetic calibration-change events.
    enabled: bool = False

    # Default probability that a calibration event occurs for one calibration
    # key and one batch item.
    event_probability: float = 0.5

    # Optional per-calibration-key overrides of `event_probability`.
    # Example: {"imu": 0.5, "lidar": 0.2}.
    event_probability_by_key: dict[str, float] = field(default_factory=dict)

    # Conditional probability that an occurred event contains a rotation change.
    rotation_probability: float = 0.0

    # Conditional probability that an occurred event contains a translation change.
    translation_probability: float = 0.0

    # Conditional probability that an occurred event contains a time-offset change.
    time_offset_probability: float = 0.0

    # Magnitude distribution for the event rotation vector delta_phi [rad].
    # Required when rotation_probability is positive.
    rotation: PerturbationMagnitudeConfig | None = None

    # Magnitude distribution for the event tangent translation delta_rho [m].
    # Required when translation_probability is positive.
    translation: PerturbationMagnitudeConfig | None = None

    # Magnitude distribution for the event temporal change delta_tau [s].
    # Required when time_offset_probability is positive.
    time_offset: PerturbationMagnitudeConfig | None = None

    # Relative sampling weights of the allowed transition profiles.
    # Zero weight disables a profile without removing it from the configuration.
    profile_weights: dict[TransitionProfile, float] = field(
        default_factory=lambda: {
            TransitionProfile.STEP: 1.0,
            TransitionProfile.LINEAR: 1.0,
            TransitionProfile.SMOOTHSTEP: 1.0,
        }
    )

    # Minimum duration [s] of LINEAR and SMOOTHSTEP transitions.
    # STEP transitions always have zero duration.
    minimum_transition_duration_s: float | None = None

    # Maximum duration [s] of LINEAR and SMOOTHSTEP transitions.
    # STEP transitions always have zero duration.
    maximum_transition_duration_s: float | None = None

    # Minimum fraction of the window that must remain before the beginning of
    # the complete transition.
    minimum_pre_event_fraction: float = 0.2

    # Minimum fraction of the window that must remain after the end of the
    # complete transition.
    minimum_post_event_fraction: float = 0.2

    def __post_init__(self) -> None:
        _validate_probability("event_probability", self.event_probability)
        _validate_probability_map("event_probability_by_key", self.event_probability_by_key)

        _validate_probability("rotation_probability", self.rotation_probability)
        _validate_probability("translation_probability", self.translation_probability)
        _validate_probability("time_offset_probability", self.time_offset_probability)

        if self.rotation_probability > 0.0 and self.rotation is None:
            raise ValueError("A positive rotation_probability requires a rotation magnitude config.")

        if self.translation_probability > 0.0 and self.translation is None:
            raise ValueError("A positive translation_probability requires a translation magnitude config.")

        if self.time_offset_probability > 0.0 and self.time_offset is None:
            raise ValueError("A positive time_offset_probability requires a time-offset magnitude config.")

        _validate_profile_weights(self.profile_weights)
        _validate_transition_durations(self.minimum_transition_duration_s, self.maximum_transition_duration_s)

        _validate_probability("minimum_pre_event_fraction", self.minimum_pre_event_fraction)
        _validate_probability("minimum_post_event_fraction", self.minimum_post_event_fraction)

        if self.minimum_pre_event_fraction + self.minimum_post_event_fraction >= 1.0:
            raise ValueError("minimum_pre_event_fraction + minimum_post_event_fraction must be smaller than 1.")

        if not self.enabled:
            return

        has_possible_event = self.event_probability > 0.0 or any(probability > 0.0 for probability in self.event_probability_by_key.values())

        if not has_possible_event:
            raise ValueError("Enabled calibration-event augmentation requires a nonzero event probability.")

        has_possible_component = self.rotation_probability > 0.0 or self.translation_probability > 0.0 or self.time_offset_probability > 0.0

        if not has_possible_component:
            raise ValueError("Enabled calibration-event augmentation requires at least one event component with nonzero probability.")

        has_finite_transition = any(
            profile in (TransitionProfile.LINEAR, TransitionProfile.SMOOTHSTEP) and weight > 0.0
            for profile, weight in self.profile_weights.items()
        )

        if has_finite_transition and (self.minimum_transition_duration_s is None or self.maximum_transition_duration_s is None):
            raise ValueError("LINEAR or SMOOTHSTEP events require minimum and maximum transition durations.")


@dataclass(frozen=True)
class VectorNoiseConfig:
    """Additive noise configuration for ordinary vector measurements."""

    # Per-sample zero-mean Gaussian measurement-noise standard deviation.
    gaussian_std: float = 0.0

    # Standard deviation of one constant additive bias sampled for the complete
    # vector stream within a window.
    window_bias_std: float = 0.0

    def __post_init__(self) -> None:
        _validate_nonnegative_finite("gaussian_std", self.gaussian_std)
        _validate_nonnegative_finite("window_bias_std", self.window_bias_std)


@dataclass(frozen=True)
class SO3NoiseConfig:
    """Tangent-space rotational noise configuration for SO3 measurements."""

    # Standard deviation [rad] of rotational noise sampled in so(3).
    rotation_std: float = 0.0

    def __post_init__(self) -> None:
        _validate_nonnegative_finite("rotation_std", self.rotation_std)


@dataclass(frozen=True)
class SE3NoiseConfig:
    """Separate rotational and translational tangent-space noise for SE3 measurements."""

    # Standard deviation [rad] of the rotational tangent component.
    rotation_std: float = 0.0

    # Standard deviation [m] of the translational tangent component.
    translation_std: float = 0.0

    def __post_init__(self) -> None:
        _validate_nonnegative_finite("rotation_std", self.rotation_std)
        _validate_nonnegative_finite("translation_std", self.translation_std)


@dataclass(frozen=True)
class NoiseAugmentationConfig:
    """Measurement-type-aware noise and small constant vector bias configuration."""

    # Enable or disable measurement-noise augmentation.
    enabled: bool = False

    # Noise configurations for ordinary VECTOR measurement types.
    vector_by_type: dict[MeasurementType, VectorNoiseConfig] = field(default_factory=dict)

    # Noise configurations for SO3-valued measurement types.
    so3_by_type: dict[MeasurementType, SO3NoiseConfig] = field(default_factory=dict)

    # Noise configurations for SE3-valued measurement types.
    se3_by_type: dict[MeasurementType, SE3NoiseConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_typed_noise_map("vector_by_type", self.vector_by_type, VectorNoiseConfig)
        _validate_typed_noise_map("so3_by_type", self.so3_by_type, SO3NoiseConfig)
        _validate_typed_noise_map("se3_by_type", self.se3_by_type, SE3NoiseConfig)

        configured_types = list(self.vector_by_type) + list(self.so3_by_type) + list(self.se3_by_type)

        if len(configured_types) != len(set(configured_types)):
            raise ValueError("A MeasurementType may be configured in only one noise map.")

        if self.enabled and not self._has_nonzero_noise():
            raise ValueError("Enabled noise augmentation requires at least one positive noise or bias standard deviation.")

    def _has_nonzero_noise(self) -> bool:
        if any(config.gaussian_std > 0.0 or config.window_bias_std > 0.0 for config in self.vector_by_type.values()):
            return True

        if any(config.rotation_std > 0.0 for config in self.so3_by_type.values()):
            return True

        if any(config.rotation_std > 0.0 or config.translation_std > 0.0 for config in self.se3_by_type.values()):
            return True

        return False


@dataclass(frozen=True)
class AugmentationConfig:
    """Top-level composition of independently configurable augmentation stages."""

    # Coordinate-frame randomization configuration.
    frame_randomization: FrameRandomizationConfig = field(default_factory=FrameRandomizationConfig)

    # Imperfect model-input calibration-prior configuration.
    prior_perturbation: PriorPerturbationConfig = field(default_factory=PriorPerturbationConfig)

    # True within-window calibration-change event configuration.
    calibration_event: CalibrationEventConfig = field(default_factory=CalibrationEventConfig)

    # Measurement-noise augmentation configuration.
    noise: NoiseAugmentationConfig = field(default_factory=NoiseAugmentationConfig)


def _validate_probability(name: str, probability: float) -> None:
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1].")


def _validate_probability_map(name: str, values: dict[str, float]) -> None:
    for calibration_key, probability in values.items():
        if not calibration_key:
            raise ValueError(f"{name} keys must be non-empty.")

        _validate_probability(f"{name}[{calibration_key!r}]", probability)


def _validate_profile_weights(profile_weights: dict[TransitionProfile, float]) -> None:
    if not profile_weights:
        raise ValueError("profile_weights must not be empty.")

    for profile, weight in profile_weights.items():
        if not isinstance(profile, TransitionProfile):
            raise TypeError("profile_weights keys must be TransitionProfile values.")

        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("profile_weights values must be finite and nonnegative.")

    if sum(profile_weights.values()) <= 0.0:
        raise ValueError("profile_weights must contain at least one positive weight.")


def _validate_transition_durations(minimum_transition_duration_s: float | None, maximum_transition_duration_s: float | None) -> None:
    if (minimum_transition_duration_s is None) != (maximum_transition_duration_s is None):
        raise ValueError("Minimum and maximum transition durations must be configured together.")

    if minimum_transition_duration_s is None:
        return

    if not math.isfinite(minimum_transition_duration_s) or minimum_transition_duration_s <= 0.0:
        raise ValueError("minimum_transition_duration_s must be finite and positive.")

    if not math.isfinite(maximum_transition_duration_s) or maximum_transition_duration_s <= 0.0:
        raise ValueError("maximum_transition_duration_s must be finite and positive.")

    if maximum_transition_duration_s < minimum_transition_duration_s:
        raise ValueError("maximum_transition_duration_s must be at least minimum_transition_duration_s.")


def _validate_nonnegative_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")


def _validate_typed_noise_map(name: str, values: dict[MeasurementType, object], expected_type: type) -> None:
    for measurement_type, config in values.items():
        if not isinstance(measurement_type, MeasurementType):
            raise TypeError(f"{name} keys must be MeasurementType values.")

        if not isinstance(config, expected_type):
            raise TypeError(f"{name} values must be {expected_type.__name__} instances.")