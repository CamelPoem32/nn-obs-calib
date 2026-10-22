"""Configuration objects for the learned calibration architecture."""

from __future__ import annotations

from dataclasses import dataclass, field


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def _validate_nonnegative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value}.")


def _validate_dropout(dropout: float) -> None:
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}.")


@dataclass(frozen=True)
class TransformerConfig:
    """Configuration for the summary-token Transformer encoder."""

    input_dim: int
    d_model: int
    n_heads: int
    num_layers: int
    dim_feedforward: int
    dropout: float = 0.0
    num_summary_tokens: int = 1
    activation: str = "gelu"

    def __post_init__(self) -> None:
        for name in ("input_dim", "d_model", "n_heads", "num_layers", "dim_feedforward", "num_summary_tokens"):
            _validate_positive(name, getattr(self, name))

        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        if self.activation not in {"relu", "gelu"}:
            raise ValueError("Transformer activation must be 'relu' or 'gelu'.")

        _validate_dropout(self.dropout)


@dataclass(frozen=True)
class MLPConfig:
    """Configuration for a shared MLP whose output feeds calibration heads."""

    hidden_dims: tuple[int, ...]
    output_dim: int
    activation: str = "gelu"
    dropout: float = 0.0

    def __post_init__(self) -> None:
        _validate_positive("output_dim", self.output_dim)

        for index, width in enumerate(self.hidden_dims):
            _validate_positive(f"hidden_dims[{index}]", width)

        _validate_dropout(self.dropout)


@dataclass(frozen=True)
class CalibrationHeadConfig:
    """Dimensions for a reusable sensor-specific calibration head."""

    calibration_context_dim: int
    hidden_dims: tuple[int, ...] = (128,)
    activation: str = "gelu"
    dropout: float = 0.0

    def __post_init__(self) -> None:
        _validate_positive("calibration_context_dim", self.calibration_context_dim)

        for index, width in enumerate(self.hidden_dims):
            _validate_positive(f"hidden_dims[{index}]", width)

        _validate_dropout(self.dropout)


@dataclass(frozen=True)
class LossWeights:
    """Relative weights of supervised calibration-loss components."""

    lambda_rotation: float = 1.0
    lambda_translation: float = 1.0
    lambda_time_offset: float = 1.0
    lambda_change_event: float = 1.0
    lambda_change_time: float = 1.0
    lambda_consistency: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (("lambda_rotation", self.lambda_rotation), ("lambda_translation", self.lambda_translation), ("lambda_time_offset", self.lambda_time_offset), ("lambda_change_event", self.lambda_change_event), ("lambda_change_time", self.lambda_change_time), ("lambda_consistency", self.lambda_consistency)):
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative, got {value}.")


@dataclass(frozen=True)
class CalibrationLossConfig:
    """Configuration of supervised calibration-loss behavior."""

    weights: LossWeights = field(default_factory=LossWeights)
    change_time_beta_s: float = 0.1
    change_positive_weight: float = 1.0
    correction_loss_mode: str = "standard"
    observability_hold_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.change_time_beta_s <= 0.0:
            raise ValueError(f"change_time_beta_s must be positive, got {self.change_time_beta_s}.")

        if self.change_positive_weight <= 0.0:
            raise ValueError(f"change_positive_weight must be positive, got {self.change_positive_weight}.")

        if self.correction_loss_mode not in {"standard", "observability_aware"}:
            raise ValueError("correction_loss_mode must be 'standard' or 'observability_aware'.")

        if self.observability_hold_weight < 0.0:
            raise ValueError(f"observability_hold_weight must be nonnegative, got {self.observability_hold_weight}.")


@dataclass(frozen=True)
class ObsCalibModelConfig:
    """Configuration tying together the learned model component contracts."""

    transformer: TransformerConfig
    mlp: MLPConfig
    calibration_head: CalibrationHeadConfig
    head_keys: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.head_keys:
            raise ValueError("At least one calibration head key is required.")

        if len(set(self.head_keys)) != len(self.head_keys):
            raise ValueError("Calibration head keys must be unique.")

    @property
    def summary_feature_dim(self) -> int:
        """Flattened size of all learned Transformer summary tokens."""

        return self.transformer.num_summary_tokens * self.transformer.d_model

    @property
    def shared_feature_dim(self) -> int:
        """Feature size emitted by the shared MLP for every calibration head."""

        return self.mlp.output_dim


@dataclass(frozen=True)
class MeasurementEncoderConfig:
    """Configuration for sensor-type-specific measurement projections."""

    d_measurement: int = 6

    def __post_init__(self) -> None:
        _validate_positive("d_measurement", self.d_measurement)


@dataclass(frozen=True)
class WindowingConfig:
    """Configuration for temporal sensor-stream window construction."""

    window_duration_s: float = 5.0
    window_stride_s: float | None = None
    max_samples_per_sensor: int = 700

    def __post_init__(self) -> None:
        if self.window_duration_s <= 0.0:
            raise ValueError(f"window_duration_s must be positive, got {self.window_duration_s}.")

        if self.window_stride_s is not None and self.window_stride_s <= 0.0:
            raise ValueError(f"window_stride_s must be positive when provided, got {self.window_stride_s}.")

        if self.max_samples_per_sensor <= 0:
            raise ValueError(f"max_samples_per_sensor must be positive, got {self.max_samples_per_sensor}.")

    @property
    def resolved_window_stride_s(self) -> float:
        """Use non-overlapping windows when no explicit stride is configured."""

        return self.window_duration_s if self.window_stride_s is None else self.window_stride_s


@dataclass(frozen=True)
class ObservabilityPredictionHeadConfig:
    """MLP configuration for one seven-coordinate observability predictor."""

    hidden_dims: tuple[int, ...] = (128, 64)
    activation: str = "gelu"
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for index, hidden_dim in enumerate(self.hidden_dims):
            _validate_positive(f"hidden_dims[{index}]", hidden_dim)

        _validate_dropout(self.dropout)


@dataclass(frozen=True)
class ObservabilityPredictionModelConfig:
    """Configuration for calibration-wise observability prediction."""

    transformer: TransformerConfig
    calibration_keys: tuple[str, ...] = field(default_factory=tuple)
    head: ObservabilityPredictionHeadConfig = field(default_factory=ObservabilityPredictionHeadConfig)
    head_mode: str = "shared"

    def __post_init__(self) -> None:
        if not self.calibration_keys:
            raise ValueError("At least one calibration key is required.")

        if any(not isinstance(key, str) or not key for key in self.calibration_keys):
            raise ValueError("calibration_keys must contain non-empty strings.")

        if len(self.calibration_keys) != len(set(self.calibration_keys)):
            raise ValueError("calibration_keys must be unique.")

        if self.head_mode not in {"individual", "shared"}:
            raise ValueError("head_mode must be either 'individual' or 'shared'.")


@dataclass(frozen=True)
class CalibrationPoolerConfig:
    """Configuration for shared calibration-wise attention pooling."""

    n_heads: int = 8
    dropout: float = 0.0
    calibration_context_dim: int = 0

    def __post_init__(self) -> None:
        _validate_positive("n_heads", self.n_heads)
        _validate_nonnegative("calibration_context_dim", self.calibration_context_dim)
        _validate_dropout(self.dropout)


@dataclass(frozen=True)
class PooledCalibrationHeadConfig:
    """Configuration for the shared calibration prediction head."""

    calibration_context_dim: int = 0
    hidden_dims: tuple[int, ...] = (128, 64)
    activation: str = "gelu"
    dropout: float = 0.0

    def __post_init__(self) -> None:
        _validate_nonnegative("calibration_context_dim", self.calibration_context_dim)

        for index, width in enumerate(self.hidden_dims):
            _validate_positive(f"hidden_dims[{index}]", width)

        _validate_dropout(self.dropout)


@dataclass(frozen=True)
class PooledCalibrationModelConfig:
    """Configuration for global encoding, calibration-wise pooling, and shared prediction."""

    transformer: TransformerConfig
    pooler: CalibrationPoolerConfig
    mlp: MLPConfig
    head: PooledCalibrationHeadConfig

    def __post_init__(self) -> None:
        if self.transformer.d_model % self.pooler.n_heads != 0:
            raise ValueError("Transformer d_model must be divisible by pooler n_heads.")

        active_context_dims = {dim for dim in (self.pooler.calibration_context_dim, self.head.calibration_context_dim) if dim > 0}

        if len(active_context_dims) > 1:
            raise ValueError("Pooler and head calibration_context_dim must match when both use calibration context.")

# ---------------------------------------------------------------------------
# Deterministic calibration-correction Flow Matching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationFlowHeadSettings:
    """Additional FM-head settings; hidden layers are inherited from the pooled head.

    State coordinates use the repository's [rotation (rad), translation (m)]
    Lie-algebra ordering, followed by the time-offset correction in seconds.
    These scales normalize *inputs*, not the physical output velocities.
    """

    rotation_state_scale_rad: float = 1.0
    translation_state_scale_m: float = 0.1
    time_offset_state_scale_s: float = 0.25
    initialize_state_time_independent: bool = True

    def __post_init__(self) -> None:
        import math

        for name in ("rotation_state_scale_rad", "translation_state_scale_m", "time_offset_state_scale_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value}.")
        if not isinstance(self.initialize_state_time_independent, bool):
            raise ValueError("initialize_state_time_independent must be boolean.")


@dataclass(frozen=True)
class CalibrationFlowIntegratorConfig:
    """Deterministic SE(3) x R Lie integration and step-count ablation settings."""

    num_steps: int = 8
    method: str = "midpoint"
    ablation_steps: tuple[int, ...] = (1, 2, 4, 8, 16, 32)

    def __post_init__(self) -> None:
        if isinstance(self.num_steps, bool) or not isinstance(self.num_steps, int) or self.num_steps <= 0:
            raise ValueError("num_steps must be a positive integer.")
        if self.method not in ("euler", "midpoint"):
            raise ValueError("method must be 'euler' or 'midpoint'.")
        if not self.ablation_steps or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in self.ablation_steps):
            raise ValueError("ablation_steps must be a nonempty tuple of positive integers.")
        if len(set(self.ablation_steps)) != len(self.ablation_steps):
            raise ValueError("ablation_steps must be unique.")

    def num_field_evaluations(self, *, steps: int | None = None) -> int:
        """Nominal cost for the configured fixed-step integration scheme."""

        count = self.num_steps if steps is None else steps
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("steps must be a positive integer.")
        return count * (2 if self.method == "midpoint" else 1)


@dataclass(frozen=True)
class CalibrationFlowLossConfig:
    """Velocity-MSE normalization matching the initial notebook-42 scales.

    The source of truth for the loss *implementation* is
    ``obscalib.flow_matching.loss.calibration_velocity_loss``. Conversion to
    its weights object is explicit and lazy to avoid importing torch/geometry
    while loading the architecture configuration.
    """

    rotation_scale_rad: float = 0.08726646259971647  # 5 degrees
    translation_scale_m: float = 0.05
    time_offset_scale_s: float = 0.05
    rotation_weight: float = 2.0
    translation_weight: float = 1.0
    time_offset_weight: float = 1.0
    active_rotation_weight: float = 0.0
    active_rotation_threshold_rad: float = 1e-6

    def __post_init__(self) -> None:
        import math

        for name in ("rotation_scale_rad", "translation_scale_m", "time_offset_scale_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value}.")
        for name in ("rotation_weight", "translation_weight", "time_offset_weight", "active_rotation_weight", "active_rotation_threshold_rad"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative, got {value}.")

    def to_loss_weights(self):
        """Return the exact weights type accepted by calibration_velocity_loss."""

        from obscalib.flow_matching.loss import CalibrationFlowLossWeights

        return CalibrationFlowLossWeights(
            rotation_scale_rad=self.rotation_scale_rad,
            translation_scale_m=self.translation_scale_m,
            time_offset_scale_s=self.time_offset_scale_s,
            rotation_weight=self.rotation_weight,
            translation_weight=self.translation_weight,
            time_offset_weight=self.time_offset_weight,
            active_rotation_weight=self.active_rotation_weight,
            active_rotation_threshold_rad=self.active_rotation_threshold_rad,
        )


@dataclass(frozen=True)
class PooledCalibrationFlowConfig:
    """FM experiment configuration wrapping the existing pooled backbone.

    This is intentionally a companion to PooledCalibrationModelConfig rather
    than a replacement. To preserve checkpoint compatibility, construct the
    flow model using::

        flow_config = PooledCalibrationFlowConfig(backbone=old_model_config)
        model = PooledCalibrationFlowModel(flow_config.backbone, **flow_config.model_kwargs())

    Initialize the shared feature extractor from a pretrained regression model
    separately; ``freeze_backbone_initially`` records that training-stage choice.
    The prior-perturbation distribution (0-60 degrees initially) continues to be
    configured by the augmentation pipeline, not by this model config.
    """

    backbone: PooledCalibrationModelConfig
    head: CalibrationFlowHeadSettings = field(default_factory=CalibrationFlowHeadSettings)
    integrator: CalibrationFlowIntegratorConfig = field(default_factory=CalibrationFlowIntegratorConfig)
    loss: CalibrationFlowLossConfig = field(default_factory=CalibrationFlowLossConfig)
    freeze_backbone_initially: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.backbone, PooledCalibrationModelConfig):
            raise TypeError("backbone must be a PooledCalibrationModelConfig.")
        if not isinstance(self.head, CalibrationFlowHeadSettings):
            raise TypeError("head must be a CalibrationFlowHeadSettings.")
        if not isinstance(self.integrator, CalibrationFlowIntegratorConfig):
            raise TypeError("integrator must be a CalibrationFlowIntegratorConfig.")
        if not isinstance(self.loss, CalibrationFlowLossConfig):
            raise TypeError("loss must be a CalibrationFlowLossConfig.")
        if not isinstance(self.freeze_backbone_initially, bool):
            raise TypeError("freeze_backbone_initially must be boolean.")

    def model_kwargs(self) -> dict[str, int | float | bool | str]:
        """Constructor kwargs for the existing PooledCalibrationFlowModel API."""

        return {
            "integration_steps": self.integrator.num_steps,
            "integration_method": self.integrator.method,
            "rotation_state_scale_rad": self.head.rotation_state_scale_rad,
            "translation_state_scale_m": self.head.translation_state_scale_m,
            "time_offset_state_scale_s": self.head.time_offset_state_scale_s,
            "initialize_state_time_independent": self.head.initialize_state_time_independent,
        }
