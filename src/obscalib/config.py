"""Configuration objects for the learned calibration architecture."""

from __future__ import annotations

from dataclasses import dataclass, field


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


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
        for name in (
            "input_dim",
            "d_model",
            "n_heads",
            "num_layers",
            "dim_feedforward",
            "num_summary_tokens",
        ):
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
    """Weights applied to precomputed scalar calibration-loss components."""

    lambda_rotation: float = 1.0
    lambda_translation: float = 1.0
    lambda_time_offset: float = 1.0
    lambda_change: float = 1.0
    lambda_change_time: float = 1.0
    lambda_prior: float = 1.0
    lambda_consistency: float = 1.0


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

