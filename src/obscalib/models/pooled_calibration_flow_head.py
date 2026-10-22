"""Shared calibration-wise Lie-group velocity head for deterministic flow matching.

The input state is the *integrated correction from identity*, not an absolute
calibration. The 7D output is a physical left-trivialized velocity with ordering
[rotation (rad), translation-algebra (m), clock offset (s)].
"""

import math

import torch
from torch import nn

from obscalib.config import PooledCalibrationHeadConfig
from obscalib.models.activations import make_activation


class PooledCalibrationFlowHead(nn.Module):
    """Predict one SE(3) x R correction velocity per flattened calibration key.

    Feature and calibration-context dimensions come from the existing pooled
    model configuration. The input state is [delta_xi (6), delta_tau (1)] and
    time is a separate scalar. State scales are for *head inputs only*; the
    returned 7-vector stays in physical units for integration and FM loss.
    """

    def __init__(
        self,
        config: PooledCalibrationHeadConfig,
        *,
        shared_feature_dim: int,
        rotation_state_scale_rad: float = 1.0,
        translation_state_scale_m: float = 0.1,
        time_offset_state_scale_s: float = 0.25,
        initialize_state_time_independent: bool = True,
    ) -> None:
        super().__init__()
        if shared_feature_dim <= 0:
            raise ValueError("shared_feature_dim must be positive.")
        for name, scale in (
            ("rotation_state_scale_rad", rotation_state_scale_rad),
            ("translation_state_scale_m", translation_state_scale_m),
            ("time_offset_state_scale_s", time_offset_state_scale_s),
        ):
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"{name} must be finite and positive.")

        self.config = config
        self.shared_feature_dim = shared_feature_dim
        self.condition_dim = shared_feature_dim + config.calibration_context_dim
        self.state_dim = 7
        self.input_dim = self.condition_dim + self.state_dim + 1
        self.register_buffer("state_scales", torch.tensor([rotation_state_scale_rad] * 3 + [translation_state_scale_m] * 3 + [time_offset_state_scale_s]))

        layers: list[nn.Module] = []
        previous_dim = self.input_dim
        for hidden_dim in config.hidden_dims:
            layers.extend((nn.Linear(previous_dim, hidden_dim), make_activation(config.activation), nn.Dropout(config.dropout)))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 7))
        self.net = nn.Sequential(*layers)

        if initialize_state_time_independent:
            self.initialize_without_state_time()

    @torch.no_grad()
    def initialize_without_state_time(self) -> None:
        """Keep random feature weights, but zero the initial dependence on D and t.

        A subsequent gradient update can learn nonzero state/time weights. This
        deliberately does *not* copy the old regression head or zero the output.
        """

        first_linear = next(layer for layer in self.net if isinstance(layer, nn.Linear))
        first_linear.weight[:, self.condition_dim:].zero_()

    def forward(
        self,
        shared_features: torch.Tensor,
        correction_xi: torch.Tensor,
        correction_tau: torch.Tensor,
        time: torch.Tensor,
        calibration_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the physical velocity [N, 7] from flattened [N, ...] inputs."""

        if shared_features.ndim != 2 or shared_features.shape[-1] != self.shared_feature_dim:
            raise ValueError(f"shared_features must be [N, {self.shared_feature_dim}].")
        n = shared_features.shape[0]
        for name, value, width in (("correction_xi", correction_xi, 6), ("correction_tau", correction_tau, 1), ("time", time, 1)):
            if value.shape != (n, width):
                raise ValueError(f"{name} must have shape {(n, width)}; got {tuple(value.shape)}.")
            if value.device != shared_features.device or value.dtype != shared_features.dtype:
                raise ValueError(f"{name} must match shared_features dtype and device.")

        if self.config.calibration_context_dim > 0:
            expected_shape = (n, self.config.calibration_context_dim)
            if calibration_context is None or calibration_context.shape != expected_shape:
                raise ValueError(f"calibration_context must have shape {expected_shape}.")
            if calibration_context.device != shared_features.device or calibration_context.dtype != shared_features.dtype:
                raise ValueError("calibration_context must match shared_features dtype and device.")
            condition = torch.cat((shared_features, calibration_context), dim=-1)
        else:
            if calibration_context is not None and calibration_context.shape != (n, 0):
                raise ValueError(f"calibration_context must be None or have shape {(n, 0)}.")
            condition = shared_features

        normalized_state = torch.cat((correction_xi, correction_tau), dim=-1) / self.state_scales
        return self.net(torch.cat((condition, normalized_state, time), dim=-1))


__all__ = ["PooledCalibrationFlowHead"]
