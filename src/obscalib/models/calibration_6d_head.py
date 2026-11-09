"""Direct 6D-rotation, 3D-translation, 1D-time calibration regression head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from obscalib.config import PooledCalibrationHeadConfig
from obscalib.geometry.rotation_6d import rotation_6d_to_matrix


def _activation(name: str) -> nn.Module:
    options = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh, "elu": nn.ELU}
    if name.lower() not in options:
        raise ValueError(f"Unsupported activation {name!r}; expected one of {tuple(options)}")
    return options[name.lower()]()


@dataclass
class Calibration6DPrediction:
    """Per-sensor correction. Translation is additive in the calibration parent frame."""

    rotation_6d: torch.Tensor       # [B, 6], two columns of left correction R_delta
    delta_translation: torch.Tensor  # [B, 3], parent-frame additive translation
    delta_tau: torch.Tensor          # [B, 1], additive time offset

    @property
    def delta_rotation(self) -> torch.Tensor:
        """Proper [B, 3, 3] left-composed SO(3) correction."""
        return rotation_6d_to_matrix(self.rotation_6d)


class Calibration6DHead(nn.Module):
    """Shared calibration-wise head compatible with the existing pooled features.

    Uses the hidden dimensions, activation and calibration-context dimension from
    the existing ``PooledCalibrationHeadConfig``. The last layer is initialized to
    produce the identity correction and zero translation/time shift.
    """

    def __init__(self, config: PooledCalibrationHeadConfig, *, shared_feature_dim: int) -> None:
        super().__init__()
        if shared_feature_dim <= 0:
            raise ValueError("shared_feature_dim must be positive")
        self.calibration_context_dim = int(config.calibration_context_dim)
        if self.calibration_context_dim < 0:
            raise ValueError("calibration_context_dim must be nonnegative")

        layers: list[nn.Module] = []
        current_dim = shared_feature_dim + self.calibration_context_dim
        for hidden_dim in config.hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("head hidden dimensions must be positive")
            layers.extend((nn.Linear(current_dim, hidden_dim), _activation(config.activation)))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            current_dim = hidden_dim

        self.body = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, 10)
        self.reset_output_parameters()

    def reset_output_parameters(self) -> None:
        """Start near identity, without exactly zero gradient to hidden features."""
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.output_layer.bias)
        with torch.no_grad():
            self.output_layer.bias[:6].copy_(self.output_layer.bias.new_tensor((1.0, 0.0, 0.0, 0.0, 1.0, 0.0)))

    def forward(self, shared_features: torch.Tensor, calibration_context: torch.Tensor | None = None) -> Calibration6DPrediction:
        if shared_features.ndim != 2:
            raise ValueError("shared_features must have shape [B*C, H]")
        n = shared_features.shape[0]
        if self.calibration_context_dim:
            if calibration_context is None or calibration_context.shape != (n, self.calibration_context_dim):
                raise ValueError(f"calibration_context must have shape {(n, self.calibration_context_dim)}")
            features = torch.cat((shared_features, calibration_context), dim=-1)
        else:
            features = shared_features
        output = self.output_layer(self.body(features))
        return Calibration6DPrediction(rotation_6d=output[:, :6], delta_translation=output[:, 6:9], delta_tau=output[:, 9:10])


__all__ = ["Calibration6DPrediction", "Calibration6DHead"]
