"""Velocity head for Euclidean 6D rotation + translation + time correction flow."""

from __future__ import annotations

import torch
from torch import nn

from obscalib.config import PooledCalibrationHeadConfig
from obscalib.flow_matching.rotation_6d_path import IDENTITY_ROTATION_6D


def _activation(name: str) -> nn.Module:
    options = {'gelu': nn.GELU, 'relu': nn.ReLU, 'silu': nn.SiLU, 'tanh': nn.Tanh, 'elu': nn.ELU}
    if name.lower() not in options:
        raise ValueError(f'Unsupported head activation {name!r}')
    return options[name.lower()]()


class PooledCalibrationFlow6DHead(nn.Module):
    """Shared head: [features, context, normalized 10D state, time] -> 10D velocity.

    The predicted velocity is in physical embedding/m/s units. Input normalization
    is only for conditioning, not a change of the ODE state or endpoint.
    """

    def __init__(self, config: PooledCalibrationHeadConfig, *, shared_feature_dim: int, translation_state_scale_m: float = 0.1, time_offset_state_scale_s: float = 0.25) -> None:
        super().__init__()
        if shared_feature_dim <= 0 or translation_state_scale_m <= 0 or time_offset_state_scale_s <= 0:
            raise ValueError('Feature dimension and state normalization scales must be positive')
        self.calibration_context_dim = int(config.calibration_context_dim)
        self.translation_state_scale_m = float(translation_state_scale_m)
        self.time_offset_state_scale_s = float(time_offset_state_scale_s)
        input_dim = shared_feature_dim + self.calibration_context_dim + 10 + 1
        layers: list[nn.Module] = []
        for hidden_dim in config.hidden_dims:
            if hidden_dim <= 0:
                raise ValueError('Head hidden dimensions must be positive')
            layers.extend((nn.Linear(input_dim, hidden_dim), _activation(config.activation)))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            input_dim = hidden_dim
        self.body = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_dim, 10)
        # Unlike older 1e-4 heads, allow useful gradients into pretrained features.
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.output_layer.bias)
        self.register_buffer('identity_rotation_6d', torch.tensor(IDENTITY_ROTATION_6D), persistent=False)

    def forward(self, shared_features: torch.Tensor, state: torch.Tensor, time: torch.Tensor, calibration_context: torch.Tensor | None = None) -> torch.Tensor:
        if shared_features.ndim != 2 or state.shape != (len(shared_features), 10) or time.shape != (len(shared_features), 1):
            raise ValueError('Expected shared features [N,H], state [N,10], time [N,1]')
        if self.calibration_context_dim:
            if calibration_context is None or calibration_context.shape != (len(shared_features), self.calibration_context_dim):
                raise ValueError('calibration_context must have shape [N,head.calibration_context_dim]')
        else:
            calibration_context = shared_features.new_empty((len(shared_features), 0))
        normalized_state = torch.cat((state[:, :6] - self.identity_rotation_6d.to(dtype=state.dtype), state[:, 6:9] / self.translation_state_scale_m, state[:, 9:10] / self.time_offset_state_scale_s), dim=-1)
        features = torch.cat((shared_features, calibration_context, normalized_state, time), dim=-1)
        return self.output_layer(self.body(features))


__all__ = ['PooledCalibrationFlow6DHead']
