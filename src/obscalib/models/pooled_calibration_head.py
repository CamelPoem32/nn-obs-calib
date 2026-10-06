"""Shared deterministic calibration head for pooled calibration features."""

import torch
from torch import nn

from obscalib.config import PooledCalibrationHeadConfig
from obscalib.models.activations import make_activation
from obscalib.models.structures import CalibrationPrediction


class PooledCalibrationHead(nn.Module):
    """Predict one calibration correction from one pooled calibration feature."""

    def __init__(self, config: PooledCalibrationHeadConfig, input_dim: int) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        previous_dim = input_dim

        for hidden_dim in config.hidden_dims:
            layers.extend((nn.Linear(previous_dim, hidden_dim), make_activation(config.activation), nn.Dropout(config.dropout)))
            previous_dim = hidden_dim

        self.body = nn.Sequential(*layers) if layers else nn.Identity()
        self.change_event_logit_head = nn.Linear(previous_dim, 1)
        self.change_time_head = nn.Linear(previous_dim, 1)
        self.delta_xi_head = nn.Linear(previous_dim, 6)
        self.delta_tau_head = nn.Linear(previous_dim, 1)

    def forward(self, features: torch.Tensor) -> CalibrationPrediction:
        hidden = self.body(features)

        return CalibrationPrediction(
            change_event_logit=self.change_event_logit_head(hidden),
            change_time=self.change_time_head(hidden),
            delta_xi=self.delta_xi_head(hidden),
            delta_tau=self.delta_tau_head(hidden),
        )