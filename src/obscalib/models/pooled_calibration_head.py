"""Shared calibration head for pooled calibration-wise features."""

import torch
from torch import nn

from obscalib.config import PooledCalibrationHeadConfig
from obscalib.models.activations import make_activation
from obscalib.models.structures import CalibrationPrediction


class PooledCalibrationHead(nn.Module):
    """Predict one calibration update from pooled measurement features and optional numeric calibration context."""

    def __init__(self, config: PooledCalibrationHeadConfig, *, shared_feature_dim: int) -> None:
        super().__init__()

        if shared_feature_dim <= 0:
            raise ValueError("shared_feature_dim must be positive.")

        self.config = config
        self.shared_feature_dim = shared_feature_dim

        input_dim = shared_feature_dim + config.calibration_context_dim
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

    def forward(self, shared_features: torch.Tensor, calibration_context: torch.Tensor | None = None) -> CalibrationPrediction:
        """Predict one calibration correction using the same shared head for every calibration key."""

        if shared_features.ndim != 2 or shared_features.shape[-1] != self.shared_feature_dim:
            raise ValueError(f"shared_features must have shape [B, {self.shared_feature_dim}], got {tuple(shared_features.shape)}.")

        if self.config.calibration_context_dim > 0:
            if calibration_context is None:
                raise ValueError("calibration_context is required when calibration_context_dim > 0.")

            if calibration_context.ndim != 2 or calibration_context.shape[-1] != self.config.calibration_context_dim:
                raise ValueError(f"calibration_context must have shape [B, {self.config.calibration_context_dim}], got {tuple(calibration_context.shape)}.")

            if shared_features.shape[0] != calibration_context.shape[0]:
                raise ValueError("shared_features and calibration_context must share batch size.")

            if shared_features.device != calibration_context.device:
                raise ValueError("shared_features and calibration_context must be on the same device.")

            if shared_features.dtype != calibration_context.dtype:
                raise ValueError("shared_features and calibration_context must use the same dtype.")

            head_input = torch.cat((shared_features, calibration_context), dim=-1)

        else:
            if calibration_context is not None and calibration_context.shape != (shared_features.shape[0], 0):
                raise ValueError(f"calibration_context must be None or have shape {(shared_features.shape[0], 0)} when calibration_context_dim == 0.")

            head_input = shared_features

        hidden = self.body(head_input)

        return CalibrationPrediction(
            change_event_logit=self.change_event_logit_head(hidden),
            change_time=self.change_time_head(hidden),
            delta_xi=self.delta_xi_head(hidden),
            delta_tau=self.delta_tau_head(hidden),
        )


__all__ = ["PooledCalibrationHead"]