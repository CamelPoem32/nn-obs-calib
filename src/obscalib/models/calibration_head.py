"""Reusable head producing one sensor's deterministic calibration outputs."""

import torch
from torch import nn

from obscalib.config import CalibrationHeadConfig
from obscalib.models.structures import CalibrationPrediction
from obscalib.models.activations import make_activation


class CalibrationHead(nn.Module):
    """Combine shared features with an externally prepared state context."""

    def __init__(
        self,
        config: CalibrationHeadConfig,
        *,
        shared_feature_dim: int,
    ) -> None:
        super().__init__()
        if shared_feature_dim <= 0:
            raise ValueError("shared_feature_dim must be positive.")
        self.config = config
        self.shared_feature_dim = shared_feature_dim
        input_dim = shared_feature_dim + config.calibration_context_dim
        layers: list[nn.Module] = []
        previous_dim = input_dim

        # Build the sensor-specific hidden representation.
        for hidden_dim in config.hidden_dims:
            layers.extend(
                (
                    nn.Linear(previous_dim, hidden_dim),
                    make_activation(config.activation),
                    nn.Dropout(config.dropout),
                )
            )
            previous_dim = hidden_dim
        self.body = nn.Sequential(*layers) if layers else nn.Identity()

        # Separate named heads preserve the semantics of each deterministic output.
        self.change_event_logit_head = nn.Linear(previous_dim, 1)
        self.change_time_head = nn.Linear(previous_dim, 1)
        self.delta_xi_head = nn.Linear(previous_dim, 6)
        self.delta_tau_head = nn.Linear(previous_dim, 1)
        # TODO: Add an uncertainty head only after its training semantics are fixed.

    def forward(
        self,
        shared_features: torch.Tensor,
        calibration_context: torch.Tensor,
    ) -> CalibrationPrediction:
        """Predict calibration changes for one configured sensor or sensor type."""

        if (
            shared_features.ndim != 2
            or shared_features.shape[-1] != self.shared_feature_dim
        ):
            raise ValueError(
                "shared_features must have shape "
                f"[B, {self.shared_feature_dim}]."
            )
        if (
            calibration_context.ndim != 2
            or calibration_context.shape[-1] != self.config.calibration_context_dim
        ):
            raise ValueError(
                "calibration_context must have shape "
                f"[B, {self.config.calibration_context_dim}]."
            )
        if shared_features.shape[0] != calibration_context.shape[0]:
            raise ValueError("shared_features and calibration_context must share B.")
        # shared_features:     [B, d_shared]
        # calibration_context: [B, d_calib]
        # combined:            [B, d_shared + d_calib]
        combined = torch.cat((shared_features, calibration_context), dim=-1)
        hidden = self.body(combined)
        return CalibrationPrediction(
            change_event_logit=self.change_event_logit_head(hidden),
            change_time=self.change_time_head(hidden),
            delta_xi=self.delta_xi_head(hidden),
            delta_tau=self.delta_tau_head(hidden),
        )
