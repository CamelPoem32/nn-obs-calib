"""Shared classification head for per-sensor calibration validity.

This head does not produce pose corrections. It predicts three independent binary
logits: an operational calibration fault, an excessive rotation error, and an
excessive time-offset error. Sigmoid is applied only for inference, never before
BCEWithLogitsLoss.
"""

from dataclasses import dataclass

import torch
from torch import nn

from obscalib.config import PooledCalibrationHeadConfig
from obscalib.models.activations import make_activation


@dataclass(frozen=True)
class CalibrationFaultPrediction:
    """Raw binary logits, each with shape [B, 1]."""

    unreliable_logit: torch.Tensor
    rotation_fault_logit: torch.Tensor
    time_fault_logit: torch.Tensor

    @property
    def unreliable_probability(self) -> torch.Tensor:
        return self.unreliable_logit.sigmoid()

    @property
    def rotation_fault_probability(self) -> torch.Tensor:
        return self.rotation_fault_logit.sigmoid()

    @property
    def time_fault_probability(self) -> torch.Tensor:
        return self.time_fault_logit.sigmoid()


class PooledCalibrationClassifierHead(nn.Module):
    """Apply the same classification network to every calibration-key feature.

    The existing PooledCalibrationHeadConfig controls the hidden layers and
    optional numeric context, so no changes to obscalib/config.py are needed.
    """

    def __init__(self, config: PooledCalibrationHeadConfig, *, shared_feature_dim: int) -> None:
        super().__init__()
        if shared_feature_dim <= 0:
            raise ValueError("shared_feature_dim must be positive.")

        self.config = config
        self.shared_feature_dim = shared_feature_dim
        previous_dim = shared_feature_dim + config.calibration_context_dim
        layers: list[nn.Module] = []
        for hidden_dim in config.hidden_dims:
            layers.extend((nn.Linear(previous_dim, hidden_dim), make_activation(config.activation), nn.Dropout(config.dropout)))
            previous_dim = hidden_dim
        self.body = nn.Sequential(*layers) if layers else nn.Identity()

        # Three independent logits: joint operational fault, rotation, time.
        self.unreliable_logit_head = nn.Linear(previous_dim, 1)
        self.rotation_fault_logit_head = nn.Linear(previous_dim, 1)
        self.time_fault_logit_head = nn.Linear(previous_dim, 1)

    def forward(self, shared_features: torch.Tensor, calibration_context: torch.Tensor | None = None) -> CalibrationFaultPrediction:
        if shared_features.ndim != 2 or shared_features.shape[-1] != self.shared_feature_dim:
            raise ValueError(f"shared_features must have shape [B, {self.shared_feature_dim}], got {tuple(shared_features.shape)}.")

        context_dim = self.config.calibration_context_dim
        if context_dim > 0:
            if calibration_context is None or calibration_context.ndim != 2 or calibration_context.shape != (shared_features.shape[0], context_dim):
                raise ValueError(f"calibration_context must have shape {(shared_features.shape[0], context_dim)}.")
            if calibration_context.device != shared_features.device or calibration_context.dtype != shared_features.dtype:
                raise ValueError("calibration_context must match feature device and dtype.")
            head_input = torch.cat((shared_features, calibration_context), dim=-1)
        else:
            if calibration_context is not None and calibration_context.shape != (shared_features.shape[0], 0):
                raise ValueError("With zero-dimensional context, calibration_context must be None or [B, 0].")
            head_input = shared_features

        hidden = self.body(head_input)
        return CalibrationFaultPrediction(
            unreliable_logit=self.unreliable_logit_head(hidden),
            rotation_fault_logit=self.rotation_fault_logit_head(hidden),
            time_fault_logit=self.time_fault_logit_head(hidden),
        )


__all__ = ["CalibrationFaultPrediction", "PooledCalibrationClassifierHead"]
