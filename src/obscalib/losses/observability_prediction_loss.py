"""Supervised loss for seven-coordinate observability prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from obscalib.models.structures import ObservabilityPredictionOutput


@dataclass
class ObservabilityPredictionLossResult:
    """Total and per-calibration observability losses."""

    total: torch.Tensor
    per_calibration: dict[str, torch.Tensor]


class ObservabilityPredictionLoss(nn.Module):
    """Mean squared error between predicted and target soft-projector coordinates."""

    def forward(self, output: ObservabilityPredictionOutput, targets: dict[str, torch.Tensor]) -> ObservabilityPredictionLossResult:
        if set(output.predictions) != set(targets):
            raise ValueError(f"Prediction and target calibration keys differ: predictions={sorted(output.predictions)}, targets={sorted(targets)}.")

        per_calibration: dict[str, torch.Tensor] = {}

        for calibration_key, prediction in output.predictions.items():
            target = targets[calibration_key].to(device=prediction.device, dtype=prediction.dtype)

            if prediction.ndim != 2 or prediction.shape[-1] != 7:
                raise ValueError(f"Prediction for {calibration_key!r} must have shape [B, 7].")

            if target.shape != prediction.shape:
                raise ValueError(f"Target for {calibration_key!r} must have shape {tuple(prediction.shape)}, got {tuple(target.shape)}.")

            if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
                raise ValueError(f"Prediction and target for {calibration_key!r} must be finite.")

            per_calibration[calibration_key] = F.mse_loss(prediction, target)

        total = torch.stack(tuple(per_calibration.values())).mean()

        return ObservabilityPredictionLossResult(total=total, per_calibration=per_calibration)