"""Direct supervised full-correction loss for the new 6D rotation regressor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from obscalib.calibration.state import CalibrationState
from obscalib.data.structures import CalibrationTargetBatch
from obscalib.geometry.rotation_6d import matrix_to_rotation_6d
from obscalib.models.calibration_6d_head import Calibration6DPrediction


@dataclass(frozen=True)
class Calibration6DLossWeights:
    """Dimensionless loss weights; translation/time are scaled by physical units."""

    rotation_weight: float = 2.0
    rotation_6d_weight: float = 1.0
    rotation_chordal_weight: float = 1.0
    translation_weight: float = 1.0
    time_offset_weight: float = 1.0
    translation_scale_m: float = 0.05
    time_offset_scale_s: float = 0.05

    def __post_init__(self) -> None:
        values = (self.rotation_weight, self.rotation_6d_weight, self.rotation_chordal_weight, self.translation_weight, self.time_offset_weight)
        if any(value < 0 for value in values) or self.translation_scale_m <= 0 or self.time_offset_scale_s <= 0:
            raise ValueError("Loss weights must be nonnegative and physical scales positive")


@dataclass
class Calibration6DLossResult:
    total: torch.Tensor
    rotation: torch.Tensor
    rotation_6d: torch.Tensor
    rotation_chordal: torch.Tensor
    translation: torch.Tensor
    time_offset: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {key: float(getattr(self, key).detach()) for key in ("total", "rotation", "rotation_6d", "rotation_chordal", "translation", "time_offset")}


def calibration_6d_loss(predictions: Mapping[str, Calibration6DPrediction], current_calibration: Mapping[str, CalibrationState], targets: Mapping[str, CalibrationTargetBatch], *, weights: Calibration6DLossWeights | None = None) -> Calibration6DLossResult:
    """Supervise full correction on SO(3), parent-frame translation and time offset.

    Rotation target: ``R_gt @ R_prior.T``. Rotation is trained with both direct
    continuous 6D regression and squared chordal matrix distance. The chordal term
    has no acos/log cut-locus singularity. No bounded 30-degree teacher is used.
    """
    weights = Calibration6DLossWeights() if weights is None else weights
    if not predictions or set(predictions) != set(current_calibration) or set(predictions) != set(targets):
        raise ValueError("Prediction, current-calibration and target keys must match and be nonempty")

    rot6d_terms, chordal_terms, translation_terms, tau_terms = [], [], [], []
    for key, prediction in predictions.items():
        prior, target = current_calibration[key], targets[key]
        if target.next_transform is None or target.next_time_offset is None:
            raise ValueError(f"Full GT target for {key!r} must provide transform and time offset")
        R_prior = prior.transform[..., :3, :3]
        R_gt = target.next_transform[..., :3, :3]
        R_delta_gt = R_gt @ R_prior.transpose(-1, -2)
        R_delta_pred = prediction.delta_rotation
        target_6d = matrix_to_rotation_6d(R_delta_gt)

        rot6d_terms.append((prediction.rotation_6d - target_6d).square().mean(dim=-1))
        chordal_terms.append(0.5 * (R_delta_pred - R_delta_gt).square().sum(dim=(-2, -1)))
        target_translation = target.next_transform[..., :3, 3] - prior.transform[..., :3, 3]
        translation_terms.append(((prediction.delta_translation - target_translation) / weights.translation_scale_m).square().mean(dim=-1))
        target_tau = target.next_time_offset - prior.time_offset
        tau_terms.append(((prediction.delta_tau - target_tau) / weights.time_offset_scale_s).square().mean(dim=-1))

    # Stack equally-sized [B] terms into [B, C] and average across windows/sensors.
    rotation_6d = torch.stack(rot6d_terms, dim=-1).mean()
    rotation_chordal = torch.stack(chordal_terms, dim=-1).mean()
    translation = torch.stack(translation_terms, dim=-1).mean()
    time_offset = torch.stack(tau_terms, dim=-1).mean()
    rotation = weights.rotation_6d_weight * rotation_6d + weights.rotation_chordal_weight * rotation_chordal
    total = weights.rotation_weight * rotation + weights.translation_weight * translation + weights.time_offset_weight * time_offset
    return Calibration6DLossResult(total=total, rotation=rotation, rotation_6d=rotation_6d, rotation_chordal=rotation_chordal, translation=translation, time_offset=time_offset)


__all__ = ["Calibration6DLossWeights", "Calibration6DLossResult", "calibration_6d_loss"]
