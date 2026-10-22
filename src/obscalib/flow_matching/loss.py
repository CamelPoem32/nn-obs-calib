"""Direct conditional-flow-matching velocity supervision for SE(3) x R.

No ODE integration or endpoint loss is used during primary training. The target
is the exact left-trivialized velocity along the analytic exponential curve.
Component scales and default weights mirror notebook 42's regression loss.
"""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CalibrationFlowLossWeights:
    """Physical normalization and component weights for velocity MSE.

    Set active_rotation_weight=0.75 to reproduce notebook 42's additional
    active-rotation emphasis. It defaults to zero for a clean FM baseline.
    """

    rotation_scale_rad: float = math.radians(5.0)
    translation_scale_m: float = 0.05
    time_offset_scale_s: float = 0.05
    rotation_weight: float = 2.0
    translation_weight: float = 1.0
    time_offset_weight: float = 1.0
    active_rotation_weight: float = 0.0
    active_rotation_threshold_rad: float = 1e-6

    def __post_init__(self) -> None:
        for name in ("rotation_scale_rad", "translation_scale_m", "time_offset_scale_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("rotation_weight", "translation_weight", "time_offset_weight", "active_rotation_weight", "active_rotation_threshold_rad"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")


@dataclass(frozen=True)
class CalibrationFlowLoss:
    """Scalar training loss and differentiable component losses."""

    total: torch.Tensor
    rotation: torch.Tensor
    translation: torch.Tensor
    time_offset: torch.Tensor
    active_rotation: torch.Tensor
    num_active_rotations: int

    def detached_metrics(self) -> dict[str, float]:
        """Convert loss components into simple numbers for training logs."""

        return {
            "loss_fm_total": self.total.detach().item(),
            "loss_fm_rotation": self.rotation.detach().item(),
            "loss_fm_translation": self.translation.detach().item(),
            "loss_fm_time_offset": self.time_offset.detach().item(),
            "loss_fm_active_rotation": self.active_rotation.detach().item(),
            "fm_active_rotation_count": float(self.num_active_rotations),
        }


def calibration_velocity_loss(predicted_velocity: torch.Tensor, target_velocity: torch.Tensor, *, weights: CalibrationFlowLossWeights | None = None, active_rotation_mask: torch.Tensor | None = None) -> CalibrationFlowLoss:
    """Weighted componentwise MSE on physical velocity vectors of shape [..., 7].

    The six algebra dimensions are [rotation (3), translation (3)] and the final
    component is time offset in seconds. The active mask, if supplied, is [...].
    An empty active subset contributes zero without producing NaNs.
    """

    if predicted_velocity.ndim < 2 or predicted_velocity.shape[-1] != 7 or predicted_velocity.shape != target_velocity.shape:
        raise ValueError("predicted_velocity and target_velocity must have matching shapes [..., 7].")
    if not predicted_velocity.is_floating_point() or predicted_velocity.device != target_velocity.device or predicted_velocity.dtype != target_velocity.dtype:
        raise ValueError("velocity tensors must use the same floating dtype and device.")
    if predicted_velocity.numel() == 0:
        raise ValueError("velocity tensors must not be empty.")
    if weights is None:
        weights = CalibrationFlowLossWeights()

    errors = predicted_velocity - target_velocity
    rotation_per_sample = (errors[..., :3] / weights.rotation_scale_rad).square().mean(dim=-1)
    translation_per_sample = (errors[..., 3:6] / weights.translation_scale_m).square().mean(dim=-1)
    time_per_sample = (errors[..., 6] / weights.time_offset_scale_s).square()
    rotation = rotation_per_sample.mean()
    translation = translation_per_sample.mean()
    time_offset = time_per_sample.mean()

    if active_rotation_mask is None:
        active_rotation_mask = torch.linalg.vector_norm(target_velocity[..., :3], dim=-1) > weights.active_rotation_threshold_rad
    elif active_rotation_mask.shape != target_velocity.shape[:-1] or active_rotation_mask.dtype != torch.bool or active_rotation_mask.device != target_velocity.device:
        raise ValueError("active_rotation_mask must be a boolean tensor of shape [...] on the same device.")

    active_count = int(active_rotation_mask.sum().item())
    active_rotation = rotation_per_sample[active_rotation_mask].mean() if active_count else rotation_per_sample.sum() * 0.0
    total = weights.rotation_weight * rotation + weights.translation_weight * translation + weights.time_offset_weight * time_offset + weights.active_rotation_weight * weights.rotation_weight * active_rotation
    return CalibrationFlowLoss(total=total, rotation=rotation, translation=translation, time_offset=time_offset, active_rotation=active_rotation, num_active_rotations=active_count)


__all__ = ["CalibrationFlowLossWeights", "CalibrationFlowLoss", "calibration_velocity_loss"]
