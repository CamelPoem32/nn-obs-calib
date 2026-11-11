"""Velocity and differentiable endpoint supervision for 6D rotation Flow Matching."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from obscalib.calibration.state import CalibrationState
from obscalib.data.structures import CalibrationTargetBatch
from obscalib.geometry.rotation_6d import matrix_to_rotation_6d
from obscalib.models.calibration_6d_head import Calibration6DPrediction
from obscalib.training.rotation_6d_loss import Calibration6DLossWeights, calibration_6d_loss


@dataclass(frozen=True)
class Rotation6DFlowLossWeights:
    """Weights with dimensional scales so tau does not swamp rotation by default."""

    velocity_weight: float = 0.1
    endpoint_weight: float = 1.0
    velocity_rotation_6d_scale: float = 1.0
    velocity_translation_scale_m: float = 0.1
    velocity_time_offset_scale_s: float = 0.25
    endpoint: Calibration6DLossWeights = Calibration6DLossWeights()

    def __post_init__(self) -> None:
        if self.velocity_weight < 0 or self.endpoint_weight < 0 or min(self.velocity_rotation_6d_scale, self.velocity_translation_scale_m, self.velocity_time_offset_scale_s) <= 0:
            raise ValueError('Loss weights must be nonnegative and physical scales positive')


@dataclass(frozen=True)
class Rotation6DFlowLossResult:
    total: torch.Tensor
    velocity: torch.Tensor
    velocity_rotation: torch.Tensor
    velocity_translation: torch.Tensor
    velocity_time_offset: torch.Tensor
    endpoint: torch.Tensor
    endpoint_rotation: torch.Tensor
    endpoint_translation: torch.Tensor
    endpoint_time_offset: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {k: float(getattr(self, k).detach()) for k in self.__dataclass_fields__}


def rotation_6d_targets_from_calibration(current: Mapping[str, CalibrationState], targets: Mapping[str, CalibrationTargetBatch]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Full GT correction in the same convention as apply_calibration_6d_update.

    Rotation R_delta = R_gt @ R_prior.T, translation=t_gt-t_prior, tau=tau_gt-tau_prior.
    No SE(3) Log, and no rotation-induced change to parent-frame translation.
    """
    if set(current) != set(targets) or not current:
        raise ValueError('Calibration/GT keys must match and be nonempty')
    rotation, translation, tau = {}, {}, {}
    for key, state in current.items():
        gt = targets[key]
        if gt.next_transform is None or gt.next_time_offset is None:
            raise ValueError(f'Missing full GT calibration target for {key!r}')
        delta_R = gt.next_transform[..., :3, :3] @ state.transform[..., :3, :3].transpose(-1, -2)
        rotation[key] = matrix_to_rotation_6d(delta_R)
        translation[key] = gt.next_transform[..., :3, 3] - state.transform[..., :3, 3]
        tau[key] = gt.next_time_offset - state.time_offset
    return rotation, translation, tau


def rotation_6d_flow_loss(
    predicted_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    *,
    endpoint_predictions: Mapping[str, Calibration6DPrediction] | None = None,
    current_calibration: Mapping[str, CalibrationState] | None = None,
    targets: Mapping[str, CalibrationTargetBatch] | None = None,
    weights: Rotation6DFlowLossWeights | None = None,
) -> Rotation6DFlowLossResult:
    """Train matching velocity and optionally differentiable integrated endpoints.

    Providing endpoint_predictions requires both current_calibration and targets.
    The velocity objective is supervised per coordinate in dimensionless units.
    Endpoint loss uses raw 6D MSE + chordal SO(3) loss and translation/time.
    """
    weights = Rotation6DFlowLossWeights() if weights is None else weights
    if predicted_velocity.shape != target_velocity.shape or predicted_velocity.ndim < 2 or predicted_velocity.shape[-1] != 10:
        raise ValueError('Velocity tensors must have equal shape [...,10]')
    error = predicted_velocity - target_velocity
    rotation = (error[..., :6] / weights.velocity_rotation_6d_scale).square().mean()
    translation = (error[..., 6:9] / weights.velocity_translation_scale_m).square().mean()
    time_offset = (error[..., 9:10] / weights.velocity_time_offset_scale_s).square().mean()
    # Average three physical components instead of summing 10 differently scaled numbers.
    velocity = (rotation + translation + time_offset) / 3.0
    endpoint = predicted_velocity.new_zeros(())
    endpoint_rotation = endpoint
    endpoint_translation = endpoint
    endpoint_time_offset = endpoint
    if endpoint_predictions is not None:
        if current_calibration is None or targets is None:
            raise ValueError('Endpoint requires current_calibration and targets')
        result = calibration_6d_loss(endpoint_predictions, current_calibration, targets, weights=weights.endpoint)
        endpoint, endpoint_rotation, endpoint_translation, endpoint_time_offset = result.total, result.rotation, result.translation, result.time_offset
    elif current_calibration is not None or targets is not None:
        raise ValueError('Endpoint requires endpoint_predictions along with states/targets')
    total = weights.velocity_weight * velocity + weights.endpoint_weight * endpoint
    return Rotation6DFlowLossResult(total, velocity, rotation, translation, time_offset, endpoint, endpoint_rotation, endpoint_translation, endpoint_time_offset)


__all__ = ['Rotation6DFlowLossWeights', 'Rotation6DFlowLossResult', 'rotation_6d_targets_from_calibration', 'rotation_6d_flow_loss']
