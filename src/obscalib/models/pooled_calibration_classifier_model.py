"""Calibration-fault classifier with checkpoint-compatible pooled backbone.

The feature extractor is inherited from PooledCalibrationModel. Only the
regression head is replaced, leaving transformer_encoder/pooler/mlp_body
unchanged and transferable from either pooled regression variant (including
002.27's PooledCalibration6DModel when it shares these module interfaces).
"""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from obscalib.config import PooledCalibrationModelConfig
from obscalib.data.structures import TokenBatch
from obscalib.models.pooled_calibration_classifier_head import CalibrationFaultPrediction, PooledCalibrationClassifierHead
from obscalib.models.pooled_calibration_model import PooledCalibrationModel


BACKBONE_MODULE_NAMES = ("transformer_encoder", "pooler", "mlp_body")


@dataclass(frozen=True)
class CalibrationClassifierOutput:
    """Per-key predictions and stacked logits in the TokenBatch calibration order.

    logits has shape [B, C, 3], with its last dimension ordered as:
    [operational/unreliable, rotation fault, time-offset fault].
    """

    predictions: dict[str, CalibrationFaultPrediction]
    shared_features: torch.Tensor
    calibration_keys: tuple[str, ...]
    logits: torch.Tensor

    @property
    def probabilities(self) -> torch.Tensor:
        return self.logits.sigmoid()


class PooledCalibrationClassifierModel(PooledCalibrationModel):
    """Reuse the pooled regression feature path, but predict fault logits only."""

    def __init__(self, config: PooledCalibrationModelConfig) -> None:
        # Parent creates the exact shared feature extractor of the regression
        # model. Its original regression head is then discarded, not retained.
        super().__init__(config)
        self.head = PooledCalibrationClassifierHead(config.head, shared_feature_dim=config.mlp.output_dim)

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze feature extraction; the new classifier head is unaffected."""
        for name in BACKBONE_MODULE_NAMES:
            getattr(self, name).requires_grad_(trainable)

    def forward(self, tokens: TokenBatch, calibration_context: Mapping[str, torch.Tensor] | None = None) -> CalibrationClassifierOutput:
        shared, stacked_context = self.encode_calibration_features(tokens, calibration_context)
        batch_size, num_calibrations, feature_dim = shared.shape
        flat_shared = shared.reshape(batch_size * num_calibrations, feature_dim)
        flat_context = self.flatten_head_context(shared, stacked_context)
        flat = self.head(flat_shared, flat_context)
        logits = torch.stack((flat.unreliable_logit, flat.rotation_fault_logit, flat.time_fault_logit), dim=-1).reshape(batch_size, num_calibrations, 3)

        calibration_keys = tuple(tokens.calibration_keys)
        if len(calibration_keys) != num_calibrations:
            raise ValueError("Calibration keys and pooled feature count disagree.")
        predictions = {
            key: CalibrationFaultPrediction(
                unreliable_logit=logits[:, idx, 0:1],
                rotation_fault_logit=logits[:, idx, 1:2],
                time_fault_logit=logits[:, idx, 2:3],
            )
            for idx, key in enumerate(calibration_keys)
        }
        return CalibrationClassifierOutput(predictions=predictions, shared_features=shared, calibration_keys=calibration_keys, logits=logits)


def initialize_classifier_from_regressor(classifier: PooledCalibrationClassifierModel, regressor: nn.Module, *, copy_head_body: bool = False) -> None:
    """Copy all backbone tensors strictly, without trying to load regression heads.

    The pretrained regressor must already be instantiated and checkpoint-loaded
    with its OWN original architecture. This supports 002.27's 6D correction
    model and the newer Lie-vector model, provided their backbone modules share
    the names and parameter structure of the supplied pooled model.

    Every backbone tensor must match. Refuse partial transfers: accidentally
    leaving random Transformer layers would invalidate the experiment.
    """

    for name in BACKBONE_MODULE_NAMES:
        source_module = getattr(regressor, name, None)
        destination_module = getattr(classifier, name)
        if not isinstance(source_module, nn.Module):
            raise TypeError(f"Source regressor is missing required module {name!r}.")
        source_state = source_module.state_dict()
        destination_state = destination_module.state_dict()
        if source_state.keys() != destination_state.keys():
            missing = sorted(destination_state.keys() - source_state.keys())
            extra = sorted(source_state.keys() - destination_state.keys())
            raise ValueError(f"Incompatible {name} state keys: missing={missing[:10]}, extra={extra[:10]}.")
        mismatched = [key for key in source_state if source_state[key].shape != destination_state[key].shape]
        if mismatched:
            raise ValueError(f"Incompatible {name} state shapes for {mismatched[:10]}.")
        destination_module.load_state_dict(source_state, strict=True)

    if copy_head_body:
        source_head = getattr(regressor, "head", None)
        source_body = getattr(source_head, "body", None)
        if not isinstance(source_body, nn.Module):
            raise TypeError("copy_head_body=True, but the regression head has no .body module.")
        classifier.head.body.load_state_dict(source_body.state_dict(), strict=True)


def calibration_fault_loss(
    output: CalibrationClassifierOutput,
    rotation_fault_targets: torch.Tensor,
    time_fault_targets: torch.Tensor,
    *,
    joint_fault_targets: torch.Tensor | None = None,
    joint_weight: float = 1.0,
    rotation_weight: float = 0.5,
    time_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Multi-task BCE loss for binary fault labels of shape [B, C].

    If joint_fault_targets is supplied, it supersedes the default per-sensor
    rotation/time OR and may represent a broader majority-based trust policy.
    Rotation and time targets always describe each sensor's actual faults.
    All three targets must be binary 0/1 and share the logits device/shape.
    """

    logits = output.logits
    expected_shape = logits.shape[:2]
    if rotation_fault_targets.shape != expected_shape or time_fault_targets.shape != expected_shape:
        raise ValueError(f"Fault targets must both have shape {tuple(expected_shape)}.")
    if min(joint_weight, rotation_weight, time_weight) < 0 or joint_weight + rotation_weight + time_weight <= 0:
        raise ValueError("Loss weights must be nonnegative with a positive total.")

    rotation_targets = rotation_fault_targets.to(device=logits.device, dtype=logits.dtype)
    time_targets = time_fault_targets.to(device=logits.device, dtype=logits.dtype)
    if not bool(torch.all((rotation_targets == 0) | (rotation_targets == 1))) or not bool(torch.all((time_targets == 0) | (time_targets == 1))):
        raise ValueError("Fault targets must be binary 0/1 values.")

    if joint_fault_targets is None:
        joint_targets = torch.maximum(rotation_targets, time_targets)
    else:
        if joint_fault_targets.shape != expected_shape:
            raise ValueError(f"joint_fault_targets must have shape {tuple(expected_shape)}.")
        joint_targets = joint_fault_targets.to(device=logits.device, dtype=logits.dtype)
        if not bool(torch.all((joint_targets == 0) | (joint_targets == 1))):
            raise ValueError("joint_fault_targets must contain only binary 0/1 values.")

    losses = {
        "joint": F.binary_cross_entropy_with_logits(logits[..., 0], joint_targets),
        "rotation": F.binary_cross_entropy_with_logits(logits[..., 1], rotation_targets),
        "time": F.binary_cross_entropy_with_logits(logits[..., 2], time_targets),
    }
    total = joint_weight * losses["joint"] + rotation_weight * losses["rotation"] + time_weight * losses["time"]
    return total, losses


__all__ = ["CalibrationClassifierOutput", "PooledCalibrationClassifierModel", "initialize_classifier_from_regressor", "calibration_fault_loss"]
