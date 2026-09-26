"""Supervised geometric and temporal calibration losses."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from obscalib.calibration import CalibrationState
from obscalib.config import CalibrationLossConfig
from obscalib.data.structures import CalibrationTargetBatch
from obscalib.geometry import se3_log, so3_exp, so3_log
from obscalib.losses.structures import LossComponents, combine_loss_components
from obscalib.models.structures import CalibrationPrediction
from obscalib.observability.structures import ObservabilityResult


class CalibrationLoss(nn.Module):
    """
    Compute supervised calibration losses for one minibatch.

    Current and next GT calibration states define the target correction using

        delta_T_gt = T_next @ inverse(T_current)
        delta_xi_gt = Log_SE3(delta_T_gt)

    with the established convention

        delta_xi = [phi, rho].

    Spatial and temporal correction losses are applied on both change and
    no-change windows. Therefore no-change samples explicitly teach the network
    to predict corrections close to zero.

    change_time is supervised only on windows whose change_label is true.

    Consistency loss is currently disabled and returned as zero.
    """

    def __init__(self, config: CalibrationLossConfig) -> None:
        super().__init__()

        self.config = config

    def forward(self, predictions: Mapping[str, CalibrationPrediction], current_calibration: Mapping[str, CalibrationState], targets: Mapping[str, CalibrationTargetBatch], observability: ObservabilityResult | None = None) -> LossComponents:
        """Compute and aggregate losses over all calibration prediction heads."""

        calibration_keys = self._validate_keys(predictions, current_calibration, targets)

        if self.config.correction_loss_mode == "observability_aware":
            if observability is None:
                raise ValueError("observability must be provided when correction_loss_mode='observability_aware'.")

            if observability.calibration_features is None:
                raise ValueError("observability.calibration_features must be populated when correction_loss_mode='observability_aware'.")

        rotation_losses: list[torch.Tensor] = []
        translation_losses: list[torch.Tensor] = []
        time_offset_losses: list[torch.Tensor] = []
        change_event_losses: list[torch.Tensor] = []
        change_time_losses: list[torch.Tensor] = []

        for calibration_key in calibration_keys:
            prediction = predictions[calibration_key]
            current_state = current_calibration[calibration_key]
            target = targets[calibration_key]

            current_state.validate()
            target.validate()

            next_transform, next_time_offset, change_label, change_time = self._require_target_fields(calibration_key, target)
            self._validate_prediction_and_target_shapes(calibration_key, prediction, current_state, next_transform, next_time_offset, change_label, change_time)

            target_delta_transform = next_transform @ torch.linalg.inv(current_state.transform)
            target_delta_xi = se3_log(target_delta_transform)
            target_delta_tau = next_time_offset - current_state.time_offset

            predicted_phi = prediction.delta_xi[..., :3]
            predicted_rho = prediction.delta_xi[..., 3:]
            target_phi = target_delta_xi[..., :3]
            target_rho = target_delta_xi[..., 3:]

            if self.config.correction_loss_mode == "standard":
                rotation_losses.append(self._rotation_loss(predicted_phi, target_phi))
                translation_losses.append(self._translation_loss(predicted_rho, target_rho))
                time_offset_losses.append(self._time_offset_loss(prediction.delta_tau, target_delta_tau))
            else:
                calibration_observability = self._require_calibration_observability(calibration_key, observability, prediction.delta_xi.shape[0], prediction.delta_xi)
                rotation_losses.append(self._observability_aware_rotation_loss(predicted_phi, target_phi, calibration_observability[..., :3]))
                translation_losses.append(self._observability_aware_translation_loss(predicted_rho, target_rho, calibration_observability[..., 3:6]))
                time_offset_losses.append(self._observability_aware_time_offset_loss(prediction.delta_tau, target_delta_tau, calibration_observability[..., 6:7]))

            change_event_losses.append(self._change_event_loss(prediction.change_event_logit, change_label))
            change_time_losses.append(self._change_time_loss(prediction.change_time, change_time, change_label))

        rotation = torch.stack(rotation_losses).mean()
        translation = torch.stack(translation_losses).mean()
        time_offset = torch.stack(time_offset_losses).mean()
        change_event = torch.stack(change_event_losses).mean()
        change_time = torch.stack(change_time_losses).mean()

        consistency = rotation * 0.0

        components = LossComponents(rotation=rotation, translation=translation, time_offset=time_offset, change_event=change_event, change_time=change_time, consistency=consistency)

        return combine_loss_components(components, self.config.weights)

    @staticmethod
    def _rotation_loss(predicted_phi: torch.Tensor, target_phi: torch.Tensor) -> torch.Tensor:
        """
        Compute mean squared geodesic SO(3) error.

        For every sample

            R_error = R_pred @ R_target^T
            error_phi = Log_SO3(R_error),

        and the scalar loss is the minibatch mean of ||error_phi||^2.
        """

        predicted_rotation = so3_exp(predicted_phi)
        target_rotation = so3_exp(target_phi)
        error_rotation = predicted_rotation @ target_rotation.transpose(-1, -2)
        error_phi = so3_log(error_rotation)

        return torch.sum(error_phi * error_phi, dim=-1).mean()

    @staticmethod
    def _translation_loss(predicted_rho: torch.Tensor, target_rho: torch.Tensor) -> torch.Tensor:
        """Compute the minibatch mean squared norm of translational tangent error."""

        error_rho = predicted_rho - target_rho

        return torch.sum(error_rho * error_rho, dim=-1).mean()

    @staticmethod
    def _time_offset_loss(predicted_delta_tau: torch.Tensor, target_delta_tau: torch.Tensor) -> torch.Tensor:
        """Compute mean squared temporal-offset correction error."""

        error_delta_tau = predicted_delta_tau - target_delta_tau

        return torch.mean(error_delta_tau * error_delta_tau)

    def _require_calibration_observability(self, calibration_key: str, observability: ObservabilityResult, batch_size: int, reference_tensor: torch.Tensor) -> torch.Tensor:
        """Return one validated [phi, rho, tau] observability block for a calibration head."""

        if observability.calibration_features is None:
            raise ValueError("observability.calibration_features must be populated.")

        if calibration_key not in observability.calibration_features:
            raise KeyError(f"Observability features are missing calibration key {calibration_key!r}.")

        calibration_observability = observability.calibration_features[calibration_key]

        if calibration_observability.shape != (batch_size, 7):
            raise ValueError(f"Observability features for {calibration_key!r} must have shape [B, 7].")

        if not torch.isfinite(calibration_observability).all():
            raise ValueError(f"Observability features for {calibration_key!r} must be finite.")

        if torch.any(calibration_observability < 0.0) or torch.any(calibration_observability > 1.0):
            raise ValueError(f"Observability features for {calibration_key!r} must lie in [0, 1].")

        return calibration_observability.detach().to(device=reference_tensor.device, dtype=reference_tensor.dtype)

    def _observability_aware_rotation_loss(self, predicted_phi: torch.Tensor, target_phi: torch.Tensor, observability_phi: torch.Tensor) -> torch.Tensor:
        """Blend target-correction and hold-state rotation losses coordinate-wise."""

        predicted_rotation = so3_exp(predicted_phi)
        target_rotation = so3_exp(target_phi)
        target_error_phi = so3_log(predicted_rotation @ target_rotation.transpose(-1, -2))
        hold_error_phi = so3_log(predicted_rotation)
        target_loss = observability_phi * target_error_phi.square()
        hold_loss = (1.0 - observability_phi) * hold_error_phi.square()

        return torch.sum(target_loss + self.config.observability_hold_weight * hold_loss, dim=-1).mean()

    def _observability_aware_translation_loss(self, predicted_rho: torch.Tensor, target_rho: torch.Tensor, observability_rho: torch.Tensor) -> torch.Tensor:
        """Blend target-correction and hold-state translation losses coordinate-wise."""

        target_error_rho = predicted_rho - target_rho
        target_loss = observability_rho * target_error_rho.square()
        hold_loss = (1.0 - observability_rho) * predicted_rho.square()

        return torch.sum(target_loss + self.config.observability_hold_weight * hold_loss, dim=-1).mean()

    def _observability_aware_time_offset_loss(self, predicted_delta_tau: torch.Tensor, target_delta_tau: torch.Tensor, observability_tau: torch.Tensor) -> torch.Tensor:
        """Blend target-correction and hold-state temporal losses coordinate-wise."""

        target_error_tau = predicted_delta_tau - target_delta_tau
        target_loss = observability_tau * target_error_tau.square()
        hold_loss = (1.0 - observability_tau) * predicted_delta_tau.square()

        return torch.mean(target_loss + self.config.observability_hold_weight * hold_loss)

    def _change_event_loss(self, change_event_logit: torch.Tensor, change_label: torch.Tensor) -> torch.Tensor:
        """Compute binary change-event classification loss directly from logits."""

        target = change_label.to(dtype=change_event_logit.dtype)
        positive_weight = change_event_logit.new_tensor(self.config.change_positive_weight)

        return F.binary_cross_entropy_with_logits(change_event_logit, target, pos_weight=positive_weight)

    def _change_time_loss(self, predicted_change_time: torch.Tensor, target_change_time: torch.Tensor, change_label: torch.Tensor) -> torch.Tensor:
        """
        Compute masked Smooth L1 loss for change time.

        Times are measured in seconds relative to the beginning of the current
        window. No-change samples contribute exactly zero to this loss.
        """

        per_sample_loss = F.smooth_l1_loss(predicted_change_time, target_change_time.to(dtype=predicted_change_time.dtype), beta=self.config.change_time_beta_s, reduction="none")

        change_mask = (change_label > 0.5).to(dtype=per_sample_loss.dtype)
        num_change_samples = change_mask.sum()

        return (per_sample_loss * change_mask).sum() / num_change_samples.clamp_min(1.0)

    @staticmethod
    def _validate_keys(predictions: Mapping[str, CalibrationPrediction], current_calibration: Mapping[str, CalibrationState], targets: Mapping[str, CalibrationTargetBatch]) -> tuple[str, ...]:
        """Require predictions, current states, and targets for the same calibration keys."""

        prediction_keys = tuple(predictions.keys())

        if not prediction_keys:
            raise ValueError("At least one calibration prediction is required.")

        if set(predictions.keys()) != set(current_calibration.keys()):
            raise ValueError("predictions and current_calibration must contain the same calibration keys.")

        if set(predictions.keys()) != set(targets.keys()):
            raise ValueError("predictions and targets must contain the same calibration keys.")

        return prediction_keys

    @staticmethod
    def _require_target_fields(calibration_key: str, target: CalibrationTargetBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the supervised target fields required by CalibrationLoss."""

        if target.next_transform is None:
            raise ValueError(f"Target {calibration_key!r} is missing next_transform.")

        if target.next_time_offset is None:
            raise ValueError(f"Target {calibration_key!r} is missing next_time_offset.")

        if target.change_label is None:
            raise ValueError(f"Target {calibration_key!r} is missing change_label.")

        if target.change_time is None:
            raise ValueError(f"Target {calibration_key!r} is missing change_time.")

        return target.next_transform, target.next_time_offset, target.change_label, target.change_time

    @staticmethod
    def _validate_prediction_and_target_shapes(calibration_key: str, prediction: CalibrationPrediction, current_state: CalibrationState, next_transform: torch.Tensor, next_time_offset: torch.Tensor, change_label: torch.Tensor, change_time: torch.Tensor) -> None:
        """Validate tensor shapes required by the supervised loss."""

        batch_size = current_state.transform.shape[0]

        if prediction.delta_xi.shape != (batch_size, 6):
            raise ValueError(f"Prediction {calibration_key!r} delta_xi must have shape [B, 6].")

        if prediction.delta_tau.shape != (batch_size, 1):
            raise ValueError(f"Prediction {calibration_key!r} delta_tau must have shape [B, 1].")

        if prediction.change_event_logit.shape != (batch_size, 1):
            raise ValueError(f"Prediction {calibration_key!r} change_event_logit must have shape [B, 1].")

        if prediction.change_time.shape != (batch_size, 1):
            raise ValueError(f"Prediction {calibration_key!r} change_time must have shape [B, 1].")

        if next_transform.shape != (batch_size, 4, 4):
            raise ValueError(f"Target {calibration_key!r} next_transform must have shape [B, 4, 4].")

        if next_time_offset.shape != (batch_size, 1):
            raise ValueError(f"Target {calibration_key!r} next_time_offset must have shape [B, 1].")

        if change_label.shape != (batch_size, 1):
            raise ValueError(f"Target {calibration_key!r} change_label must have shape [B, 1].")

        if change_time.shape != (batch_size, 1):
            raise ValueError(f"Target {calibration_key!r} change_time must have shape [B, 1].")