"""Pooled sensor-conditioned Flow Matching model for calibration *corrections*.

The transformer/pooler/MLPBody architecture is identical to PooledCalibrationModel
and can be initialized from its pretrained weights. A fresh shared velocity head
predicts on SE(3) x R, starting at identity correction and zero clock offset.

Training is analytic and does NOT integrate an ODE. Inference integrates and
returns the existing ModelOutput / CalibrationPrediction structure so the
CalibrationUpdater can continue applying left-multiplicative corrections.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch

from obscalib.config import PooledCalibrationModelConfig
from obscalib.data.structures import TokenBatch
from obscalib.flow_matching.calibration_path import CalibrationFlowBatch, make_calibration_flow_batch
from obscalib.flow_matching.lie_integrator import CalibrationFlowResult, integrate_calibration_flow
from obscalib.models.pooled_calibration_flow_head import PooledCalibrationFlowHead
from obscalib.models.pooled_calibration_model import PooledCalibrationModel
from obscalib.models.structures import CalibrationPrediction, ModelOutput


@dataclass(frozen=True)
class PooledCalibrationFlowTrainingOutput:
    """One analytic FM training query for every active calibration key.

    predicted_velocity and flow_batch.target_velocity have shape [B, C, 7].
    flow_batch.correction_xi and correction_tau have shape [B, C, 6/1].
    The order of C follows calibration_keys; use by_key() for dictionary losses.
    """

    predicted_velocity: torch.Tensor
    flow_batch: CalibrationFlowBatch
    shared_features: torch.Tensor
    calibration_keys: tuple[str, ...]

    def by_key(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Map keys to (predicted_velocity, target_velocity), each [B, 7]."""

        return {key: (self.predicted_velocity[:, index], self.flow_batch.target_velocity[:, index]) for index, key in enumerate(self.calibration_keys)}


class PooledCalibrationFlowModel(PooledCalibrationModel):
    """Drop-in inference model with an explicit integration-free FM training API.

    Until a dedicated flow config is introduced, reuse PooledCalibrationModelConfig
    for the exact pretrained backbone/head widths and pass integration settings
    explicitly. No existing config module needs modification in this stage.
    """

    def __init__(
        self,
        config: PooledCalibrationModelConfig,
        *,
        integration_steps: int = 8,
        integration_method: Literal["euler", "midpoint"] = "midpoint",
        rotation_state_scale_rad: float = 1.0,
        translation_state_scale_m: float = 0.1,
        time_offset_state_scale_s: float = 0.25,
        initialize_state_time_independent: bool = True,
    ) -> None:
        super().__init__(config)
        self.head = PooledCalibrationFlowHead(
            config.head,
            shared_feature_dim=config.mlp.output_dim,
            rotation_state_scale_rad=rotation_state_scale_rad,
            translation_state_scale_m=translation_state_scale_m,
            time_offset_state_scale_s=time_offset_state_scale_s,
            initialize_state_time_independent=initialize_state_time_independent,
        )
        self.feature_extractor_frozen = False
        self.integration_steps = integration_steps
        self.integration_method = integration_method
        self._validate_integrator_settings(integration_steps, integration_method)

    @staticmethod
    def _validate_integrator_settings(steps: int, method: str) -> None:
        if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
            raise ValueError("integration_steps must be a positive integer.")
        if method not in ("euler", "midpoint"):
            raise ValueError("integration_method must be 'euler' or 'midpoint'.")

    def freeze_feature_extractor(self, freeze: bool = True) -> None:
        """Freeze/unfreeze pretrained transformer, calibration pooler, and MLPBody.

        Frozen modules remain in eval mode during FM-head training, disabling
        dropout without disabling gradients to the new velocity head.
        """

        self.feature_extractor_frozen = freeze
        for component in (self.transformer_encoder, self.pooler, self.mlp_body):
            component.requires_grad_(not freeze)
            component.eval() if freeze else component.train(self.training)

    def train(self, mode: bool = True) -> "PooledCalibrationFlowModel":
        super().train(mode)
        if self.feature_extractor_frozen:
            for component in (self.transformer_encoder, self.pooler, self.mlp_body):
                component.eval()
        return self

    def initialize_backbone_from_regression(self, regression_model: PooledCalibrationModel, *, freeze: bool = True) -> None:
        """Copy only pretrained feature-extractor weights, NEVER the regression head.

        Load the old checkpoint into a PooledCalibrationModel of the same config,
        then call this function. The FM head retains its random feature weights
        and zero-initialized correction-state/time input columns.
        """

        if not isinstance(regression_model, PooledCalibrationModel):
            raise TypeError("regression_model must be a PooledCalibrationModel instance.")
        for name in ("transformer_encoder", "pooler", "mlp_body"):
            getattr(self, name).load_state_dict(getattr(regression_model, name).state_dict(), strict=True)
        self.freeze_feature_extractor(freeze)

    def velocity_from_features(
        self,
        shared_features: torch.Tensor,
        correction_xi: torch.Tensor,
        correction_tau: torch.Tensor,
        time: torch.Tensor,
        stacked_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict [B, C, 7] velocities without rerunning the transformer."""

        if shared_features.ndim != 3 or shared_features.shape[-1] != self.config.mlp.output_dim:
            raise ValueError(f"shared_features must have shape [B, C, {self.config.mlp.output_dim}].")
        batch_size, num_calibrations, shared_dim = shared_features.shape
        for name, tensor, last_dim in (("correction_xi", correction_xi, 6), ("correction_tau", correction_tau, 1), ("time", time, 1)):
            if tensor.shape != (batch_size, num_calibrations, last_dim):
                raise ValueError(f"{name} must have shape {(batch_size, num_calibrations, last_dim)}.")

        flat_context = self.flatten_head_context(shared_features, stacked_context)
        velocities = self.head(
            shared_features.reshape(-1, shared_dim),
            correction_xi.reshape(-1, 6),
            correction_tau.reshape(-1, 1),
            time.reshape(-1, 1),
            flat_context,
        )
        return velocities.reshape(batch_size, num_calibrations, 7)

    def forward_flow_matching(
        self,
        tokens: TokenBatch,
        target_delta_xi: Mapping[str, torch.Tensor],
        target_delta_tau: Mapping[str, torch.Tensor],
        calibration_context: Mapping[str, torch.Tensor] | None = None,
        *,
        time: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        shared_across_calibrations: bool = False,
    ) -> PooledCalibrationFlowTrainingOutput:
        """Make analytic training paths and predict their velocities without ODEs.

        Targets are keyed like TokenBatch.calibration_keys and must contain [B,6]
        or [B,1] physical left-applied updates. Time, if supplied, is [B,C,1].
        To compute FM loss, compare predicted_velocity to
        flow_batch.target_velocity using calibration_velocity_loss.
        """

        shared, stacked_context = self.encode_calibration_features(tokens, calibration_context)
        batch_size, num_calibrations, _ = shared.shape
        keys = tuple(tokens.calibration_keys)
        missing_xi = [key for key in keys if key not in target_delta_xi]
        missing_tau = [key for key in keys if key not in target_delta_tau]
        if missing_xi or missing_tau:
            raise KeyError(f"Missing calibration targets, delta_xi={missing_xi}, delta_tau={missing_tau}.")

        def stack_target(mapping: Mapping[str, torch.Tensor], width: int, name: str) -> torch.Tensor:
            rows = []
            for key in keys:
                value = mapping[key]
                if value.shape != (batch_size, width):
                    raise ValueError(f"{name}[{key!r}] must have shape {(batch_size, width)}.")
                if value.dtype != shared.dtype or value.device != shared.device:
                    raise ValueError(f"{name}[{key!r}] must match shared feature dtype and device.")
                rows.append(value)
            return torch.stack(rows, dim=1)

        delta_xi = stack_target(target_delta_xi, 6, "target_delta_xi")
        delta_tau = stack_target(target_delta_tau, 1, "target_delta_tau")
        flow_batch = make_calibration_flow_batch(delta_xi, delta_tau, time=time, generator=generator, shared_across_calibrations=shared_across_calibrations)
        velocity = self.velocity_from_features(shared, flow_batch.correction_xi, flow_batch.correction_tau, flow_batch.time, stacked_context)
        return PooledCalibrationFlowTrainingOutput(predicted_velocity=velocity, flow_batch=flow_batch, shared_features=shared, calibration_keys=keys)

    def integrate_from_features(
        self,
        shared_features: torch.Tensor,
        stacked_context: torch.Tensor | None = None,
        *,
        num_steps: int | None = None,
        method: Literal["euler", "midpoint"] | None = None,
        record_trajectory: bool = False,
    ) -> CalibrationFlowResult:
        """Integrate from zero correction; reuses features for every field call."""

        steps = self.integration_steps if num_steps is None else num_steps
        solver = self.integration_method if method is None else method
        self._validate_integrator_settings(steps, solver)

        def velocity_field(xi: torch.Tensor, tau: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            return self.velocity_from_features(shared_features, xi, tau, time, stacked_context)

        return integrate_calibration_flow(velocity_field, shared_features, num_steps=steps, method=solver, record_trajectory=record_trajectory)

    def forward(
        self,
        tokens: TokenBatch,
        calibration_context: Mapping[str, torch.Tensor] | None = None,
        *,
        num_steps: int | None = None,
        method: Literal["euler", "midpoint"] | None = None,
    ) -> ModelOutput:
        """Integrate a deterministic correction and return existing ModelOutput.

        Set model.eval() and use torch.no_grad() for validation/inference. The
        default signature remains compatible with WindowStep and the updater.
        """

        shared, stacked_context = self.encode_calibration_features(tokens, calibration_context)
        result = self.integrate_from_features(shared, stacked_context, num_steps=num_steps, method=method)
        predictions = {}
        for index, key in enumerate(tokens.calibration_keys):
            zeros = result.delta_tau[:, index].new_zeros(result.delta_tau.shape[0], 1)
            predictions[key] = CalibrationPrediction(
                change_event_logit=zeros,
                change_time=zeros.clone(),
                delta_xi=result.delta_xi[:, index],
                delta_tau=result.delta_tau[:, index],
            )
        return ModelOutput(predictions=predictions, shared_features=shared)


__all__ = ["PooledCalibrationFlowTrainingOutput", "PooledCalibrationFlowModel"]
