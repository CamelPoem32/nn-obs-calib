"""Sensor-conditioned 6D rotation Flow Matching using the existing pooled backbone.

This is a separate model from the original 7D Lie-group Flow Matching model.
The flow state/velocity is Euclidean [rotation columns(6), delta_translation(3),
delta_tau(1)]. Its inference output is compatible with apply_calibration_6d_update,
NOT with the SE(3) CalibrationUpdater that expects a delta_xi tangent vector.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch

from obscalib.config import PooledCalibrationModelConfig
from obscalib.data.structures import TokenBatch
from obscalib.flow_matching.rotation_6d_integrator import Rotation6DFlowResult, integrate_rotation_6d_flow
from obscalib.flow_matching.rotation_6d_path import Rotation6DFlowBatch, make_rotation_6d_flow_batch, stack_rotation_6d_flow_target
from obscalib.models.calibration_6d_head import Calibration6DPrediction
from obscalib.models.pooled_calibration_6d_model import PooledCalibration6DOutput
from obscalib.models.pooled_calibration_flow_6d_head import PooledCalibrationFlow6DHead
from obscalib.models.pooled_calibration_model import PooledCalibrationModel


@dataclass(frozen=True)
class PooledCalibrationFlow6DTrainingOutput:
    """Analytic correction-to-go query; velocity tensors shaped [B,C,10]."""

    predicted_velocity: torch.Tensor
    flow_batch: Rotation6DFlowBatch
    shared_features: torch.Tensor
    calibration_keys: tuple[str, ...]
    stacked_context: torch.Tensor | None = None

    def by_key(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return {key: (self.predicted_velocity[:, i], self.flow_batch.target_velocity[:, i]) for i, key in enumerate(self.calibration_keys)}


class PooledCalibrationFlow6DModel(PooledCalibrationModel):
    """Reuse Transformer/pooler/MLPBody; replace only the output head and ODE space."""

    def __init__(
        self,
        config: PooledCalibrationModelConfig,
        *,
        integration_steps: int = 16,
        integration_method: Literal['euler', 'midpoint'] = 'midpoint',
        translation_state_scale_m: float = 0.1,
        time_offset_state_scale_s: float = 0.25,
    ) -> None:
        super().__init__(config)
        del self.head
        self.head_6d_flow = PooledCalibrationFlow6DHead(config.head, shared_feature_dim=config.mlp.output_dim, translation_state_scale_m=translation_state_scale_m, time_offset_state_scale_s=time_offset_state_scale_s)
        self.integration_steps = integration_steps
        self.integration_method = integration_method
        self.feature_extractor_frozen = False
        self._validate_integration(integration_steps, integration_method)

    @staticmethod
    def _validate_integration(steps: int, method: str) -> None:
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1 or method not in ('euler', 'midpoint'):
            raise ValueError('Invalid integration steps or method')

    def freeze_feature_extractor(self, freeze: bool = True) -> None:
        """Freeze/unfreeze Transformer, pooler and shared MLPBody together."""
        self.feature_extractor_frozen = bool(freeze)
        for component in (self.transformer_encoder, self.pooler, self.mlp_body):
            component.requires_grad_(not freeze)
            component.eval() if freeze else component.train(self.training)

    def train(self, mode: bool = True) -> 'PooledCalibrationFlow6DModel':
        super().train(mode)
        if self.feature_extractor_frozen:
            for component in (self.transformer_encoder, self.pooler, self.mlp_body):
                component.eval()
        return self

    def initialize_backbone_from_regression(self, source: PooledCalibrationModel, *, freeze: bool = False) -> None:
        """Copy ONLY the shared feature extractor from a pretrained 6D or 3D model."""
        if not isinstance(source, PooledCalibrationModel):
            raise TypeError('source must be a PooledCalibrationModel')
        for name in ('transformer_encoder', 'pooler', 'mlp_body'):
            getattr(self, name).load_state_dict(getattr(source, name).state_dict(), strict=True)
        self.freeze_feature_extractor(freeze)

    def velocity_from_features(self, shared_features: torch.Tensor, state: torch.Tensor, time: torch.Tensor, stacked_context: torch.Tensor | None = None) -> torch.Tensor:
        """Compute [B,C,10] velocity while reusing one Transformer encoding."""
        if shared_features.ndim != 3 or shared_features.shape[-1] != self.config.mlp.output_dim:
            raise ValueError(f'shared_features must be [B,C,{self.config.mlp.output_dim}]')
        b, c, hidden = shared_features.shape
        if state.shape != (b, c, 10) or time.shape != (b, c, 1):
            raise ValueError('state/time must be [B,C,10] / [B,C,1]')
        context = self.flatten_head_context(shared_features, stacked_context)
        velocity = self.head_6d_flow(shared_features.reshape(-1, hidden), state.reshape(-1, 10), time.reshape(-1, 1), context)
        return velocity.reshape(b, c, 10)

    def forward_flow_matching(
        self,
        tokens: TokenBatch,
        target_rotation_6d: Mapping[str, torch.Tensor],
        target_delta_translation: Mapping[str, torch.Tensor],
        target_delta_tau: Mapping[str, torch.Tensor],
        calibration_context: Mapping[str, torch.Tensor] | None = None,
        *,
        time: torch.Tensor | None = None,
        mode: str = 'mixed',
        off_path_probability: float = 0.4,
        identity_probability: float = 0.15,
        rotation_noise_std: float = 0.10,
        translation_noise_std_m: float = 0.02,
        tau_noise_std_s: float = 0.02,
        max_time: float = 0.95,
        generator: torch.Generator | None = None,
        shared_across_calibrations: bool = False,
    ) -> PooledCalibrationFlow6DTrainingOutput:
        """Produce analytic 10D FM training velocities; no ODE integration."""
        if self.feature_extractor_frozen:
            with torch.no_grad():
                shared, context = self.encode_calibration_features(tokens, calibration_context)
        else:
            shared, context = self.encode_calibration_features(tokens, calibration_context)
        b, c, _ = shared.shape
        keys = tuple(tokens.calibration_keys)
        if len(keys) != c:
            raise ValueError('Number of calibration keys and pooled features differ')

        def stack(mapping: Mapping[str, torch.Tensor], width: int, label: str) -> torch.Tensor:
            result = []
            for key in keys:
                if key not in mapping:
                    raise KeyError(f'Missing {label} for calibration {key!r}')
                value = mapping[key]
                if value.shape != (b, width) or value.dtype != shared.dtype or value.device != shared.device:
                    raise ValueError(f'{label}[{key!r}] must match [B,{width}] feature dtype/device')
                result.append(value)
            return torch.stack(result, dim=1)

        target = stack_rotation_6d_flow_target(stack(target_rotation_6d, 6, 'target_rotation_6d'), stack(target_delta_translation, 3, 'target_delta_translation'), stack(target_delta_tau, 1, 'target_delta_tau'))
        flow_batch = make_rotation_6d_flow_batch(target, time=time, mode=mode, off_path_probability=off_path_probability, identity_probability=identity_probability, rotation_noise_std=rotation_noise_std, translation_noise_std_m=translation_noise_std_m, tau_noise_std_s=tau_noise_std_s, max_time=max_time, generator=generator, shared_across_calibrations=shared_across_calibrations)
        predicted = self.velocity_from_features(shared, flow_batch.state, flow_batch.time, context)
        return PooledCalibrationFlow6DTrainingOutput(predicted, flow_batch, shared, keys, context)

    def integrate_from_features(self, shared_features: torch.Tensor, stacked_context: torch.Tensor | None = None, *, num_steps: int | None = None, method: Literal['euler', 'midpoint'] | None = None, record_trajectory: bool = False) -> Rotation6DFlowResult:
        """Integrate differentiably; can be used for training endpoint loss."""
        steps = self.integration_steps if num_steps is None else num_steps
        solver = self.integration_method if method is None else method
        self._validate_integration(steps, solver)
        return integrate_rotation_6d_flow(lambda state, t: self.velocity_from_features(shared_features, state, t, stacked_context), shared_features, num_steps=steps, method=solver, record_trajectory=record_trajectory)

    @staticmethod
    def predictions_from_result(result: Rotation6DFlowResult, calibration_keys: tuple[str, ...]) -> dict[str, Calibration6DPrediction]:
        if result.state.ndim != 3 or result.state.shape[1] != len(calibration_keys):
            raise ValueError('result.state must have shape [B,C,10] matching calibration_keys')
        return {key: Calibration6DPrediction(rotation_6d=result.state[:, i, :6], delta_translation=result.state[:, i, 6:9], delta_tau=result.state[:, i, 9:10]) for i, key in enumerate(calibration_keys)}

    def forward(self, tokens: TokenBatch, calibration_context: Mapping[str, torch.Tensor] | None = None, *, num_steps: int | None = None, method: Literal['euler', 'midpoint'] | None = None) -> PooledCalibration6DOutput:
        """Return the standard 6D regression prediction interface for update/loss.

        Existing 7D FM CalibrationUpdater / WindowStep expects delta_xi and is NOT
        compatible. Use apply_calibration_6d_update instead.
        """
        shared, context = self.encode_calibration_features(tokens, calibration_context)
        result = self.integrate_from_features(shared, context, num_steps=num_steps, method=method)
        return PooledCalibration6DOutput(self.predictions_from_result(result, tuple(tokens.calibration_keys)), shared)


__all__ = ['PooledCalibrationFlow6DTrainingOutput', 'PooledCalibrationFlow6DModel']
