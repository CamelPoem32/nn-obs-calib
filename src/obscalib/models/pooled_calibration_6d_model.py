"""Pooled Transformer regression of full-range calibration rotations in 6D."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import pickle

import torch

from obscalib.config import PooledCalibrationModelConfig
from obscalib.data.structures import TokenBatch
from obscalib.models.calibration_6d_head import Calibration6DHead, Calibration6DPrediction
from obscalib.models.pooled_calibration_model import PooledCalibrationModel


@dataclass
class PooledCalibration6DOutput:
    """Predictions share weights across calibration keys and follow token key order."""

    predictions: dict[str, Calibration6DPrediction]
    shared_features: torch.Tensor


class PooledCalibration6DModel(PooledCalibrationModel):
    """Reuse the original Transformer/pooler/MLP, with a new 10D regression head.

    The legacy head is removed, not altered. This deliberately does not implement
    ``ModelOutput`` or ``delta_xi``: callers must apply predictions via the new
    rotation-matrix update, not an SE(3) exponential or FM integrator.
    """

    def __init__(self, config: PooledCalibrationModelConfig) -> None:
        super().__init__(config)
        del self.head
        self.head_6d = Calibration6DHead(config.head, shared_feature_dim=config.mlp.output_dim)

    def freeze_feature_extractor(self, freeze: bool = True) -> None:
        """Freeze/unfreeze pretrained Transformer, pooler and shared MLP together."""
        for module in (self.transformer_encoder, self.pooler, self.mlp_body):
            module.requires_grad_(not freeze)

    def forward(self, tokens: TokenBatch, calibration_context: Mapping[str, torch.Tensor] | None = None) -> PooledCalibration6DOutput:
        shared, stacked_context = self.encode_calibration_features(tokens, calibration_context)
        batch_size, num_calibrations, shared_dim = shared.shape
        flat_shared = shared.reshape(batch_size * num_calibrations, shared_dim)
        flat_context = self.flatten_head_context(shared, stacked_context)
        flat_prediction = self.head_6d(flat_shared, flat_context)
        predictions = {}
        for calibration_id, calibration_key in enumerate(tokens.calibration_keys):
            predictions[calibration_key] = Calibration6DPrediction(
                rotation_6d=flat_prediction.rotation_6d.reshape(batch_size, num_calibrations, 6)[:, calibration_id],
                delta_translation=flat_prediction.delta_translation.reshape(batch_size, num_calibrations, 3)[:, calibration_id],
                delta_tau=flat_prediction.delta_tau.reshape(batch_size, num_calibrations, 1)[:, calibration_id],
            )
        return PooledCalibration6DOutput(predictions=predictions, shared_features=shared)


def load_6d_pretrained_features(model: PooledCalibration6DModel, checkpoint: str | Path | Mapping, *, strict: bool = True, trusted_checkpoint: bool = False) -> tuple[str, ...]:
    """Transfer only Transformer, pooler and shared MLP weights from FM/regression.

    Accepts a state_dict or a training-checkpoint dictionary saved with keys
    ``model_state_dict``, ``state_dict``, or ``model``. The 6D head is always kept
    freshly initialized. ``strict=True`` refuses missing or mismatched backbone
    tensors; no silent partial pretraining is allowed. If a local training
    checkpoint contains Python objects that PyTorch cannot load in safe mode,
    ``trusted_checkpoint=True`` explicitly permits full deserialization.
    """
    if isinstance(checkpoint, (str, Path)):
        try:
            checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError:
            if not trusted_checkpoint:
                raise RuntimeError("Checkpoint requires Python-object deserialization. Only for a trusted local file, retry with trusted_checkpoint=True.") from None
            checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a path or mapping")

    source = checkpoint
    for key in ("model_state_dict", "state_dict", "model"):
        if key in source and isinstance(source[key], Mapping):
            source = source[key]
            break

    own = model.state_dict()
    prefixes = ("transformer_encoder.", "pooler.", "mlp_body.")
    required = {key for key in own if key.startswith(prefixes)}
    candidates = {}
    for key, value in source.items():
        normalized = key
        for prefix in ("module.", "backbone."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        if normalized in required:
            candidates[normalized] = value
    missing = required - candidates.keys()
    wrong_shape = [key for key in candidates if candidates[key].shape != own[key].shape]
    if wrong_shape:
        raise ValueError(f"Pretrained feature tensor shapes differ for {wrong_shape}")
    if strict and missing:
        raise KeyError(f"Missing {len(missing)} feature tensors in checkpoint: {sorted(missing)[:12]}")
    if not candidates:
        raise ValueError("Checkpoint contains no matching Transformer/pooler/shared-MLP parameters")

    result = model.load_state_dict(candidates, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected keys during feature loading: {result.unexpected_keys}")
    return tuple(sorted(candidates))


__all__ = ["PooledCalibration6DOutput", "PooledCalibration6DModel", "load_6d_pretrained_features"]
