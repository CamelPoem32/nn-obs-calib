"""Calibration prediction through global attention and shared calibration-wise pooling."""

from collections.abc import Mapping

import torch
from torch import nn

from obscalib.config import PooledCalibrationModelConfig
from obscalib.data.structures import TokenBatch
from obscalib.models.calibration_pooler import CalibrationPooler
from obscalib.models.mlp_body import MLPBody
from obscalib.models.pooled_calibration_head import PooledCalibrationHead
from obscalib.models.structures import CalibrationPrediction, ModelOutput
from obscalib.models.transformer_encoder import SummaryTokenTransformerEncoder


class PooledCalibrationModel(nn.Module):
    """Predict arbitrary numbers of calibration states with shared weights."""

    def __init__(self, config: PooledCalibrationModelConfig) -> None:
        super().__init__()

        self.config = config
        self.calibration_context_dim = max(config.pooler.calibration_context_dim, config.head.calibration_context_dim)
        self.uses_calibration_context = self.calibration_context_dim > 0

        self.transformer_encoder = SummaryTokenTransformerEncoder(config.transformer)
        self.pooler = CalibrationPooler(config.transformer.d_model, config.pooler.n_heads, config.pooler.dropout, calibration_context_dim=config.pooler.calibration_context_dim)
        self.mlp_body = MLPBody(config.mlp, input_dim=config.transformer.d_model)
        self.head = PooledCalibrationHead(config.head, shared_feature_dim=config.mlp.output_dim)

    def _stack_calibration_context(self, tokens: TokenBatch, calibration_context: Mapping[str, torch.Tensor] | None, batch_size: int) -> torch.Tensor | None:
        if not self.uses_calibration_context:
            return None

        if calibration_context is None:
            raise ValueError("PooledCalibrationModel requires calibration_context when either pooler or head calibration context is enabled.")

        missing_calibration_keys = [calibration_key for calibration_key in tokens.calibration_keys if calibration_key not in calibration_context]

        if missing_calibration_keys:
            raise KeyError(f"calibration_context is missing calibration keys required by pooled predictions: {missing_calibration_keys}.")

        context_rows = []

        for calibration_key in tokens.calibration_keys:
            context = calibration_context[calibration_key]

            if context.ndim != 2 or context.shape != (batch_size, self.calibration_context_dim):
                raise ValueError(f"calibration_context[{calibration_key!r}] must have shape {(batch_size, self.calibration_context_dim)}, got {tuple(context.shape)}.")

            if context.device != tokens.x.device:
                raise ValueError(f"calibration_context[{calibration_key!r}] and tokens must be on the same device.")

            if context.dtype != tokens.x.dtype:
                raise ValueError(f"calibration_context[{calibration_key!r}] and tokens must use the same dtype.")

            context_rows.append(context)

        return torch.stack(context_rows, dim=1)

    def forward(self, tokens: TokenBatch, calibration_context: Mapping[str, torch.Tensor] | None = None) -> ModelOutput:
        tokens.validate()

        if tokens.calibration_ids is None or not tokens.calibration_keys:
            raise ValueError("PooledCalibrationModel requires calibration routing metadata.")

        batch_size = tokens.x.shape[0]
        stacked_context = self._stack_calibration_context(tokens, calibration_context, batch_size)

        transformer_output = self.transformer_encoder(tokens)

        # Measurement tokens occupy the first N positions in the current encoder.
        encoded_measurements = transformer_output.sequence[:, :tokens.x.shape[1]]

        pooled = self.pooler(
            encoded_measurements,
            tokens.token_mask,
            tokens.calibration_ids,
            len(tokens.calibration_keys),
            calibration_context=stacked_context if self.config.pooler.calibration_context_dim > 0 else None,
        )

        batch_size, num_calibrations, pooled_dim = pooled.shape
        flat_pooled = pooled.reshape(batch_size * num_calibrations, pooled_dim)
        flat_shared = self.mlp_body(flat_pooled)
        shared = flat_shared.reshape(batch_size, num_calibrations, -1)

        if self.config.head.calibration_context_dim > 0:
            flat_context = stacked_context.reshape(batch_size * num_calibrations, self.config.head.calibration_context_dim)
        else:
            flat_context = flat_shared.new_empty((batch_size * num_calibrations, 0))

        flat_prediction = self.head(flat_shared, flat_context)

        predictions = {}

        for calibration_id, calibration_key in enumerate(tokens.calibration_keys):
            predictions[calibration_key] = CalibrationPrediction(
                change_event_logit=flat_prediction.change_event_logit.reshape(batch_size, num_calibrations, 1)[:, calibration_id],
                change_time=flat_prediction.change_time.reshape(batch_size, num_calibrations, 1)[:, calibration_id],
                delta_xi=flat_prediction.delta_xi.reshape(batch_size, num_calibrations, 6)[:, calibration_id],
                delta_tau=flat_prediction.delta_tau.reshape(batch_size, num_calibrations, 1)[:, calibration_id],
            )

        return ModelOutput(predictions=predictions, shared_features=shared)