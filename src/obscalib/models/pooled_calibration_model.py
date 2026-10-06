"""Calibration prediction through global attention and shared calibration-wise pooling."""

import torch
from torch import nn

from obscalib.config import PooledCalibrationModelConfig
from obscalib.data.structures import TokenBatch
from obscalib.models.calibration_pooler import CalibrationPooler
from obscalib.models.mlp_body import MLPBody
from obscalib.models.pooled_calibration_head import PooledCalibrationHead
from obscalib.models.structures import ModelOutput, CalibrationPrediction
from obscalib.models.transformer_encoder import SummaryTokenTransformerEncoder


class PooledCalibrationModel(nn.Module):
    """Predict arbitrary numbers of calibration states with shared weights."""

    uses_calibration_context = False
    def __init__(self, config: PooledCalibrationModelConfig) -> None:
        super().__init__()

        self.config = config
        self.transformer_encoder = SummaryTokenTransformerEncoder(config.transformer)
        self.pooler = CalibrationPooler(config.transformer.d_model, config.pooler.n_heads, config.pooler.dropout)
        self.mlp_body = MLPBody(config.mlp, input_dim=config.transformer.d_model)
        self.head = PooledCalibrationHead(config.head, input_dim=config.mlp.output_dim)

    def forward(self, tokens: TokenBatch) -> ModelOutput:
        tokens.validate()

        if tokens.calibration_ids is None or not tokens.calibration_keys:
            raise ValueError("PooledCalibrationModel requires calibration routing metadata.")

        transformer_output = self.transformer_encoder(tokens)

        # Measurement tokens occupy the first N positions in the current encoder.
        encoded_measurements = transformer_output.sequence[:, :tokens.x.shape[1]]

        pooled = self.pooler(
            encoded_measurements,
            tokens.token_mask,
            tokens.calibration_ids,
            len(tokens.calibration_keys),
        )

        batch_size, num_calibrations, pooled_dim = pooled.shape
        flat_pooled = pooled.reshape(batch_size * num_calibrations, pooled_dim)
        flat_shared = self.mlp_body(flat_pooled)
        flat_prediction = self.head(flat_shared)

        shared = flat_shared.reshape(batch_size, num_calibrations, -1)

        predictions = {}

        for calibration_id, calibration_key in enumerate(tokens.calibration_keys):
            predictions[calibration_key] = CalibrationPrediction(
                change_event_logit=flat_prediction.change_event_logit.reshape(batch_size, num_calibrations, 1)[:, calibration_id],
                change_time=flat_prediction.change_time.reshape(batch_size, num_calibrations, 1)[:, calibration_id],
                delta_xi=flat_prediction.delta_xi.reshape(batch_size, num_calibrations, 6)[:, calibration_id],
                delta_tau=flat_prediction.delta_tau.reshape(batch_size, num_calibrations, 1)[:, calibration_id],
            )

        return ModelOutput(predictions=predictions, shared_features=shared)