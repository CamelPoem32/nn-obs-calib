"""Composition of the learned portion of the calibration pipeline."""

from collections.abc import Mapping

import torch
from torch import nn

from obscalib.config import ObsCalibModelConfig
from obscalib.data.structures import TokenBatch
from obscalib.models.structures import ModelOutput
from obscalib.models.calibration_head import CalibrationHead
from obscalib.models.mlp_body import MLPBody
from obscalib.models.transformer_encoder import SummaryTokenTransformerEncoder


class ObsCalibModel(nn.Module):
    """Transformer, shared MLP, and reusable sensor-specific heads."""

    def __init__(self, config: ObsCalibModelConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer_encoder = SummaryTokenTransformerEncoder(config.transformer)
        self.mlp_body = MLPBody(
            config.mlp,
            input_dim=config.summary_feature_dim,
        )
        self.heads = nn.ModuleDict(
            {
                head_key: CalibrationHead(
                    config.calibration_head,
                    shared_feature_dim=config.shared_feature_dim,
                )
                for head_key in config.head_keys
            }
        )

    def forward(
        self,
        tokens: TokenBatch,
        calibration_context: Mapping[str, torch.Tensor],
    ) -> ModelOutput:
        """Predict per-head calibration changes from prepared sequence tokens."""

        # Project and encode measurement tokens, including learned summaries.
        transformer_output = self.transformer_encoder(tokens)

        # Flatten all K summaries so their count remains a direct ablation:
        # [B, K, d_model] -> [B, K * d_model].
        flattened_summaries = transformer_output.summary_features.flatten(start_dim=1)

        # Build one shared calibration feature vector.
        # shared_features: [B, d_shared]
        shared_features = self.mlp_body(flattened_summaries)

        # Evaluate each configured sensor-specific calibration head.
        missing = set(self.heads) - set(calibration_context)
        if missing:
            raise KeyError(f"Missing calibration context for heads: {sorted(missing)}")
        predictions = {
            head_key: head(shared_features, calibration_context[head_key])
            for head_key, head in self.heads.items()
        }
        return ModelOutput(predictions=predictions, shared_features=shared_features)
