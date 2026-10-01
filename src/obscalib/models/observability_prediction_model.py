"""Calibration-wise observability prediction from encoded multi-sensor measurement tokens."""

from __future__ import annotations

import torch
from torch import nn

from obscalib.config import ObservabilityPredictionHeadConfig, ObservabilityPredictionModelConfig
from obscalib.data.structures import TokenBatch
from obscalib.models.activations import make_activation
from obscalib.models.structures import ObservabilityPredictionOutput
from obscalib.models.transformer_encoder import SummaryTokenTransformerEncoder


class ObservabilityPredictionHead(nn.Module):
    """Predict seven bounded observability coordinates from one calibration-wise feature vector."""

    def __init__(self, config: ObservabilityPredictionHeadConfig, input_dim: int) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        layers: list[nn.Module] = []
        previous_dim = input_dim

        for hidden_dim in config.hidden_dims:
            layers.extend((nn.Linear(previous_dim, hidden_dim), make_activation(config.activation), nn.Dropout(config.dropout)))
            previous_dim = hidden_dim

        self.body = nn.Sequential(*layers) if layers else nn.Identity()
        self.output = nn.Linear(previous_dim, 7)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("features must have shape [B, D].")

        return torch.sigmoid(self.output(self.body(features)))


class ObservabilityPredictionModel(nn.Module):
    """
    Predict per-calibration coordinate observability.

    The Transformer remains globally multi-sensor: every measurement token may
    attend to every other measurement token. After encoding, measurement tokens
    are grouped by calibration ID and mean-pooled. The resulting calibration-wise
    feature is passed either through one independent head per calibration or one
    shared head reused for every calibration.
    """

    def __init__(self, config: ObservabilityPredictionModelConfig) -> None:
        super().__init__()

        self.config = config
        self.transformer_encoder = SummaryTokenTransformerEncoder(config.transformer)

        if config.head_mode == "individual":
            self.individual_heads = nn.ModuleDict({calibration_key: ObservabilityPredictionHead(config.head, config.transformer.d_model) for calibration_key in config.calibration_keys})
            self.shared_head = None
        else:
            self.individual_heads = nn.ModuleDict()
            self.shared_head = ObservabilityPredictionHead(config.head, config.transformer.d_model)

    def _pool_calibration_features(self, encoded_measurements: torch.Tensor, tokens: TokenBatch) -> dict[str, torch.Tensor]:
        pooled_features: dict[str, torch.Tensor] = {}

        for calibration_id, calibration_key in enumerate(self.config.calibration_keys):
            calibration_mask = tokens.token_mask & (tokens.calibration_ids == calibration_id)
            token_counts = calibration_mask.sum(dim=1)

            if torch.any(token_counts == 0):
                missing_batch_indices = torch.nonzero(token_counts == 0, as_tuple=False).flatten().tolist()
                raise ValueError(f"Calibration {calibration_key!r} has no valid measurement tokens for batch items {missing_batch_indices}.")

            mask = calibration_mask.unsqueeze(-1).to(dtype=encoded_measurements.dtype)
            pooled_features[calibration_key] = (encoded_measurements * mask).sum(dim=1) / token_counts.unsqueeze(-1).to(dtype=encoded_measurements.dtype)

        return pooled_features

    def forward(self, tokens: TokenBatch) -> ObservabilityPredictionOutput:
        tokens.validate()

        if tokens.calibration_ids is None:
            raise ValueError("ObservabilityPredictionModel requires TokenBatch.calibration_ids.")

        transformer_output = self.transformer_encoder(tokens)
        num_measurement_tokens = tokens.x.shape[1]
        encoded_measurements = transformer_output.sequence[:, :num_measurement_tokens]
        pooled_features = self._pool_calibration_features(encoded_measurements, tokens)

        if self.config.head_mode == "individual":
            predictions = {calibration_key: self.individual_heads[calibration_key](pooled_features[calibration_key]) for calibration_key in self.config.calibration_keys}
        else:
            predictions = {calibration_key: self.shared_head(pooled_features[calibration_key]) for calibration_key in self.config.calibration_keys}

        return ObservabilityPredictionOutput(predictions=predictions, pooled_features=pooled_features)