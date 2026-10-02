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


class SharedCrossAttentionPooler(nn.Module):
    """Pool one calibration token group with one shared learned query."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()

        self.query = nn.Parameter(
            torch.empty(1, 1, d_model)
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(
            d_model
        )

        nn.init.normal_(
            self.query,
            mean=0.0,
            std=0.02,
        )

    def forward(
        self,
        encoded_measurements: torch.Tensor,
        calibration_mask: torch.Tensor,
    ) -> torch.Tensor:
        if encoded_measurements.ndim != 3:
            raise ValueError(
                "encoded_measurements must have shape [B, N, D]."
            )

        if calibration_mask.ndim != 2:
            raise ValueError(
                "calibration_mask must have shape [B, N]."
            )

        if encoded_measurements.shape[:2] != calibration_mask.shape:
            raise ValueError(
                "encoded_measurements and calibration_mask must agree in batch and token dimensions."
            )

        token_counts = calibration_mask.sum(
            dim=1
        )

        if torch.any(token_counts == 0):
            missing_batch_indices = torch.nonzero(
                token_counts == 0,
                as_tuple=False,
            ).flatten().tolist()

            raise ValueError(
                f"Cross-attention pooling received no valid tokens for batch items {missing_batch_indices}."
            )

        query = self.query.expand(
            encoded_measurements.shape[0],
            -1,
            -1,
        )

        attention_output, _ = self.cross_attention(
            query=query,
            key=encoded_measurements,
            value=encoded_measurements,
            key_padding_mask=~calibration_mask,
            need_weights=False,
        )

        pooled = self.norm(
            query
            + attention_output
        )

        return pooled[:, 0]


class ObservabilityPredictionModel(nn.Module):
    """
    Predict per-calibration coordinate observability.

    The first Transformer remains globally multi-sensor: every measurement token
    may attend to every other measurement token. After global encoding, the same
    shared cross-attention pooler is applied independently to each calibration
    group. The pooler uses one learned query to extract a fixed-dimensional
    calibration representation from a variable number of encoded measurement
    tokens.

    All calibration-wise pooled features are stacked and normalized jointly by one
    shared BatchNorm. The normalized features are then passed either through one
    independent head per calibration or one shared head reused for every
    calibration. The shared mode therefore has no calibration-specific learned
    prediction parameters.
    """

    def __init__(self, config: ObservabilityPredictionModelConfig) -> None:
        super().__init__()

        self.config = config

        ##################################################
        # Global multi-sensor Transformer
        ##################################################

        self.transformer_encoder = SummaryTokenTransformerEncoder(config.transformer)

        ##################################################
        # Shared calibration-wise learned pooling
        ##################################################

        self.calibration_pooler = SharedCrossAttentionPooler(
            d_model=config.transformer.d_model,
            n_heads=config.transformer.n_heads,
            dropout=config.transformer.dropout,
        )

        ##################################################
        # Shared pooled-feature normalization
        ##################################################

        self.pooled_feature_norm = nn.BatchNorm1d(config.transformer.d_model)

        ##################################################
        # Prediction heads
        ##################################################

        if config.head_mode == "individual":
            self.individual_heads = nn.ModuleDict({
                calibration_key: ObservabilityPredictionHead(config.head, config.transformer.d_model)
                for calibration_key in config.calibration_keys
            })

            self.shared_head = None

        else:
            self.individual_heads = nn.ModuleDict()
            self.shared_head = ObservabilityPredictionHead(config.head, config.transformer.d_model)

    def _pool_calibration_features(self, encoded_measurements: torch.Tensor, tokens: TokenBatch) -> dict[str, torch.Tensor]:
        """
        Cross-attention-pool globally encoded measurement tokens per calibration.

        Every calibration uses exactly the same SharedCrossAttentionPooler
        parameters. Only the runtime calibration mask changes.
        """

        pooled_features: dict[str, torch.Tensor] = {}

        for calibration_id, calibration_key in enumerate(self.config.calibration_keys):
            calibration_mask = tokens.token_mask & (tokens.calibration_ids == calibration_id)
            token_counts = calibration_mask.sum(dim=1)

            if torch.any(token_counts == 0):
                missing_batch_indices = torch.nonzero(token_counts == 0, as_tuple=False).flatten().tolist()
                raise ValueError(f"Calibration {calibration_key!r} has no valid measurement tokens for batch items {missing_batch_indices}.")

            pooled_features[calibration_key] = self.calibration_pooler(
                encoded_measurements,
                calibration_mask,
            )

        return pooled_features

    def _normalize_pooled_features(self, pooled_features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Normalize calibration-wise features jointly with one shared BatchNorm.

        The dictionary of [B, D] features is converted to [B, K, D], then to
        [B * K, D]. BatchNorm therefore sees all batch items and all calibration
        groups together and maintains one calibration-independent set of running
        statistics.
        """

        if not self.config.calibration_keys:
            raise ValueError("At least one calibration key is required.")

        stacked_features = torch.stack([pooled_features[calibration_key] for calibration_key in self.config.calibration_keys], dim=1)

        batch_size, num_calibrations, feature_dim = stacked_features.shape

        if feature_dim != self.config.transformer.d_model:
            raise ValueError(f"Pooled feature dimension must equal transformer d_model={self.config.transformer.d_model}, got {feature_dim}.")

        normalized_flattened_features = self.pooled_feature_norm(stacked_features.reshape(batch_size * num_calibrations, feature_dim))
        normalized_stacked_features = normalized_flattened_features.reshape(batch_size, num_calibrations, feature_dim)

        return {
            calibration_key: normalized_stacked_features[:, calibration_id]
            for calibration_id, calibration_key in enumerate(self.config.calibration_keys)
        }

    def forward(self, tokens: TokenBatch) -> ObservabilityPredictionOutput:
        tokens.validate()

        if tokens.calibration_ids is None:
            raise ValueError("ObservabilityPredictionModel requires TokenBatch.calibration_ids.")

        ##################################################
        # Global multi-sensor encoding
        ##################################################

        transformer_output = self.transformer_encoder(tokens)

        # SummaryTokenTransformerEncoder appends its learned global summary tokens
        # after all measurement tokens. Calibration pooling uses only the encoded
        # measurement-token portion.
        num_measurement_tokens = tokens.x.shape[1]
        encoded_measurements = transformer_output.sequence[:, :num_measurement_tokens]

        ##################################################
        # Shared calibration-wise cross-attention pooling
        ##################################################

        pooled_features = self._pool_calibration_features(encoded_measurements, tokens)

        ##################################################
        # Shared feature normalization
        ##################################################

        normalized_pooled_features = self._normalize_pooled_features(pooled_features)

        ##################################################
        # Prediction
        ##################################################

        if self.config.head_mode == "individual":
            predictions = {
                calibration_key: self.individual_heads[calibration_key](normalized_pooled_features[calibration_key])
                for calibration_key in self.config.calibration_keys
            }

        else:
            if self.shared_head is None:
                raise RuntimeError("shared_head must be initialized when head_mode is not 'individual'.")

            predictions = {
                calibration_key: self.shared_head(normalized_pooled_features[calibration_key])
                for calibration_key in self.config.calibration_keys
            }

        # Return the pre-BatchNorm pooled representation. This keeps the diagnostic
        # output meaningful: it reflects what the global Transformer and learned
        # calibration pooler produced before prediction-stage normalization.
        return ObservabilityPredictionOutput(
            predictions=predictions,
            pooled_features=pooled_features,
        )