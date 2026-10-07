"""Shared calibration-wise pooling of globally contextualized measurement tokens."""

import torch
from torch import nn


class CalibrationPooler(nn.Module):
    """
    Pool globally contextualized measurement tokens independently for each physical calibration using shared weights.

    calibration_ids are routing metadata only and are never embedded as learned numerical sensor identities. When calibration_context_dim > 0, the numeric calibration state is encoded into a query offset that conditions attention for that calibration. Each query still attends only to tokens routed to its own calibration; those tokens have already passed through the global Transformer and therefore already contain cross-sensor context.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, calibration_context_dim: int = 0) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive.")

        if n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError("n_heads must be positive and divide d_model.")

        if calibration_context_dim < 0:
            raise ValueError("calibration_context_dim must be nonnegative.")

        self.d_model = d_model
        self.calibration_context_dim = calibration_context_dim

        self.query = nn.Parameter(torch.empty(1, 1, d_model))
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        if calibration_context_dim > 0:
            self.context_encoder = nn.Sequential(
                nn.Linear(calibration_context_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

            # Keep the initial conditioning relatively small so the new model starts near the historical shared-query behavior.
            nn.init.normal_(self.context_encoder[-1].weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.context_encoder[-1].bias)
        else:
            self.context_encoder = None

        nn.init.normal_(self.query, mean=0.0, std=0.02)

    def forward(self, encoded_tokens: torch.Tensor, token_mask: torch.Tensor, calibration_ids: torch.Tensor, num_calibrations: int, calibration_context: torch.Tensor | None = None) -> torch.Tensor:
        """Return pooled calibration features with shape [B, K, D]."""

        if encoded_tokens.ndim != 3:
            raise ValueError("encoded_tokens must have shape [B, N, D].")

        batch_size, num_tokens, feature_dim = encoded_tokens.shape

        if feature_dim != self.d_model:
            raise ValueError(f"encoded_tokens must have last dimension {self.d_model}.")

        if token_mask.shape != (batch_size, num_tokens):
            raise ValueError("token_mask must have shape [B, N].")

        if calibration_ids.shape != (batch_size, num_tokens):
            raise ValueError("calibration_ids must have shape [B, N].")

        if num_calibrations <= 0:
            raise ValueError("num_calibrations must be positive.")

        if self.calibration_context_dim > 0:
            if calibration_context is None:
                raise ValueError("calibration_context is required when calibration_context_dim > 0.")

            if calibration_context.shape != (batch_size, num_calibrations, self.calibration_context_dim):
                raise ValueError(f"calibration_context must have shape {(batch_size, num_calibrations, self.calibration_context_dim)}, got {tuple(calibration_context.shape)}.")

            if calibration_context.device != encoded_tokens.device:
                raise ValueError("calibration_context and encoded_tokens must be on the same device.")

            if calibration_context.dtype != encoded_tokens.dtype:
                raise ValueError("calibration_context and encoded_tokens must use the same dtype.")

        elif calibration_context is not None and calibration_context.shape != (batch_size, num_calibrations, 0):
            raise ValueError(f"calibration_context must be None or have shape {(batch_size, num_calibrations, 0)} when calibration_context_dim == 0.")

        pooled_features = []

        for calibration_id in range(num_calibrations):
            calibration_mask = token_mask & (calibration_ids == calibration_id)

            if torch.any(calibration_mask.sum(dim=1) == 0):
                raise ValueError(f"Every batch item must contain at least one token for calibration_id={calibration_id}.")

            query = self.query.expand(batch_size, -1, -1)

            if self.context_encoder is not None:
                query = query + self.context_encoder(calibration_context[:, calibration_id]).unsqueeze(1)

            pooled, _ = self.attention(
                query=query,
                key=encoded_tokens,
                value=encoded_tokens,
                key_padding_mask=~calibration_mask,
                need_weights=False,
            )

            pooled_features.append(pooled[:, 0])

        return torch.stack(pooled_features, dim=1)