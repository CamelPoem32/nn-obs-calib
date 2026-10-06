"""Shared calibration-wise pooling of globally contextualized measurement tokens."""

import torch
from torch import nn


class CalibrationPooler(nn.Module):
    """
    Pool globally contextualized measurement tokens independently for each
    physical calibration using one shared learned query.

    calibration_ids are routing metadata only and are never embedded as learned
    numerical sensor identities.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive.")

        if n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError("n_heads must be positive and divide d_model.")

        self.d_model = d_model
        self.query = nn.Parameter(torch.empty(1, 1, d_model))
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        nn.init.normal_(self.query, mean=0.0, std=0.02)

    def forward(
        self,
        encoded_tokens: torch.Tensor,
        token_mask: torch.Tensor,
        calibration_ids: torch.Tensor,
        num_calibrations: int,
    ) -> torch.Tensor:
        """
        Return pooled calibration features with shape [B, K, D].
        """

        batch_size, _, feature_dim = encoded_tokens.shape

        if feature_dim != self.d_model:
            raise ValueError(f"encoded_tokens must have last dimension {self.d_model}.")

        pooled_features = []

        for calibration_id in range(num_calibrations):
            calibration_mask = token_mask & (calibration_ids == calibration_id)

            if torch.any(calibration_mask.sum(dim=1) == 0):
                raise ValueError(f"Every batch item must contain at least one token for calibration_id={calibration_id}.")

            query = self.query.expand(batch_size, -1, -1)

            pooled, _ = self.attention(
                query=query,
                key=encoded_tokens,
                value=encoded_tokens,
                key_padding_mask=~calibration_mask,
                need_weights=False,
            )

            pooled_features.append(pooled[:, 0])

        return torch.stack(pooled_features, dim=1)