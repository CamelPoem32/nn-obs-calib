"""Transformer wrapper with a configurable number of learned summary tokens."""

from dataclasses import dataclass

import torch
from torch import nn

from obscalib.config import TransformerConfig
from obscalib.data.structures import TokenBatch


@dataclass
class TransformerEncoderOutput:
    """Encoded summaries [B, K, D] and complete sequence [B, K + N, D]."""

    summary_features: torch.Tensor
    sequence: torch.Tensor


class SummaryTokenTransformerEncoder(nn.Module):
    """
    Encode prepared measurement tokens with K learned summary tokens.

    Temporal/positional information is expected to already be encoded in
    TokenBatch.x by the tokenization stage. This module does not add a
    separate positional embedding.
    """
    

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        if config.input_dim == config.d_model:
            self.input_projection: nn.Module = nn.Identity()
        else:
            self.input_projection = nn.Linear(config.input_dim, config.d_model)

        self.learned_summary_tokens = nn.Parameter(
            torch.empty(1, config.num_summary_tokens, config.d_model)
        )
        # Initialize learned summary tokens with small random values; std=0.02 follows
        # a common Transformer embedding convention and avoids a large initial token bias
        nn.init.normal_(self.learned_summary_tokens, mean=0.0, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)

    def forward(self, tokens: TokenBatch) -> TransformerEncoderOutput:
        """Project and encode token sequences while respecting valid masks."""

        tokens.validate()
        if tokens.x.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"tokens.x last dimension must be {self.config.input_dim}."
            )

        # Project heterogeneous token features to Transformer dimension.
        # projected: [B, N, d_model]
        projected = self.input_projection(tokens.x)
        batch_size = projected.shape[0]

        # Append K learnable global summary tokens.
        # summaries: [B, K, d_model]
        summaries = self.learned_summary_tokens.expand(batch_size, -1, -1)
        encoder_input = torch.cat((projected, summaries), dim=1)

        # PyTorch uses True for padding, opposite to this package's valid mask.
        # Summary tokens are always valid and therefore never padding.
        summary_padding = torch.zeros(
            batch_size,
            self.config.num_summary_tokens,
            dtype=torch.bool,
            device=tokens.valid_mask.device,
        )
        padding_mask = torch.cat((~tokens.valid_mask, summary_padding), dim=1)

        # sequence: [B, N + K, d_model]
        sequence = self.encoder(
            encoder_input,
            src_key_padding_mask=padding_mask,
        )

        # Keep summaries explicit; downstream flattening is intentionally simple.
        encoded_summaries = sequence[:, -self.config.num_summary_tokens :]
        return TransformerEncoderOutput(
            summary_features=encoded_summaries,
            sequence=sequence,
        )
