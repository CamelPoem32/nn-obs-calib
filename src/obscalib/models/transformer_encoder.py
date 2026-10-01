"""Transformer wrapper with a configurable number of learned summary tokens."""

from dataclasses import dataclass

import torch
from torch import nn

from obscalib.config import TransformerConfig
from obscalib.data.structures import TokenBatch


@dataclass
class TransformerEncoderOutput:
    '''
    Transformer outputs.

    measurement_features:
        Encoded original measurement tokens, shape [B, N, D].

    summary_features:
        Encoded learned global summary tokens, shape [B, K, D].

    sequence:
        Complete encoded sequence [measurements | summaries].
    '''

    measurement_features: torch.Tensor
    summary_features: torch.Tensor
    sequence: torch.Tensor

@dataclass
class CalibrationTokenReadout:
    '''
    Mean-pooled Transformer features for every physical calibration key.

    features:
        [B, K_calibration, D]

    counts:
        Number of valid measurement tokens contributing to every pooled
        calibration representation, shape [B, K_calibration].
    '''

    features: torch.Tensor
    counts: torch.Tensor
    calibration_keys: tuple[str, ...]


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
            device=tokens.token_mask.device,
        )
        padding_mask = torch.cat((~tokens.token_mask, summary_padding), dim=1)

        # sequence: [B, N + K, d_model]
        sequence = self.encoder(
            encoder_input,
            src_key_padding_mask=padding_mask,
        )

        number_measurement_tokens = tokens.x.shape[1]

        encoded_measurements = sequence[:,:number_measurement_tokens,]

        encoded_summaries = sequence[:,number_measurement_tokens:,]

        return TransformerEncoderOutput(
            measurement_features=encoded_measurements,
            summary_features=encoded_summaries,
            sequence=sequence,
        )

def mean_pool_calibration_tokens(transformer_output: TransformerEncoderOutput, tokens: TokenBatch) -> CalibrationTokenReadout:
    '''
    Mean-pool encoded measurement tokens by SensorMetadata.calibration_key.

    Self-attention has already mixed information globally before this pooling.
    Therefore one IMU readout may contain information learned from LiDAR
    measurements and vice versa, while the final grouping still keeps one
    output representation per physical calibration.
    '''

    tokens.validate()

    if tokens.calibration_ids is None:
        raise ValueError("TokenBatch does not contain calibration routing metadata.")

    if not tokens.calibration_keys:
        raise ValueError("TokenBatch does not contain calibration_keys.")

    measurement_features = transformer_output.measurement_features

    if measurement_features.shape[:2] != tokens.token_mask.shape:
        raise ValueError("Transformer measurement features and TokenBatch must share [B, N].")

    pooled_features: list[torch.Tensor] = []
    token_counts: list[torch.Tensor] = []

    ##################################################
    # Pool gyro + accel together when they share the same IMU
    # calibration key, while each LiDAR calibration remains separate.
    ##################################################

    for calibration_id, calibration_key in enumerate(tokens.calibration_keys):
        calibration_mask = (
            tokens.token_mask
            & (
                tokens.calibration_ids
                == calibration_id
            )
        )

        counts = calibration_mask.sum(
            dim=1
        )

        if torch.any(
            counts == 0
        ):
            raise ValueError(f"Calibration key {calibration_key!r} has no valid measurement tokens in at least one batch item.")

        weighted_features = (
            measurement_features
            * calibration_mask.unsqueeze(
                -1
            ).to(
                dtype=measurement_features.dtype
            )
        )

        pooled = (
            weighted_features.sum(
                dim=1
            )
            / counts.unsqueeze(
                -1
            ).to(
                dtype=measurement_features.dtype
            )
        )

        pooled_features.append(
            pooled
        )

        token_counts.append(
            counts
        )

    return CalibrationTokenReadout(
        features=torch.stack(
            pooled_features,
            dim=1,
        ),
        counts=torch.stack(
            token_counts,
            dim=1,
        ),
        calibration_keys=tokens.calibration_keys,
    )
