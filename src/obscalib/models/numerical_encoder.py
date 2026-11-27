"""Shared numerical feature encoder with exact masking after the MLP."""
import torch
from torch import nn
from obscalib.config import NumericalEncoderConfig


class NumericalEncoder(nn.Module):
    def __init__(self, input_dim: int, config: NumericalEncoderConfig | None = None) -> None:
        super().__init__()
        config = config or NumericalEncoderConfig()
        self.input_dim = input_dim
        self.output_dim = config.output_dim
        self.mlp = nn.Sequential(nn.Linear(input_dim, config.hidden_dim),nn.GELU(),nn.Linear(config.hidden_dim,config.output_dim))
        if config.zero_output_init:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Encode valid pairs; reference/invalid embeddings are always zero."""
        if values.ndim != 3 or values.shape[-1] != self.input_dim or valid.shape != values.shape[:2] or valid.dtype != torch.bool or valid.device != values.device:
            raise ValueError('Expected numerical values [B,C,F] and boolean mask [B,C].')
        safe = torch.where(valid[...,None],values,0.)
        z = self.mlp(safe)
        return torch.where(valid[...,None],z,0.)
