"""Identity-initialized feature fusion without an architectural skip connection."""
import torch
from torch import nn
from obscalib.config import CalibrationFeatureFusionConfig


class CalibrationFeatureFusion(nn.Module):
    def __init__(self, neural_dim: int, numerical_dim: int, config: CalibrationFeatureFusionConfig | None = None) -> None:
        super().__init__()
        config = config or CalibrationFeatureFusionConfig()
        hidden = config.hidden_dim or 2*neural_dim
        self.neural_dim, self.numerical_dim = neural_dim,numerical_dim
        self.mlp = nn.Sequential(nn.Linear(neural_dim+numerical_dim,hidden),nn.GELU(),nn.Linear(hidden,neural_dim))
        if config.identity_init:
            self.initialize_identity(config.numerical_init_scale)

    @torch.no_grad()
    def initialize_identity(self, numerical_init_scale: float = 0.1) -> None:
        """Initially f(h,z)=h+A z; A nonzero permits numerical learning.

        GELU(x)-GELU(-x)=x, so a paired hidden layer implements this exactly
        in real arithmetic. Additional hidden units retain random first-layer
        weights and zero output weights so they can learn subsequently.
        """
        d,n = self.neural_dim,self.numerical_dim
        first,last = self.mlp[0],self.mlp[2]
        if first.out_features < 2*d or numerical_init_scale <= 0:
            raise ValueError('Identity initialization requires >=2*neural_dim hidden units and positive numerical scale.')
        eye = torch.eye(d,device=first.weight.device,dtype=first.weight.dtype)
        A = torch.randn(d,n,device=eye.device,dtype=eye.dtype)*(numerical_init_scale/n**.5)
        W = torch.cat((eye,A),dim=-1)
        first.weight[:2*d].copy_(torch.cat((W,-W),dim=0))
        first.bias[:2*d].zero_()
        last.weight.zero_(); last.bias.zero_()
        last.weight[:,:2*d].copy_(torch.cat((eye,-eye),dim=-1))

    def forward(self, combined: torch.Tensor) -> torch.Tensor:
        if combined.shape[-1] != self.neural_dim+self.numerical_dim:
            raise ValueError('Wrong concatenated feature dimension.')
        return self.mlp(combined)
