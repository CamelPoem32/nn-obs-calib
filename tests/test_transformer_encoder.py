import pytest
import torch
from torch import nn

from obscalib.config import TransformerConfig
from obscalib.data import TokenBatch
from obscalib.models import SummaryTokenTransformerEncoder


def make_tokens() -> TokenBatch:
    return TokenBatch(x=torch.randn(2, 4, 5), token_mask=torch.tensor([[True, True, False, False], [True, True, True, False]]))


@pytest.mark.parametrize("num_summary_tokens", [1, 3])
def test_transformer_summary_token_shapes(num_summary_tokens: int) -> None:
    config = TransformerConfig(
        input_dim=5,
        d_model=8,
        n_heads=2,
        num_layers=2,
        dim_feedforward=16,
        num_summary_tokens=num_summary_tokens,
    )
    encoder = SummaryTokenTransformerEncoder(config)

    output = encoder(make_tokens())

    assert output.summary_features.shape == (2, num_summary_tokens, 8)
    assert output.sequence.shape == (2, num_summary_tokens + 4, 8)
    assert torch.isfinite(output.sequence).all()


class CaptureEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.padding_mask: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.padding_mask = src_key_padding_mask
        return x


def test_transformer_extends_padding_mask_for_always_valid_summaries() -> None:
    config = TransformerConfig(
        input_dim=5,
        d_model=8,
        n_heads=2,
        num_layers=1,
        dim_feedforward=16,
        num_summary_tokens=3,
    )
    encoder = SummaryTokenTransformerEncoder(config)
    capture = CaptureEncoder()
    encoder.encoder = capture

    encoder(make_tokens())

    assert capture.padding_mask is not None
    assert capture.padding_mask.shape == (2, 7)
    assert torch.equal(capture.padding_mask[:, :4], ~make_tokens().token_mask)
    assert not capture.padding_mask[:, 4:].any()
