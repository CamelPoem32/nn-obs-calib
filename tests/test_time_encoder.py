import torch

from obscalib.tokenization import TimeEncoder


def test_time_encoder_projects_scaled_scalar_timestamps() -> None:
    encoder = TimeEncoder(embedding_dim=5, time_scale=10.0)
    timestamps = torch.tensor([[0.0, 5.0, 10.0], [1.0, 2.0, 3.0]])

    encoded = encoder(timestamps)

    assert encoded.shape == (2, 3, 5)
    assert torch.isfinite(encoded).all()
