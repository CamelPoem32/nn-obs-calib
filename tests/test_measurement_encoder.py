import torch

from obscalib.tokenization import MeasurementEncoder


def test_measurement_encoder_shape_and_gradient_flow() -> None:
    encoder = MeasurementEncoder(input_dim=6, embedding_dim=9)
    measurements = torch.randn(2, 5, 6, requires_grad=True)

    encoded = encoder(measurements)
    encoded.square().mean().backward()

    assert encoded.shape == (2, 5, 9)
    assert measurements.grad is not None
    assert encoder.projection.weight.grad is not None
