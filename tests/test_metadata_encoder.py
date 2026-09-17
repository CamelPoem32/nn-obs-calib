import torch

from obscalib.tokenization import MetadataEncoder


def test_metadata_encoder_keeps_identity_and_type_dimensions_distinct() -> None:
    encoder = MetadataEncoder(
        num_sensor_ids=5,
        num_sensor_types=3,
        sensor_embedding_dim=4,
        type_embedding_dim=6,
    )
    sensor_ids = torch.tensor([[0, 1, 2], [2, 3, 4]])
    type_ids = torch.tensor([[0, 1, 2], [2, 1, 0]])

    encoded = encoder(sensor_ids, type_ids)

    assert encoded.shape == (2, 3, 10)
