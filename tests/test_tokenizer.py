import pytest
import torch

from obscalib.data import MeasurementSequenceBatch
from obscalib.observability import ObservabilityResult
from obscalib.tokenization import (
    MeasurementEncoder,
    MetadataEncoder,
    TimeEncoder,
    Tokenizer,
)


def test_tokenizer_concatenates_and_projects_prepared_features() -> None:
    tokenizer = Tokenizer(
        measurement_encoder=MeasurementEncoder(3, 4),
        metadata_encoder=MetadataEncoder(4, 3, 2, 3),
        time_encoder=TimeEncoder(2),
        observability_dim=3,
        output_dim=7,
    )
    measurements = MeasurementSequenceBatch(
        features=torch.randn(2, 5, 3),
        timestamps=torch.randn(2, 5),
        sensor_ids=torch.randint(0, 4, (2, 5)),
        measurement_types=torch.randint(0, 3, (2, 5)),
        token_mask=torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, True]]
        ),
    )
    observability = ObservabilityResult(features=torch.randn(2, 5, 3))

    tokens = tokenizer(measurements, observability)

    assert tokens.x.shape == (2, 5, 7)
    assert torch.equal(tokens.token_mask, measurements.token_mask)
    assert torch.equal(tokens.sensor_ids, measurements.sensor_ids)
    assert torch.equal(tokens.measurement_types, measurements.measurement_types)


def test_tokenizer_requires_nn_ready_observability_features() -> None:
    tokenizer = Tokenizer(
        measurement_encoder=MeasurementEncoder(2, 3),
        metadata_encoder=MetadataEncoder(2, 2, 2, 2),
        time_encoder=TimeEncoder(2),
        observability_dim=1,
    )
    measurements = MeasurementSequenceBatch(
        features=torch.randn(1, 2, 2),
        timestamps=torch.randn(1, 2),
        sensor_ids=torch.zeros(1, 2, dtype=torch.long),
        measurement_types=torch.zeros(1, 2, dtype=torch.long),
        token_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    observability = ObservabilityResult(raw={"matrix": torch.eye(2)})

    with pytest.raises(ValueError, match="ObservabilityMapper"):
        tokenizer(measurements, observability)
