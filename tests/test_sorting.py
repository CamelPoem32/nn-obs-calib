import torch

from obscalib.data import CanonicalMeasurements, sort_canonical_measurements


def test_sorting_applies_one_permutation_to_every_field() -> None:
    measurements = CanonicalMeasurements(
        values=torch.tensor([[[30.0], [10.0], [20.0]]]),
        timestamps=torch.tensor([[3.0, 1.0, 2.0]]),
        sensor_ids=torch.tensor([[3, 1, 2]]),
        type_ids=torch.tensor([[30, 10, 20]]),
        valid_mask=torch.tensor([[True, True, True]]),
    )

    sorted_measurements = sort_canonical_measurements(measurements)

    assert torch.equal(sorted_measurements.timestamps, torch.tensor([[1.0, 2.0, 3.0]]))
    assert torch.equal(sorted_measurements.values.squeeze(-1), torch.tensor([[10.0, 20.0, 30.0]]))
    assert torch.equal(sorted_measurements.sensor_ids, torch.tensor([[1, 2, 3]]))
    assert torch.equal(sorted_measurements.type_ids, torch.tensor([[10, 20, 30]]))


def test_sorting_moves_invalid_entries_to_end_and_preserves_them() -> None:
    measurements = CanonicalMeasurements(
        values=torch.tensor([[[99.0], [20.0], [10.0], [98.0]]]),
        timestamps=torch.tensor([[-100.0, 2.0, 1.0, -200.0]]),
        sensor_ids=torch.tensor([[9, 2, 1, 8]]),
        type_ids=torch.tensor([[9, 2, 1, 8]]),
        valid_mask=torch.tensor([[False, True, True, False]]),
    )

    result = sort_canonical_measurements(measurements)

    assert torch.equal(result.valid_mask, torch.tensor([[True, True, False, False]]))
    assert torch.equal(result.timestamps, torch.tensor([[1.0, 2.0, -100.0, -200.0]]))
    assert torch.equal(result.values.squeeze(-1), torch.tensor([[10.0, 20.0, 99.0, 98.0]]))
