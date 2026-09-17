"""Tests for chronological sorting of encoded measurement sequences."""

import torch

from obscalib.data import MeasurementSequenceBatch, sort_measurement_sequence


def test_sort_measurement_sequence_orders_valid_tokens_by_time() -> None:
    measurements = MeasurementSequenceBatch(
        features=torch.tensor([[[30.0, 31.0], [10.0, 11.0], [20.0, 21.0]]]),
        timestamps=torch.tensor([[3.0, 1.0, 2.0]]),
        sensor_ids=torch.tensor([[3, 1, 2]], dtype=torch.long),
        measurement_types=torch.tensor([[2, 0, 1]], dtype=torch.long),
        token_mask=torch.tensor([[True, True, True]]),
    )

    sorted_measurements = sort_measurement_sequence(measurements)

    torch.testing.assert_close(sorted_measurements.timestamps, torch.tensor([[1.0, 2.0, 3.0]]))
    torch.testing.assert_close(sorted_measurements.features, torch.tensor([[[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]]))
    torch.testing.assert_close(sorted_measurements.sensor_ids, torch.tensor([[1, 2, 3]], dtype=torch.long))
    torch.testing.assert_close(sorted_measurements.measurement_types, torch.tensor([[0, 1, 2]], dtype=torch.long))


def test_sort_measurement_sequence_moves_padding_to_end() -> None:
    measurements = MeasurementSequenceBatch(
        features=torch.tensor([[[20.0], [999.0], [10.0], [998.0]]]),
        timestamps=torch.tensor([[2.0, -100.0, 1.0, -200.0]]),
        sensor_ids=torch.tensor([[2, 99, 1, 98]], dtype=torch.long),
        measurement_types=torch.tensor([[1, 4, 0, 4]], dtype=torch.long),
        token_mask=torch.tensor([[True, False, True, False]]),
    )

    sorted_measurements = sort_measurement_sequence(measurements)

    torch.testing.assert_close(sorted_measurements.timestamps, torch.tensor([[1.0, 2.0, -100.0, -200.0]]))
    assert torch.equal(sorted_measurements.token_mask, torch.tensor([[True, True, False, False]]))
    torch.testing.assert_close(sorted_measurements.features[..., 0], torch.tensor([[10.0, 20.0, 999.0, 998.0]]))


def test_sort_measurement_sequence_is_stable_for_equal_timestamps() -> None:
    measurements = MeasurementSequenceBatch(
        features=torch.tensor([[[1.0], [2.0], [3.0]]]),
        timestamps=torch.tensor([[1.0, 1.0, 1.0]]),
        sensor_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
        measurement_types=torch.tensor([[0, 1, 2]], dtype=torch.long),
        token_mask=torch.tensor([[True, True, True]]),
    )

    sorted_measurements = sort_measurement_sequence(measurements)

    torch.testing.assert_close(sorted_measurements.features, measurements.features)
    assert torch.equal(sorted_measurements.sensor_ids, measurements.sensor_ids)


def test_sorting_preserves_gradient_to_measurement_features() -> None:
    features = torch.tensor([[[3.0], [1.0], [2.0]]], requires_grad=True)

    measurements = MeasurementSequenceBatch(
        features=features,
        timestamps=torch.tensor([[3.0, 1.0, 2.0]]),
        sensor_ids=torch.zeros(1, 3, dtype=torch.long),
        measurement_types=torch.zeros(1, 3, dtype=torch.long),
        token_mask=torch.ones(1, 3, dtype=torch.bool),
    )

    sorted_measurements = sort_measurement_sequence(measurements)

    weights = torch.tensor([[[1.0], [2.0], [3.0]]])
    loss = (sorted_measurements.features * weights).sum()
    loss.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()

    # Original positions correspond to timestamps [3, 1, 2], so the gradients from sorted positions [1, 2, 3] return as [3, 1, 2].
    torch.testing.assert_close(features.grad, torch.tensor([[[3.0], [1.0], [2.0]]]))