"""Tests for chronological sorting of complete measurement-token vectors."""

import torch

from obscalib.data.sorting import sort_measurement_sequence
from obscalib.data.structures import MeasurementSequenceBatch


def test_sort_measurement_sequence_orders_valid_tokens_by_time() -> None:
    measurements = MeasurementSequenceBatch(
        x=torch.tensor([[[30.0, 31.0, 3.0, 2.0], [10.0, 11.0, 1.0, 0.0], [20.0, 21.0, 2.0, 1.0]]]),
        timestamps=torch.tensor([[3.0, 1.0, 2.0]]),
        token_mask=torch.tensor([[True, True, True]]),
    )

    sorted_measurements = sort_measurement_sequence(measurements)

    torch.testing.assert_close(sorted_measurements.timestamps, torch.tensor([[1.0, 2.0, 3.0]]))
    torch.testing.assert_close(sorted_measurements.x, torch.tensor([[[10.0, 11.0, 1.0, 0.0], [20.0, 21.0, 2.0, 1.0], [30.0, 31.0, 3.0, 2.0]]]))
    assert torch.equal(sorted_measurements.token_mask, torch.tensor([[True, True, True]]))


def test_sort_measurement_sequence_moves_padding_to_end() -> None:
    measurements = MeasurementSequenceBatch(
        x=torch.tensor([[[20.0], [999.0], [10.0], [998.0]]]),
        timestamps=torch.tensor([[2.0, -100.0, 1.0, -200.0]]),
        token_mask=torch.tensor([[True, False, True, False]]),
    )

    sorted_measurements = sort_measurement_sequence(measurements)

    assert torch.equal(sorted_measurements.token_mask, torch.tensor([[True, True, False, False]]))

    # Valid measurements are sorted chronologically.
    torch.testing.assert_close(sorted_measurements.x[0, :2], torch.tensor([[10.0], [20.0]]))
    torch.testing.assert_close(sorted_measurements.timestamps[0, :2], torch.tensor([1.0, 2.0]))

    # Padding is moved behind all real measurements.
    # Stable sorting keeps the original relative order of padding entries.
    torch.testing.assert_close(sorted_measurements.x[0, 2:], torch.tensor([[999.0], [998.0]]))


def test_sort_measurement_sequence_is_stable_for_equal_timestamps() -> None:
    measurements = MeasurementSequenceBatch(
        x=torch.tensor([[[1.0], [2.0], [3.0]]]),
        timestamps=torch.tensor([[1.0, 1.0, 1.0]]),
        token_mask=torch.tensor([[True, True, True]]),
    )

    sorted_measurements = sort_measurement_sequence(measurements)

    torch.testing.assert_close(sorted_measurements.x, measurements.x)
    torch.testing.assert_close(sorted_measurements.timestamps, measurements.timestamps)
    assert torch.equal(sorted_measurements.token_mask, measurements.token_mask)


def test_sorting_preserves_gradient_to_complete_token_vectors() -> None:
    x = torch.tensor([[[3.0], [1.0], [2.0]]], requires_grad=True)

    measurements = MeasurementSequenceBatch(
        x=x,
        timestamps=torch.tensor([[3.0, 1.0, 2.0]]),
        token_mask=torch.ones(1, 3, dtype=torch.bool),
    )

    sorted_measurements = sort_measurement_sequence(measurements)

    torch.testing.assert_close(sorted_measurements.x, torch.tensor([[[1.0], [2.0], [3.0]]]))

    # Different coefficients make the expected inverse permutation visible in
    # the gradient rather than merely checking that some gradient exists.
    loss = 10.0 * sorted_measurements.x[0, 0, 0] + 20.0 * sorted_measurements.x[0, 1, 0] + 30.0 * sorted_measurements.x[0, 2, 0]
    loss.backward()

    assert x.grad is not None

    # Original order was [t=3, t=1, t=2], so sorted coefficients [10,20,30]
    # map back to original entries as [30,10,20].
    torch.testing.assert_close(x.grad, torch.tensor([[[30.0], [10.0], [20.0]]]))