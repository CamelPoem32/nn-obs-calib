"""Mask-aware sorting for merged learned measurement sequences."""

import torch

from obscalib.data.structures import MeasurementSequenceBatch


def sort_measurement_sequence(
    measurements: MeasurementSequenceBatch,
) -> MeasurementSequenceBatch:
    """
    Sort every minibatch item chronologically and move padding to the end.

    Gradients propagate through the reordered measurement features because
    torch.gather is differentiable with respect to its input tensor. The
    discrete permutation produced by torch.argsort is not differentiable with
    respect to timestamps, which is intentional here because measurement
    timestamps are fixed sensor observations.

    Padding entries receive a temporary +inf sorting key. Their stored
    timestamp values themselves are not modified.
    """

    measurements.validate()

    # Padding must always sort after all real measurements regardless of the
    # arbitrary timestamp values stored in padded positions.
    positive_infinity = torch.full_like(
        measurements.timestamps,
        float("inf"),
    )
    sort_keys = torch.where(
        measurements.token_mask,
        measurements.timestamps,
        positive_infinity,
    )

    # Stable sorting gives deterministic ordering when two sensors have exactly
    # the same timestamp.
    permutation = torch.argsort(
        sort_keys,
        dim=1,
        stable=True,
    )  # [B, N]

    feature_indices = permutation.unsqueeze(-1).expand(
        -1,
        -1,
        measurements.features.shape[-1],
    )

    return MeasurementSequenceBatch(
        features=torch.gather(
            measurements.features,
            1,
            feature_indices,
        ),
        timestamps=torch.gather(
            measurements.timestamps,
            1,
            permutation,
        ),
        sensor_ids=torch.gather(
            measurements.sensor_ids,
            1,
            permutation,
        ),
        measurement_types=torch.gather(
            measurements.measurement_types,
            1,
            permutation,
        ),
        token_mask=torch.gather(
            measurements.token_mask,
            1,
            permutation,
        ),
    )