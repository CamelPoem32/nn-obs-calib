"""Mask-aware sorting for merged canonical measurement sequences."""

import torch

from obscalib.data.structures import CanonicalMeasurements


def sort_canonical_measurements(
    measurements: CanonicalMeasurements,
) -> CanonicalMeasurements:
    """Sort each batch item by time while moving invalid padding to the end.

    Invalid entries receive a temporary positive-infinity sort key. Their
    stored timestamp values are preserved; only their position changes.
    """

    measurements.validate()

    # Invalid entries sort after every finite valid timestamp, independent of
    # the arbitrary value used for padding.
    positive_infinity = torch.full_like(measurements.timestamps, float("inf"))
    sort_keys = torch.where(
        measurements.valid_mask,
        measurements.timestamps,
        positive_infinity,
    )
    permutation = torch.argsort(sort_keys, dim=1, stable=True)  # [B, N]

    # Apply the identical sequence permutation to every aligned field.
    value_indices = permutation.unsqueeze(-1).expand(
        -1, -1, measurements.values.size(-1)
    )
    return CanonicalMeasurements(
        values=torch.gather(measurements.values, 1, value_indices),
        timestamps=torch.gather(measurements.timestamps, 1, permutation),
        sensor_ids=torch.gather(measurements.sensor_ids, 1, permutation),
        type_ids=torch.gather(measurements.type_ids, 1, permutation),
        valid_mask=torch.gather(measurements.valid_mask, 1, permutation),
    )
