"""Chronological sorting of complete measurement-token vectors."""

import torch

from obscalib.data.structures import MeasurementSequenceBatch


def sort_measurement_sequence(measurements: MeasurementSequenceBatch) -> MeasurementSequenceBatch:
    """
    Sort complete measurement feature vectors chronologically.

    Every row of x already contains

        [measurement | timestamp | measurement type | optional observability].

    Therefore sorting x as a whole guarantees that all information remains
    aligned automatically.

    Padding receives a temporary +inf sort key and is moved to the end.
    """

    measurements.validate()

    positive_infinity = torch.full_like(measurements.timestamps, float("inf"))
    sort_keys = torch.where(measurements.token_mask, measurements.timestamps, positive_infinity)
    permutation = torch.argsort(sort_keys, dim=1, stable=True)
    feature_indices = permutation.unsqueeze(-1).expand(-1, -1, measurements.x.shape[-1])

    return MeasurementSequenceBatch(x=torch.gather(measurements.x, 1, feature_indices), 
                                    timestamps=torch.gather(measurements.timestamps, 1, permutation), 
                                    token_mask=torch.gather(measurements.token_mask, 1, permutation))