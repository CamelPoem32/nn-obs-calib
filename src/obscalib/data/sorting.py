"""Chronological sorting of complete measurement-token vectors."""

import torch

from obscalib.data.structures import MeasurementSequenceBatch


def sort_measurement_sequence(measurements: MeasurementSequenceBatch) -> MeasurementSequenceBatch:
    """Sort complete measurement feature vectors and aligned token metadata chronologically."""

    measurements.validate()

    positive_infinity = torch.full_like(measurements.timestamps, float("inf"))
    sort_keys = torch.where(measurements.token_mask, measurements.timestamps, positive_infinity)
    permutation = torch.argsort(sort_keys, dim=1, stable=True)
    feature_indices = permutation.unsqueeze(-1).expand(-1, -1, measurements.x.shape[-1])
    sorted_calibration_ids = None if measurements.calibration_ids is None else torch.gather(measurements.calibration_ids, 1, permutation)

    return MeasurementSequenceBatch(
        x=torch.gather(measurements.x, 1, feature_indices),
        timestamps=torch.gather(measurements.timestamps, 1, permutation),
        token_mask=torch.gather(measurements.token_mask, 1, permutation),
        calibration_ids=sorted_calibration_ids,
        calibration_keys=measurements.calibration_keys,
    )