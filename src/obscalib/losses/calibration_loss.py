"""Future geometric and temporal calibration-loss definitions."""

from torch import nn

from obscalib.config import LossWeights
from obscalib.losses.structures import LossComponents


class CalibrationLoss(nn.Module):
    """Compute calibration terms once their scientific definitions are fixed."""

    def __init__(self, weights: LossWeights) -> None:
        super().__init__()
        self.weights = weights

    def forward(self, *args: object, **kwargs: object) -> LossComponents:
        """Compute named terms after geometric target semantics are established."""

        # TODO: Define rotation, translation, temporal, prior, and consistency
        # terms. Aggregation of precomputed terms is already implemented.
        raise NotImplementedError("Calibration loss terms are not implemented.")
