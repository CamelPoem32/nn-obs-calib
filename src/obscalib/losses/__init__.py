"""Calibration loss definitions and named loss components."""

from obscalib.losses.calibration_loss import CalibrationLoss
from obscalib.losses.structures import LossComponents, combine_loss_components


__all__ = ["CalibrationLoss", "LossComponents", "combine_loss_components"]