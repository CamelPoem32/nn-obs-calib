"""Calibration state representation, contexts, and state updates."""

from obscalib.calibration.context import CALIBRATION_CONTEXT_DIM, calibration_state_to_context
from obscalib.calibration.state import CalibrationState
from obscalib.calibration.update import CalibrationUpdater


__all__ = [
    "CALIBRATION_CONTEXT_DIM",
    "CalibrationState",
    "CalibrationUpdater",
    "calibration_state_to_context",
]