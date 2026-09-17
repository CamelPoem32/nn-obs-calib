"""Explicit future sequential propagation of calibration state."""

from collections.abc import Sequence

from obscalib.calibration.state import CalibrationState
from obscalib.data.structures import WindowBatch
from obscalib.pipeline.window_step import WindowStep


class Rollout:
    """Carry calibration state across windows without mutating the neural model."""

    def __init__(self, window_step: WindowStep) -> None:
        self.window_step = window_step

    def __call__(
        self,
        windows: Sequence[WindowBatch],
        initial_calibration: dict[str, CalibrationState],
    ) -> object:
        """Run sequential windows after WindowStep output semantics are fixed."""

        # TODO: Define a typed WindowStep result containing predictions and the
        # next externally carried calibration dictionary.
        raise NotImplementedError("Sequential rollout is not implemented.")
