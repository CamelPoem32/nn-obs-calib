"""Sequential propagation of predicted calibration states across windows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from obscalib.calibration.state import CalibrationState
from obscalib.data.structures import WindowBatch
from obscalib.pipeline.window_step import WindowStep, WindowStepResult


@dataclass
class RolloutResult:
    """
    Complete result of a sequential calibration rollout.

    steps:
        Ordered results for every processed temporal window.

    final_calibration:
        Calibration dictionary produced after the last window. If windows is
        empty, this is the supplied initial calibration.
    """

    steps: list[WindowStepResult]
    final_calibration: dict[str, CalibrationState]


class Rollout:
    """
    Sequentially carry predicted calibration state across temporal windows.

    WindowStep owns all per-window scientific and learned processing. Rollout
    only propagates each resulting calibration dictionary into the next window.

    Batch elements are assumed to preserve trajectory identity between
    consecutive WindowBatch objects.
    """

    def __init__(self, window_step: WindowStep) -> None:
        self.window_step = window_step

    def __call__(self, windows: Sequence[WindowBatch], initial_calibration: dict[str, CalibrationState]) -> RolloutResult:
        """Run all windows sequentially from the supplied initial calibration."""

        calibration = dict(initial_calibration)

        if not calibration:
            raise ValueError("initial_calibration must contain at least one calibration state.")

        for calibration_key, state in calibration.items():
            try:
                state.validate()
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid initial calibration state {calibration_key!r}.") from error

        step_results: list[WindowStepResult] = []

        for window in windows:
            step_result = self.window_step(window, calibration=calibration)
            step_results.append(step_result)
            calibration = step_result.next_calibration

        return RolloutResult(steps=step_results, final_calibration=calibration)