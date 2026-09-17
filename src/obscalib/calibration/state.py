"""Canonical definition of the externally carried calibration state."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CalibrationState:
    """
    Batched carried calibration state.

    transform:
        Calibration transform with shape [B, 4, 4].

    time_offset:
        Calibration time offset tau with shape [B, 1].
    """

    transform: torch.Tensor
    time_offset: torch.Tensor

    def validate(self) -> None:
        """Validate the currently established batch-level state convention."""

        if self.transform.ndim != 3 or self.transform.shape[-2:] != (4, 4):
            raise ValueError("transform must have shape [B, 4, 4].")
        if self.time_offset.ndim != 2 or self.time_offset.shape[1] != 1:
            raise ValueError("time_offset must have shape [B, 1].")
        if self.transform.shape[0] != self.time_offset.shape[0]:
            raise ValueError("transform and time_offset must share batch size.")

__all__ = ["CalibrationState"]
