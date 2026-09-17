"""Differentiable update of the externally carried calibration state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from obscalib.calibration.state import CalibrationState
from obscalib.geometry.lie import se3_exp

if TYPE_CHECKING:
    from obscalib.models.structures import CalibrationPrediction


class CalibrationUpdater:
    """
    Apply deterministic calibration corrections predicted by one model head.

    Spatial convention:

        delta_xi = [phi, rho]

        T_next = Exp_SE3(delta_xi) @ T_current.

    Temporal convention:

        tau_next = tau_current + delta_tau.

    change_event_logit and change_time are auxiliary supervised outputs and do
    not gate application of delta_xi or delta_tau.
    """

    def update(self, state: CalibrationState, prediction: CalibrationPrediction) -> CalibrationState:
        """Return the next differentiable carried calibration state."""

        state.validate()

        if prediction.delta_xi.ndim != 2 or prediction.delta_xi.shape[1] != 6:
            raise ValueError("prediction.delta_xi must have shape [B, 6].")

        if prediction.delta_tau.ndim != 2 or prediction.delta_tau.shape[1] != 1:
            raise ValueError("prediction.delta_tau must have shape [B, 1].")

        batch_size = state.transform.shape[0]

        if prediction.delta_xi.shape[0] != batch_size or prediction.delta_tau.shape[0] != batch_size:
            raise ValueError("Calibration state and prediction must share batch size.")

        if prediction.delta_xi.device != state.transform.device or prediction.delta_tau.device != state.time_offset.device:
            raise ValueError("Calibration state and prediction must be on the same device.")

        if prediction.delta_xi.dtype != state.transform.dtype or prediction.delta_tau.dtype != state.time_offset.dtype:
            raise ValueError("Calibration state and prediction must use matching dtypes.")

        delta_transform = se3_exp(prediction.delta_xi)

        updated_state = CalibrationState(transform=delta_transform @ state.transform, time_offset=state.time_offset + prediction.delta_tau)
        updated_state.validate()

        return updated_state

    @staticmethod
    def _validate_prediction(
        state: CalibrationState,
        prediction: CalibrationPrediction,
    ) -> None:
        """Validate prediction tensors required by the state transition."""

        if prediction.delta_xi.ndim != 2 or prediction.delta_xi.shape[1] != 6:
            raise ValueError("delta_xi must have shape [B, 6].")

        if prediction.delta_tau.ndim != 2 or prediction.delta_tau.shape[1] != 1:
            raise ValueError("delta_tau must have shape [B, 1].")

        batch_size = state.transform.shape[0]

        if prediction.delta_xi.shape[0] != batch_size:
            raise ValueError("delta_xi and calibration state must share batch size.")

        if prediction.delta_tau.shape[0] != batch_size:
            raise ValueError("delta_tau and calibration state must share batch size.")

        if prediction.delta_xi.device != state.transform.device:
            raise ValueError("delta_xi and transform must be on the same device.")

        if prediction.delta_tau.device != state.time_offset.device:
            raise ValueError("delta_tau and time_offset must be on the same device.")

        if prediction.delta_xi.dtype != state.transform.dtype:
            raise ValueError("delta_xi and transform must have the same dtype.")

        if prediction.delta_tau.dtype != state.time_offset.dtype:
            raise ValueError("delta_tau and time_offset must have the same dtype.")