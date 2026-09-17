"""Operations for advancing an externally carried calibration state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from obscalib.calibration.state import CalibrationState
from obscalib.geometry import se3_exp

if TYPE_CHECKING:
    from obscalib.models.structures import CalibrationPrediction


class CalibrationUpdater:
    """Apply predicted left-multiplicative calibration corrections."""

    def update(
        self,
        state: CalibrationState,
        prediction: CalibrationPrediction,
    ) -> CalibrationState:
        """
        Advance the carried spatial and temporal calibration state.

        The spatial correction uses the established tangent convention

            delta_xi = [phi, rho],

        where phi is the rotation-vector tangent component in radians and rho
        is the translational tangent component of se(3).

        The spatial correction is left-multiplicative:

            T_next = Exp_SE3(delta_xi) @ T_current.

        The temporal correction is additive:

            tau_next = tau_current + delta_tau.

        change_event_logit and change_time are auxiliary supervised outputs.
        They do not gate or otherwise modify the calibration-state update.
        """

        state.validate()
        self._validate_prediction(state, prediction)

        # Convert the predicted [phi, rho] tangent correction to SE(3).
        # delta_transform: [B, 4, 4]
        delta_transform = se3_exp(prediction.delta_xi)

        # Apply the correction on the left using the established convention.
        # updated_transform: [B, 4, 4]
        updated_transform = delta_transform @ state.transform

        # Apply the predicted additive temporal-offset correction.
        # updated_time_offset: [B, 1]
        updated_time_offset = state.time_offset + prediction.delta_tau

        return CalibrationState(
            transform=updated_transform,
            time_offset=updated_time_offset,
        )

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