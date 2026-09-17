"""Conversion of carried calibration states to neural calibration contexts."""

import torch

from obscalib.calibration.state import CalibrationState
from obscalib.geometry.lie import se3_log


CALIBRATION_CONTEXT_DIM = 7


def calibration_state_to_context(state: CalibrationState) -> torch.Tensor:
    """
    Convert one carried calibration state to the context consumed by a head.

    The context ordering is

        [phi_x, phi_y, phi_z, rho_x, rho_y, rho_z, tau],

    where

        [phi, rho] = Log_SE3(T_sensor_in_world).

    Input:
        transform:   [B, 4, 4]
        time_offset: [B, 1]

    Output:
        context: [B, 7]
    """

    state.validate()

    if state.transform.device != state.time_offset.device:
        raise ValueError("Calibration transform and time offset must be on the same device.")

    if state.transform.dtype != state.time_offset.dtype:
        raise ValueError("Calibration transform and time offset must have the same dtype.")

    if not torch.is_floating_point(state.time_offset):
        raise TypeError("Calibration time offset must have floating-point dtype.")

    calibration_tangent = se3_log(state.transform)

    return torch.cat((calibration_tangent, state.time_offset), dim=-1)