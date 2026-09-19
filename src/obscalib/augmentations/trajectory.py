"""Evaluation of continuous synthetic calibration trajectories."""

from __future__ import annotations

import torch

from obscalib.augmentations.profiles import evaluate_transition_profiles
from obscalib.augmentations.structures import CalibrationEvent
from obscalib.calibration.state import CalibrationState
from obscalib.geometry.lie import se3_exp


def evaluate_calibration_trajectory(
    pre_event: CalibrationState,
    event: CalibrationEvent,
    timestamps_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate the true calibration trajectory at arbitrary relative timestamps.

    Args:
        pre_event:
            True calibration state before the synthetic change event.

            transform has shape [B, 4, 4] and represents T_WS.
            time_offset has shape [B, 1].

        event:
            Synthetic calibration event describing its transition profile,
            spatial increment delta_xi, and temporal increment delta_tau.

        timestamps_s:
            Timestamps relative to the beginning of the window with shape [B, N].

            Different sensor streams may call this function using different
            timestamps while sharing the same underlying CalibrationEvent.

    Returns:
        transforms:
            Time-varying true sensor-to-world transforms T_WS(t) with
            shape [B, N, 4, 4].

        time_offsets:
            Time-varying true temporal offsets tau(t) with shape [B, N, 1].

    Spatial interpolation follows the left-multiplicative event convention

        T_WS(t) = Exp(s(t) * delta_xi) @ T_WS_pre,

    where delta_xi = [phi, rho].

    Temporal interpolation is additive:

        tau(t) = tau_pre + s(t) * delta_tau.

    The normalized transition progress s(t) is evaluated by profiles.py.
    """

    _validate_inputs(
        pre_event=pre_event,
        event=event,
        timestamps_s=timestamps_s,
    )

    progress = evaluate_transition_profiles(
        timestamps_s=timestamps_s,
        change_time_s=event.change_time_s,
        transition_duration_s=event.transition_duration_s,
        profiles=event.profiles,
    )

    # Scale the complete event tangent increment by the normalized transition
    # progress independently at every requested timestamp.
    xi_t = progress[..., None] * event.delta_xi[:, None, :]

    # Convert the time-varying tangent increments to SE3 and apply them using
    # the project's established left-multiplicative calibration convention.
    delta_T_t = se3_exp(xi_t)

    transforms = delta_T_t @ pre_event.transform[:, None, :, :]

    # The temporal calibration follows the same normalized profile but remains
    # an ordinary additive scalar offset.
    time_offsets = pre_event.time_offset[:, None, :] + progress[..., None] * event.delta_tau[:, None, :]

    return transforms, time_offsets


def evaluate_calibration_progress(
    event: CalibrationEvent,
    timestamps_s: torch.Tensor,
) -> torch.Tensor:
    """Evaluate only the normalized transition progress s(t).

    Returns:
        Tensor with shape [B, N] and values in [0, 1].

    This helper is useful when a renderer needs the scalar event realization
    directly in addition to T_WS(t) or tau(t).
    """

    event.validate()
    _validate_timestamps(
        timestamps_s=timestamps_s,
        batch_size=event.occurred.shape[0],
        device=event.delta_xi.device,
        dtype=event.delta_xi.dtype,
    )

    return evaluate_transition_profiles(
        timestamps_s=timestamps_s,
        change_time_s=event.change_time_s,
        transition_duration_s=event.transition_duration_s,
        profiles=event.profiles,
    )


def _validate_inputs(
    pre_event: CalibrationState,
    event: CalibrationEvent,
    timestamps_s: torch.Tensor,
) -> None:
    """Validate trajectory state, event metadata, and requested timestamps."""

    pre_event.validate()
    event.validate()

    batch_size = pre_event.transform.shape[0]

    if event.occurred.shape[0] != batch_size:
        raise ValueError("pre_event and event must share batch size.")

    if not torch.is_floating_point(pre_event.transform):
        raise TypeError("pre_event.transform must have floating-point dtype.")

    if not torch.is_floating_point(pre_event.time_offset):
        raise TypeError("pre_event.time_offset must have floating-point dtype.")

    if pre_event.transform.device != pre_event.time_offset.device:
        raise ValueError("pre_event.transform and pre_event.time_offset must be on the same device.")

    if pre_event.transform.dtype != pre_event.time_offset.dtype:
        raise ValueError("pre_event.transform and pre_event.time_offset must have the same dtype.")

    if event.delta_xi.device != pre_event.transform.device or event.delta_tau.device != pre_event.transform.device:
        raise ValueError("Event tensors and pre_event calibration must be on the same device.")

    if event.delta_xi.dtype != pre_event.transform.dtype or event.delta_tau.dtype != pre_event.transform.dtype:
        raise ValueError("Event tensors and pre_event calibration must have the same dtype.")

    _validate_timestamps(
        timestamps_s=timestamps_s,
        batch_size=batch_size,
        device=pre_event.transform.device,
        dtype=pre_event.transform.dtype,
    )


def _validate_timestamps(
    timestamps_s: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Validate window-relative timestamps used to evaluate a trajectory."""

    if timestamps_s.ndim != 2:
        raise ValueError("timestamps_s must have shape [B, N].")

    if timestamps_s.shape[0] != batch_size:
        raise ValueError("timestamps_s and calibration trajectory must share batch size.")

    if not torch.is_floating_point(timestamps_s):
        raise TypeError("timestamps_s must have floating-point dtype.")

    if timestamps_s.device != device:
        raise ValueError("timestamps_s and calibration trajectory must be on the same device.")

    if timestamps_s.dtype != dtype:
        raise TypeError("timestamps_s and calibration trajectory must have the same dtype.")

    if not torch.all(torch.isfinite(timestamps_s)):
        raise ValueError("timestamps_s must contain only finite values.")

    if torch.any(timestamps_s < 0.0):
        raise ValueError("timestamps_s must be nonnegative because they are relative to the window start.")