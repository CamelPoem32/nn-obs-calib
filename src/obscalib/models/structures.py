"""Explicit tensor contracts exchanged by calibration pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CalibrationPrediction:
    """
    Deterministic output for one calibration head.

    change_event_logit:
        Auxiliary supervised change-event score. It does not gate the predicted
        calibration corrections.

    change_time:
        Predicted calibration-change time relative to the current window start.

    delta_xi:
        Spatial calibration correction with shape [B, 6] and ordering

            [phi, rho],

        where phi is the SO(3) rotation vector and rho is the standard SE(3)
        translational tangent component.

        Corrections use the left-multiplicative convention

            T_next = Exp(delta_xi) @ T_current.

    delta_tau:
        Additive temporal correction with shape [B, 1]:

            tau_next = tau_current + delta_tau.

    Covariance prediction is intentionally absent from the current deterministic
    model.
    """

    change_event_logit: torch.Tensor
    change_time: torch.Tensor
    delta_xi: torch.Tensor
    delta_tau: torch.Tensor


@dataclass
class ModelOutput:
    """Named calibration predictions plus an optional analysis shared_features."""

    predictions: dict[str, CalibrationPrediction]
    shared_features: torch.Tensor | None = None
