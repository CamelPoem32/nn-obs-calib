"""Explicit tensor contracts exchanged by calibration pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CalibrationPrediction:
    """Deterministic output for one calibration head.

    # change_logit: Auxiliary supervised change-event score. It does not gate delta_xi or
    # delta_tau; sigmoid(change_logit) may be thresholded only when a
    # binary change/no-change metric or diagnostic decision is required.

    delta_xi uses shape [B, 6]. Before state updates are implemented, the
    project must fix [rho, phi] versus [phi, rho] ordering, whether rho is
    direct translation or the translational component of an se(3)
    perturbation, left versus right perturbation, local/body versus
    global/world frame interpretation, and Exp(delta_xi) @ T versus
    T @ Exp(delta_xi).

    delta_tau is intended as a future additive time-offset correction.
    Covariance/uncertainty prediction remains a possible later extension and
    is intentionally absent from this deterministic first-stage output.
    """

    change_logit: torch.Tensor
    change_time: torch.Tensor
    delta_xi: torch.Tensor
    delta_tau: torch.Tensor


@dataclass
class ModelOutput:
    """Named calibration predictions plus an optional analysis shared_features."""

    predictions: dict[str, CalibrationPrediction]
    shared_features: torch.Tensor | None = None
