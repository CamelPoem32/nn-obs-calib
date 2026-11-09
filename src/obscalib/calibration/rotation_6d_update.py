"""Apply direct full-range SO(3) calibration corrections without an SE(3) Log."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import torch

from obscalib.calibration.state import CalibrationState
from obscalib.models.calibration_6d_head import Calibration6DPrediction


def apply_calibration_6d_update(states: Mapping[str, CalibrationState], predictions: Mapping[str, Calibration6DPrediction], *, keys_to_update: Sequence[str] | None = None) -> dict[str, CalibrationState]:
    """Apply left SO(3) correction, parent-frame additive translation and time shift.

    The rotation is ``R_delta @ R_prior``; translation is ``t_prior + delta_t``.
    This intentionally differs from applying ``Exp([phi, rho]) @ T_prior``, which
    couples translation to rotation. Healthy keys can be held fixed with
    ``keys_to_update`` for an isolated-sensor ablation.
    """
    if set(predictions) != set(states):
        raise ValueError(f"State/prediction key mismatch: {set(states) ^ set(predictions)}")
    selected = set(states) if keys_to_update is None else set(keys_to_update)
    if not selected.issubset(states):
        raise KeyError(f"Unknown calibration keys: {sorted(selected - states.keys())}")

    result = {}
    for key, state in states.items():
        if key not in selected:
            result[key] = state
            continue

        prediction = predictions[key]
        prior = state.transform
        if prior.ndim != 3 or prior.shape[-2:] != (4, 4):
            raise ValueError(f"state {key!r} transform must be [B, 4, 4]")
        batch_size = prior.shape[0]
        if prediction.rotation_6d.shape != (batch_size, 6) or prediction.delta_translation.shape != (batch_size, 3) or prediction.delta_tau.shape != (batch_size, 1):
            raise ValueError(f"Prediction shapes for {key!r} must be [B,6], [B,3], [B,1]")
        if state.time_offset.shape != (batch_size, 1):
            raise ValueError(f"state {key!r} time_offset must be [B,1]")

        R_new = prediction.delta_rotation @ prior[:, :3, :3]
        t_new = prior[:, :3, 3] + prediction.delta_translation
        top = torch.cat((R_new, t_new.unsqueeze(-1)), dim=-1)
        transform = torch.cat((top, prior[:, 3:4, :]), dim=-2)
        result[key] = replace(state, transform=transform, time_offset=state.time_offset + prediction.delta_tau)
    return result


__all__ = ["apply_calibration_6d_update"]
