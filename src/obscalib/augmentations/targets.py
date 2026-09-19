"""Construction of training targets for synthetically augmented windows."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from obscalib.augmentations.structures import CalibrationTrajectory
from obscalib.calibration.state import CalibrationState
from obscalib.data.structures import CalibrationTargetBatch


def build_augmented_targets(
    current_calibration: Mapping[str, CalibrationState],
    calibration_truth: Mapping[str, CalibrationTrajectory],
) -> dict[str, CalibrationTargetBatch]:
    """
    Construct model supervision for a synthetically augmented window.

    Args:
        current_calibration:
            Calibration state supplied to the model for the current window.

            This may intentionally differ from the true pre-event calibration
            because prior perturbation can make the model start from an
            imperfect spatial or temporal calibration.

        calibration_truth:
            True synthetic calibration trajectory for every calibration key.

    Returns:
        One CalibrationTargetBatch per calibration key.

        Each target contains:

            next_transform:
                True calibration transform at the end of the window.

            next_time_offset:
                True temporal calibration at the end of the window.

            change_label:
                Binary indicator of whether a true calibration event occurred.

            change_time:
                True event time relative to the beginning of the window.

        The target stores the true final state rather than a precomputed
        delta_xi or delta_tau. The correction required from the model can
        therefore be derived later relative to the actual current calibration:

            delta_xi_target
                = Log(
                    T_true_final
                    @ inverse(T_current)
                )

            delta_tau_target
                = tau_true_final
                  - tau_current.

        This is important because SE3 corrections are not generally obtained by
        subtracting tangent vectors.
    """

    _validate_inputs(
        current_calibration=current_calibration,
        calibration_truth=calibration_truth,
    )

    targets: dict[str, CalibrationTargetBatch] = {}

    for calibration_key, trajectory in calibration_truth.items():
        current_state = current_calibration[calibration_key]
        final_state = trajectory.final_state()
        event = trajectory.event

        # BCE-style binary supervision is represented using the same floating
        # dtype as the calibration tensors rather than a boolean tensor.
        change_label = event.occurred.to(
            dtype=current_state.transform.dtype,
        )

        target = CalibrationTargetBatch(
            next_transform=final_state.transform,
            next_time_offset=final_state.time_offset,
            change_label=change_label,
            change_time=event.change_time_s,
        )

        target.validate()

        targets[calibration_key] = target

    return targets


def _validate_inputs(
    current_calibration: Mapping[str, CalibrationState],
    calibration_truth: Mapping[str, CalibrationTrajectory],
) -> None:
    """Validate model-input calibration and synthetic truth compatibility."""

    if not current_calibration:
        raise ValueError(
            "Augmented target construction requires current calibration."
        )

    if not calibration_truth:
        raise ValueError(
            "Augmented target construction requires calibration truth."
        )

    if set(current_calibration) != set(calibration_truth):
        missing_truth = (
            set(current_calibration)
            - set(calibration_truth)
        )

        missing_current = (
            set(calibration_truth)
            - set(current_calibration)
        )

        raise ValueError(
            "current_calibration and calibration_truth must contain identical "
            "calibration keys. "
            f"Missing truth: {sorted(missing_truth)}; "
            f"missing current calibration: {sorted(missing_current)}."
        )

    for calibration_key, current_state in current_calibration.items():
        current_state.validate()

        trajectory = calibration_truth[calibration_key]
        trajectory.validate()

        final_state = trajectory.final_state()

        batch_size = current_state.transform.shape[0]

        if trajectory.pre_event.transform.shape[0] != batch_size:
            raise ValueError(
                f"Current calibration and pre-event truth {calibration_key!r} "
                "must share batch size."
            )

        if final_state.transform.shape[0] != batch_size:
            raise ValueError(
                f"Current calibration and final truth {calibration_key!r} "
                "must share batch size."
            )

        if current_state.transform.device != final_state.transform.device:
            raise ValueError(
                f"Current calibration and final truth {calibration_key!r} "
                "must be on the same device."
            )

        if current_state.transform.dtype != final_state.transform.dtype:
            raise TypeError(
                f"Current calibration and final truth {calibration_key!r} "
                "must have the same dtype."
            )

        event = trajectory.event

        if event.change_time_s.device != current_state.transform.device:
            raise ValueError(
                f"Event timing and current calibration {calibration_key!r} "
                "must be on the same device."
            )

        if event.change_time_s.dtype != current_state.transform.dtype:
            raise TypeError(
                f"Event timing and current calibration {calibration_key!r} "
                "must have the same dtype."
            )