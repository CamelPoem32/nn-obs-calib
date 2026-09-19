"""Data contracts exchanged by augmentation components."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from obscalib.augmentations.profiles import TransitionProfile
from obscalib.calibration.state import CalibrationState
from obscalib.data.structures import WindowBatch
from obscalib.geometry.lie import se3_exp


@dataclass
class CalibrationEvent:
    """One batched true calibration-change event description.

    `change_time_s` is measured relative to the start of the window.

    For STEP, change_time_s is the jump time and transition_duration_s is zero.

    For LINEAR and SMOOTHSTEP, change_time_s is the center of the transition and
    transition_duration_s is its full duration.

    delta_xi follows the repository convention [phi, rho].
    """

    occurred: torch.Tensor
    profiles: tuple[TransitionProfile | None, ...]

    change_time_s: torch.Tensor
    transition_duration_s: torch.Tensor

    delta_xi: torch.Tensor
    delta_tau: torch.Tensor

    def validate(self) -> None:
        """Validate batched event metadata and dense event tensors."""

        if self.occurred.ndim != 2 or self.occurred.shape[1] != 1:
            raise ValueError("occurred must have shape [B, 1].")
        if self.occurred.dtype != torch.bool:
            raise TypeError("occurred must have boolean dtype.")

        batch_size = self.occurred.shape[0]

        if len(self.profiles) != batch_size:
            raise ValueError("profiles must contain one entry per batch item.")

        _validate_event_tensor("change_time_s", self.change_time_s, batch_size, 1, self.occurred.device)
        _validate_event_tensor("transition_duration_s", self.transition_duration_s, batch_size, 1, self.occurred.device)
        _validate_event_tensor("delta_xi", self.delta_xi, batch_size, 6, self.occurred.device)
        _validate_event_tensor("delta_tau", self.delta_tau, batch_size, 1, self.occurred.device)

        if torch.any(self.change_time_s < 0.0):
            raise ValueError("change_time_s must be nonnegative because it is relative to window start.")

        if torch.any(self.transition_duration_s < 0.0):
            raise ValueError("transition_duration_s must be nonnegative.")

        for batch_index, profile in enumerate(self.profiles):
            occurred = bool(self.occurred[batch_index, 0].item())

            if profile is not None and not isinstance(profile, TransitionProfile):
                raise TypeError("profiles entries must be TransitionProfile values or None.")

            if occurred and profile is None:
                raise ValueError("Every occurred event must have a transition profile.")

            if not occurred and profile is not None:
                raise ValueError("No-event batch items must not have a transition profile.")

            if not occurred:
                if torch.any(self.delta_xi[batch_index] != 0.0):
                    raise ValueError("No-event batch items must have zero delta_xi.")
                if torch.any(self.delta_tau[batch_index] != 0.0):
                    raise ValueError("No-event batch items must have zero delta_tau.")
                if torch.any(self.change_time_s[batch_index] != 0.0):
                    raise ValueError("No-event batch items must have zero change_time_s.")
                if torch.any(self.transition_duration_s[batch_index] != 0.0):
                    raise ValueError("No-event batch items must have zero transition_duration_s.")
                continue

            duration_s = self.transition_duration_s[batch_index, 0]

            if profile == TransitionProfile.STEP and duration_s != 0.0:
                raise ValueError("STEP events must have zero transition_duration_s.")

            if profile in (TransitionProfile.LINEAR, TransitionProfile.SMOOTHSTEP) and duration_s <= 0.0:
                raise ValueError("LINEAR and SMOOTHSTEP events must have positive transition_duration_s.")


@dataclass
class CalibrationTrajectory:
    """True calibration before an event plus the sampled true event.

    The final state is derived from this information rather than stored
    independently, so the event and final calibration cannot become inconsistent.
    """

    pre_event: CalibrationState
    event: CalibrationEvent

    def validate(self) -> None:
        self.pre_event.validate()
        self.event.validate()

        batch_size = self.pre_event.transform.shape[0]
        if self.event.occurred.shape[0] != batch_size:
            raise ValueError("Calibration state and event metadata must share batch size.")

        if self.event.delta_xi.device != self.pre_event.transform.device:
            raise ValueError("delta_xi and pre-event transform must be on the same device.")

        if self.event.delta_tau.device != self.pre_event.time_offset.device:
            raise ValueError("delta_tau and pre-event time offset must be on the same device.")

    def final_state(self) -> CalibrationState:
        """Return the true post-event calibration state."""

        self.validate()

        delta_transform = se3_exp(self.event.delta_xi)

        return CalibrationState(
            transform=delta_transform @ self.pre_event.transform,
            time_offset=self.pre_event.time_offset + self.event.delta_tau,
        )


@dataclass
class AugmentationRecord:
    """Sampled augmentation metadata kept separate by scientific purpose.

    `frame_randomization_by_key` stores the sampled sensor-frame transformation
    A as an SE3 matrix [B, 4, 4], using the convention:

        T_WS_randomized = T_WS @ A

    In the current version A contains an arbitrary rotation and zero translation.

    The record is primarily intended for inspection and debugging. Exact random
    measurement-noise realizations do not need to be stored here.
    """

    calibration_keys: tuple[str, ...]

    frame_randomization_by_key: dict[str, torch.Tensor] = field(default_factory=dict)

    prior_perturbation_xi_by_key: dict[str, torch.Tensor] = field(default_factory=dict)
    prior_perturbation_tau_by_key: dict[str, torch.Tensor] = field(default_factory=dict)

    events_by_key: dict[str, CalibrationEvent] = field(default_factory=dict)

    noise_bias_by_stream: dict[str, torch.Tensor] = field(default_factory=dict)

    def validate(self) -> None:
        if len(self.calibration_keys) != len(set(self.calibration_keys)):
            raise ValueError("calibration_keys must be unique.")

        if any(not calibration_key for calibration_key in self.calibration_keys):
            raise ValueError("calibration_keys must be non-empty strings.")

        calibration_keys = set(self.calibration_keys)

        _validate_record_keys("frame_randomization_by_key", self.frame_randomization_by_key, calibration_keys)
        _validate_record_keys("prior_perturbation_xi_by_key", self.prior_perturbation_xi_by_key, calibration_keys)
        _validate_record_keys("prior_perturbation_tau_by_key", self.prior_perturbation_tau_by_key, calibration_keys)

        if set(self.events_by_key) != calibration_keys:
            raise ValueError("events_by_key must contain exactly one event record for every calibration key.")

        for calibration_key, frame_randomization in self.frame_randomization_by_key.items():
            if frame_randomization.ndim != 3 or frame_randomization.shape[-2:] != (4, 4):
                raise ValueError(f"frame_randomization_by_key[{calibration_key!r}] must have shape [B, 4, 4].")
            if not torch.is_floating_point(frame_randomization):
                raise TypeError(f"frame_randomization_by_key[{calibration_key!r}] must have floating-point dtype.")
            if not torch.all(torch.isfinite(frame_randomization)):
                raise ValueError(f"frame_randomization_by_key[{calibration_key!r}] must contain finite values.")

        for calibration_key, delta_xi in self.prior_perturbation_xi_by_key.items():
            if delta_xi.ndim != 2 or delta_xi.shape[1] != 6:
                raise ValueError(f"prior_perturbation_xi_by_key[{calibration_key!r}] must have shape [B, 6].")

        for calibration_key, delta_tau in self.prior_perturbation_tau_by_key.items():
            if delta_tau.ndim != 2 or delta_tau.shape[1] != 1:
                raise ValueError(f"prior_perturbation_tau_by_key[{calibration_key!r}] must have shape [B, 1].")

        for event in self.events_by_key.values():
            event.validate()


@dataclass
class AugmentationResult:
    """Augmented window, synthetic calibration truth, and augmentation metadata."""

    augmented_window: WindowBatch
    calibration_truth: dict[str, CalibrationTrajectory]
    record: AugmentationRecord

    def validate(self) -> None:
        if set(self.calibration_truth) != set(self.record.calibration_keys):
            raise ValueError("calibration_truth and record.calibration_keys must contain the same calibration keys.")

        for trajectory in self.calibration_truth.values():
            trajectory.validate()

        self.record.validate()


def _validate_event_tensor(name: str, value: torch.Tensor, batch_size: int, width: int, device: torch.device) -> None:
    if value.shape != (batch_size, width):
        raise ValueError(f"{name} must have shape [B, {width}].")

    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must have floating-point dtype.")

    if value.device != device:
        raise ValueError(f"{name} and occurred must be on the same device.")

    if not torch.all(torch.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values.")


def _validate_record_keys(name: str, values: dict[str, torch.Tensor], calibration_keys: set[str]) -> None:
    unexpected_keys = set(values) - calibration_keys
    if unexpected_keys:
        raise ValueError(f"{name} contains unknown calibration keys: {sorted(unexpected_keys)}")