"""Transition profiles for synthetic calibration-change events."""

from enum import Enum

import torch


class TransitionProfile(str, Enum):
    """Supported calibration transition profiles."""

    STEP = "step"
    LINEAR = "linear"
    SMOOTHSTEP = "smoothstep"


def evaluate_transition_profiles(
    timestamps_s: torch.Tensor,
    change_time_s: torch.Tensor,
    transition_duration_s: torch.Tensor,
    profiles: tuple[TransitionProfile | None, ...],
) -> torch.Tensor:
    """Evaluate batched normalized calibration-transition progress.

    Args:
        timestamps_s:
            Window-relative measurement timestamps with shape [B, N].

        change_time_s:
            Event time relative to the window start with shape [B, 1].

            For STEP, this is the jump time.

            For LINEAR and SMOOTHSTEP, this is the center of the transition.

        transition_duration_s:
            Full transition duration with shape [B, 1].

            STEP requires zero duration.
            LINEAR and SMOOTHSTEP require positive duration.

        profiles:
            One transition profile per batch item. None represents a sample with
            no calibration-change event.

    Returns:
        Normalized transition progress with shape [B, N] and values in [0, 1].

        A value of 0 means the calibration is still at its pre-event state.
        A value of 1 means the calibration has reached its final state.
    """

    _validate_transition_inputs(timestamps_s, change_time_s, transition_duration_s, profiles)

    batch_size, _ = timestamps_s.shape
    progress = torch.zeros_like(timestamps_s)

    # Evaluate each batch item according to its discrete transition profile.
    for batch_index in range(batch_size):
        profile = profiles[batch_index]

        if profile is None:
            continue

        timestamps = timestamps_s[batch_index]
        change_time = change_time_s[batch_index, 0]
        transition_duration = transition_duration_s[batch_index, 0]

        if profile == TransitionProfile.STEP:
            progress[batch_index] = (timestamps >= change_time).to(dtype=timestamps_s.dtype)
            continue

        transition_start = change_time - 0.5 * transition_duration
        normalized_time = ((timestamps - transition_start) / transition_duration).clamp(0.0, 1.0)

        if profile == TransitionProfile.LINEAR:
            progress[batch_index] = normalized_time
        elif profile == TransitionProfile.SMOOTHSTEP:
            progress[batch_index] = normalized_time * normalized_time * (3.0 - 2.0 * normalized_time)
        else:
            raise ValueError(f"Unsupported transition profile: {profile!r}")

    return progress


def _validate_transition_inputs(
    timestamps_s: torch.Tensor,
    change_time_s: torch.Tensor,
    transition_duration_s: torch.Tensor,
    profiles: tuple[TransitionProfile | None, ...],
) -> None:
    """Validate batched transition-profile inputs."""

    if timestamps_s.ndim != 2:
        raise ValueError("timestamps_s must have shape [B, N].")
    if not torch.is_floating_point(timestamps_s):
        raise TypeError("timestamps_s must have floating-point dtype.")
    if not torch.all(torch.isfinite(timestamps_s)):
        raise ValueError("timestamps_s must contain only finite values.")

    batch_size = timestamps_s.shape[0]

    _validate_parameter_tensor("change_time_s", change_time_s, batch_size, timestamps_s)
    _validate_parameter_tensor("transition_duration_s", transition_duration_s, batch_size, timestamps_s)

    if len(profiles) != batch_size:
        raise ValueError("profiles must contain one entry per batch item.")

    if torch.any(change_time_s < 0.0):
        raise ValueError("change_time_s must be nonnegative because it is relative to window start.")

    if torch.any(transition_duration_s < 0.0):
        raise ValueError("transition_duration_s must be nonnegative.")

    # Profile-specific duration constraints are checked per batch item because
    # STEP and finite-duration transitions deliberately use different semantics.
    for batch_index, profile in enumerate(profiles):
        if profile is not None and not isinstance(profile, TransitionProfile):
            raise TypeError("profiles entries must be TransitionProfile values or None.")

        transition_duration = transition_duration_s[batch_index, 0]

        if profile is None:
            if transition_duration != 0.0:
                raise ValueError("No-event batch items must have zero transition_duration_s.")
            continue

        if profile == TransitionProfile.STEP and transition_duration != 0.0:
            raise ValueError("STEP transitions must have zero transition_duration_s.")

        if profile in (TransitionProfile.LINEAR, TransitionProfile.SMOOTHSTEP) and transition_duration <= 0.0:
            raise ValueError("LINEAR and SMOOTHSTEP transitions must have positive transition_duration_s.")


def _validate_parameter_tensor(name: str, value: torch.Tensor, batch_size: int, timestamps_s: torch.Tensor) -> None:
    """Validate one batched scalar transition parameter."""

    if value.shape != (batch_size, 1):
        raise ValueError(f"{name} must have shape [B, 1].")

    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must have floating-point dtype.")

    if value.device != timestamps_s.device:
        raise ValueError(f"{name} and timestamps_s must be on the same device.")

    if value.dtype != timestamps_s.dtype:
        raise TypeError(f"{name} and timestamps_s must have the same dtype.")

    if not torch.all(torch.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values.")