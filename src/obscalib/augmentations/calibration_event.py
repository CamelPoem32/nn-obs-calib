"""Sampling of true synthetic calibration-change events."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from obscalib.augmentations.config import CalibrationEventConfig
from obscalib.augmentations.profiles import TransitionProfile
from obscalib.augmentations.sampling import sample_isotropic_perturbation, sample_signed_scalar_perturbation
from obscalib.augmentations.structures import CalibrationEvent
from obscalib.calibration.state import CalibrationState
from obscalib.data.structures import WindowBatch


class CalibrationEventSampler:
    """Sample independent true calibration-change events for each calibration key."""

    def __init__(self, config: CalibrationEventConfig) -> None:
        self.config = config

    def __call__(
        self,
        window: WindowBatch,
        calibration: Mapping[str, CalibrationState],
        generator: torch.Generator | None = None,
    ) -> dict[str, CalibrationEvent]:
        """
        Sample calibration events using the temporal extent of the current window.

        Window timestamps are relative to the window start. Therefore
        change_time_s is also sampled relative to zero.

        The largest valid timestamp across all streams is used as the available
        window end for each batch item.
        """

        _validate_generator(generator)
        _validate_calibration(calibration)

        if not self.config.enabled:
            return {
                calibration_key: _make_no_event(state)
                for calibration_key, state in calibration.items()
            }

        batch_size = _get_common_batch_size(calibration)

        window_end_s = _compute_window_end_times(
            window=window,
            expected_batch_size=batch_size,
        )

        events: dict[str, CalibrationEvent] = {}

        for calibration_key, state in calibration.items():
            probability = self.config.event_probability_by_key.get(calibration_key, self.config.event_probability)

            # Convert window timing to the calibration tensor device/dtype so
            # all sampled event tensors follow the calibration-state contract.
            state_window_end_s = window_end_s.to(
                device=state.transform.device,
                dtype=state.transform.dtype,
            )

            event = self._sample_event(
                state=state,
                event_probability=probability,
                window_end_s=state_window_end_s,
                generator=generator,
            )

            event.validate()
            events[calibration_key] = event

        return events

    def _sample_event(
        self,
        state: CalibrationState,
        event_probability: float,
        window_end_s: torch.Tensor,
        generator: torch.Generator | None,
    ) -> CalibrationEvent:
        """Sample one batched event for one physical calibration key."""

        batch_size = state.transform.shape[0]
        device = state.transform.device
        dtype = state.transform.dtype

        occurred = torch.rand(
            batch_size,
            1,
            device=device,
            dtype=dtype,
            generator=generator,
        ) < event_probability

        profiles = _sample_profiles(
            occurred=occurred,
            config=self.config,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        rotation_selected, translation_selected, time_offset_selected = _sample_event_components(
            occurred=occurred,
            config=self.config,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        delta_phi = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        delta_rho = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        delta_tau = torch.zeros(batch_size, 1, device=device, dtype=dtype)

        if self.config.rotation is not None:
            sampled_rotation = sample_isotropic_perturbation(
                config=self.config.rotation,
                batch_size=batch_size,
                dimension=3,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            delta_phi = sampled_rotation * rotation_selected.to(dtype=dtype)

        if self.config.translation is not None:
            sampled_translation = sample_isotropic_perturbation(
                config=self.config.translation,
                batch_size=batch_size,
                dimension=3,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            delta_rho = sampled_translation * translation_selected.to(dtype=dtype)

        if self.config.time_offset is not None:
            sampled_time_offset = sample_signed_scalar_perturbation(
                config=self.config.time_offset,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            delta_tau = sampled_time_offset * time_offset_selected.to(dtype=dtype)

        delta_xi = torch.cat((delta_phi, delta_rho), dim=-1)

        change_time_s, transition_duration_s = _sample_event_timing(
            occurred=occurred,
            profiles=profiles,
            window_end_s=window_end_s,
            config=self.config,
            generator=generator,
        )

        return CalibrationEvent(
            occurred=occurred,
            profiles=profiles,
            change_time_s=change_time_s,
            transition_duration_s=transition_duration_s,
            delta_xi=delta_xi,
            delta_tau=delta_tau,
        )


def _sample_profiles(
    occurred: torch.Tensor,
    config: CalibrationEventConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> tuple[TransitionProfile | None, ...]:
    """Sample one transition profile for every positive event."""

    batch_size = occurred.shape[0]
    profiles: list[TransitionProfile | None] = [None] * batch_size

    event_indices = torch.nonzero(
        occurred[:, 0],
        as_tuple=False,
    ).squeeze(-1)

    if event_indices.numel() == 0:
        return tuple(profiles)

    available_profiles = [
        profile
        for profile, weight in config.profile_weights.items()
        if weight > 0.0
    ]

    weights = torch.tensor(
        [
            config.profile_weights[profile]
            for profile in available_profiles
        ],
        device=device,
        dtype=dtype,
    )

    sampled_profile_indices = torch.multinomial(
        weights,
        num_samples=event_indices.numel(),
        replacement=True,
        generator=generator,
    )

    for event_index, profile_index in zip(event_indices.tolist(), sampled_profile_indices.tolist()):
        profiles[event_index] = available_profiles[profile_index]

    return tuple(profiles)


def _sample_event_components(
    occurred: torch.Tensor,
    config: CalibrationEventConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select rotation, translation, and temporal components independently.

    Conditional on an event occurring, the three Bernoulli decisions use

        rotation_probability
        translation_probability
        time_offset_probability.

    If all three decisions fail for one positive event, that item is resampled
    until at least one physical calibration component is selected.
    """

    batch_size = occurred.shape[0]

    selected = torch.zeros(
        batch_size,
        3,
        dtype=torch.bool,
        device=device,
    )

    probabilities = torch.tensor(
        [
            config.rotation_probability,
            config.translation_probability,
            config.time_offset_probability,
        ],
        device=device,
        dtype=dtype,
    )

    unresolved_indices = torch.nonzero(
        occurred[:, 0],
        as_tuple=False,
    ).squeeze(-1)

    while unresolved_indices.numel() > 0:
        draws = torch.rand(
            unresolved_indices.numel(),
            3,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        candidate_selection = draws < probabilities
        has_component = torch.any(candidate_selection, dim=-1)

        accepted_indices = unresolved_indices[has_component]
        selected[accepted_indices] = candidate_selection[has_component]

        unresolved_indices = unresolved_indices[~has_component]

    rotation_selected = selected[:, 0:1]
    translation_selected = selected[:, 1:2]
    time_offset_selected = selected[:, 2:3]

    return rotation_selected, translation_selected, time_offset_selected


def _sample_event_timing(
    occurred: torch.Tensor,
    profiles: tuple[TransitionProfile | None, ...],
    window_end_s: torch.Tensor,
    config: CalibrationEventConfig,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample transition duration and center/jump time.

    For STEP:

        change_time_s = jump time
        transition_duration_s = 0

    For LINEAR and SMOOTHSTEP:

        change_time_s = transition center
        transition_duration_s = full transition duration

    The complete transition is constrained to remain inside the window while
    preserving the configured minimum pre-event and post-event context.
    """

    batch_size = occurred.shape[0]
    device = window_end_s.device
    dtype = window_end_s.dtype

    change_time_s = torch.zeros(batch_size, 1, device=device, dtype=dtype)
    transition_duration_s = torch.zeros(batch_size, 1, device=device, dtype=dtype)

    for batch_index, profile in enumerate(profiles):
        if not bool(occurred[batch_index, 0].item()):
            continue

        if profile is None:
            raise ValueError("Occurred events must have a transition profile.")

        window_end = window_end_s[batch_index, 0]

        minimum_pre_context = config.minimum_pre_event_fraction * window_end
        minimum_post_context = config.minimum_post_event_fraction * window_end

        if profile == TransitionProfile.STEP:
            earliest_change_time = minimum_pre_context
            latest_change_time = window_end - minimum_post_context

            change_time_s[batch_index, 0] = _sample_uniform_interval(
                lower=earliest_change_time,
                upper=latest_change_time,
                generator=generator,
            )

            continue

        if profile not in (TransitionProfile.LINEAR, TransitionProfile.SMOOTHSTEP):
            raise ValueError(f"Unsupported transition profile: {profile!r}")

        if config.minimum_transition_duration_s is None or config.maximum_transition_duration_s is None:
            raise ValueError("Finite-duration transitions require configured minimum and maximum durations.")

        available_transition_duration = window_end - minimum_pre_context - minimum_post_context

        minimum_duration = torch.as_tensor(
            config.minimum_transition_duration_s,
            device=device,
            dtype=dtype,
        )

        configured_maximum_duration = torch.as_tensor(
            config.maximum_transition_duration_s,
            device=device,
            dtype=dtype,
        )

        maximum_duration = torch.minimum(
            configured_maximum_duration,
            available_transition_duration,
        )

        if bool((maximum_duration < minimum_duration).item()):
            raise ValueError(
                "Window is too short for the configured transition duration "
                "and minimum pre/post-event context."
            )

        duration = _sample_uniform_interval(
            lower=minimum_duration,
            upper=maximum_duration,
            generator=generator,
        )

        earliest_change_time = minimum_pre_context + 0.5 * duration
        latest_change_time = window_end - minimum_post_context - 0.5 * duration

        change_time = _sample_uniform_interval(
            lower=earliest_change_time,
            upper=latest_change_time,
            generator=generator,
        )

        transition_duration_s[batch_index, 0] = duration
        change_time_s[batch_index, 0] = change_time

    return change_time_s, transition_duration_s


def _sample_uniform_interval(
    lower: torch.Tensor,
    upper: torch.Tensor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Sample uniformly from the closed numerical interval [lower, upper]."""

    if bool((upper < lower).item()):
        raise ValueError("Uniform sampling interval must satisfy lower <= upper.")

    if bool((upper == lower).item()):
        return lower

    u = torch.rand(
        (),
        device=lower.device,
        dtype=lower.dtype,
        generator=generator,
    )

    return lower + u * (upper - lower)


def _compute_window_end_times(
    window: WindowBatch,
    expected_batch_size: int,
) -> torch.Tensor:
    """
    Return the largest valid relative timestamp in each batch item.

    WindowBatch currently stores window-relative stream timestamps but no
    explicit batched window-end field, so the latest valid sensor timestamp is
    used as the available temporal boundary.
    """

    if not window.streams:
        raise ValueError("Calibration-event sampling requires at least one sensor stream.")

    if set(window.streams) != set(window.metadata):
        raise ValueError("Window streams and metadata must have identical keys.")

    reference_stream = next(iter(window.streams.values()))
    reference_stream.validate()

    if reference_stream.timestamps.shape[0] != expected_batch_size:
        raise ValueError("Window streams and calibration states must share batch size.")

    device = reference_stream.timestamps.device
    dtype = reference_stream.timestamps.dtype

    window_end_s = torch.full(
        (expected_batch_size, 1),
        -torch.inf,
        device=device,
        dtype=dtype,
    )

    has_valid_sample = torch.zeros(
        expected_batch_size,
        1,
        dtype=torch.bool,
        device=device,
    )

    for stream_key, stream in window.streams.items():
        stream.validate()

        if stream.timestamps.shape[0] != expected_batch_size:
            raise ValueError(f"Stream {stream_key!r} does not share the calibration batch size.")

        if stream.timestamps.device != device:
            raise ValueError("All stream timestamps must be on the same device for event sampling.")

        if stream.timestamps.dtype != dtype:
            raise ValueError("All stream timestamps must have the same dtype for event sampling.")

        if torch.any(stream.sample_mask & (stream.timestamps < 0.0)):
            raise ValueError("Valid window timestamps must be nonnegative and relative to the window start.")

        masked_timestamps = torch.where(
            stream.sample_mask,
            stream.timestamps,
            torch.full_like(stream.timestamps, -torch.inf),
        )

        stream_end_s = torch.max(
            masked_timestamps,
            dim=1,
            keepdim=True,
        ).values

        stream_has_valid_sample = torch.any(
            stream.sample_mask,
            dim=1,
            keepdim=True,
        )

        window_end_s = torch.maximum(
            window_end_s,
            stream_end_s,
        )

        has_valid_sample = has_valid_sample | stream_has_valid_sample

    if not torch.all(has_valid_sample):
        raise ValueError("Every batch item must contain at least one valid sensor sample.")

    if torch.any(window_end_s <= 0.0):
        raise ValueError("Every batch item must have a positive temporal extent for calibration-event sampling.")

    return window_end_s


def _make_no_event(state: CalibrationState) -> CalibrationEvent:
    """Construct a dense zero-valued no-event record."""

    batch_size = state.transform.shape[0]
    device = state.transform.device
    dtype = state.transform.dtype

    event = CalibrationEvent(
        occurred=torch.zeros(
            batch_size,
            1,
            dtype=torch.bool,
            device=device,
        ),
        profiles=(None,) * batch_size,
        change_time_s=torch.zeros(
            batch_size,
            1,
            device=device,
            dtype=dtype,
        ),
        transition_duration_s=torch.zeros(
            batch_size,
            1,
            device=device,
            dtype=dtype,
        ),
        delta_xi=torch.zeros(
            batch_size,
            6,
            device=device,
            dtype=dtype,
        ),
        delta_tau=torch.zeros(
            batch_size,
            1,
            device=device,
            dtype=dtype,
        ),
    )

    event.validate()

    return event


def _get_common_batch_size(
    calibration: Mapping[str, CalibrationState],
) -> int:
    """Require every calibration key to belong to the same minibatch."""

    batch_sizes = {
        state.transform.shape[0]
        for state in calibration.values()
    }

    if len(batch_sizes) != 1:
        raise ValueError("All calibration states must share batch size.")

    return next(iter(batch_sizes))


def _validate_calibration(
    calibration: Mapping[str, CalibrationState],
) -> None:
    """Validate calibration states required by event sampling."""

    if not calibration:
        raise ValueError("At least one calibration state is required.")

    for calibration_key, state in calibration.items():
        if not calibration_key:
            raise ValueError("Calibration keys must be non-empty.")

        state.validate()

        if not torch.is_floating_point(state.transform):
            raise TypeError(f"Calibration transform {calibration_key!r} must have floating-point dtype.")

        if not torch.is_floating_point(state.time_offset):
            raise TypeError(f"Calibration time offset {calibration_key!r} must have floating-point dtype.")

        if state.transform.device != state.time_offset.device:
            raise ValueError(f"Calibration transform and time offset {calibration_key!r} must be on the same device.")

        if state.transform.dtype != state.time_offset.dtype:
            raise ValueError(f"Calibration transform and time offset {calibration_key!r} must have the same dtype.")


def _validate_generator(
    generator: torch.Generator | None,
) -> None:
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or None.")