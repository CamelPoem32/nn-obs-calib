"""Orchestration for synthetic calibration augmentation stages."""

from __future__ import annotations

import torch

from obscalib.augmentations.calibration_event import CalibrationEventSampler
from obscalib.augmentations.config import AugmentationConfig
from obscalib.augmentations.frame_randomization import SensorFrameRandomizer
from obscalib.augmentations.noise import MeasurementNoiseAugmenter
from obscalib.augmentations.prior_perturbation import CalibrationPriorPerturber
from obscalib.augmentations.rendering import CalibrationEventRenderer
from obscalib.augmentations.structures import AugmentationRecord, AugmentationResult, CalibrationTrajectory
from obscalib.data.structures import WindowBatch
from obscalib.augmentations.targets import build_augmented_targets


class AugmentationPipeline:
    """
    Compose synthetic sensor-calibration augmentation stages.

    The augmentation stages have distinct scientific roles:

        1. randomize physical sensor coordinate frames,
        2. perturb the calibration prior supplied to the model,
        3. sample true within-window calibration-change events,
        4. construct the corresponding true calibration trajectories,
        5. render those trajectories into raw measurements and timestamps,
        6. add measurement noise.

    Frame randomization changes the arbitrary coordinate representation of a
    physical sensor while preserving the represented world-frame measurements.

    Prior perturbation changes only the calibration supplied to the model. It
    does not represent a physical sensor change and therefore does not modify
    raw measurements.

    Calibration events represent true physical changes and are consequently
    rendered into the raw sensor streams before measurement noise is added.
    """

    def __init__(self, config: AugmentationConfig) -> None:
        self.config = config

        self.frame_randomizer = SensorFrameRandomizer(
            config.frame_randomization,
        )

        self.prior_perturber = CalibrationPriorPerturber(
            config.prior_perturbation,
        )

        self.event_sampler = CalibrationEventSampler(
            config.calibration_event,
        )

        self.event_renderer = CalibrationEventRenderer()

        self.noise_augmenter = MeasurementNoiseAugmenter(
            config.noise,
        )

    def __call__(
        self,
        window: WindowBatch,
        generator: torch.Generator | None = None,
    ) -> AugmentationResult:
        """Apply configured augmentation stages and return data, truth, and sampled metadata."""

        _validate_generator(generator)
        _validate_window(window)

        # Work with new containers throughout augmentation and discard source
        # targets because they may become invalid after synthetic calibration
        # changes. New supervision should later be generated from
        # calibration_truth.
        working_window = _copy_window_without_targets(window)

        # Randomize the arbitrary coordinate convention of every physical
        # sensor. Raw measurements and calibration states are transformed
        # consistently, so the represented physical observations remain the
        # same.
        working_window, true_pre_event_calibration, frame_randomization_by_key = self.frame_randomizer(
            working_window,
            generator=generator,
        )

        # Perturb the calibration state that will be supplied to the model.
        # This is independent of the true physical event sampled below.
        current_prior, prior_perturbation_xi_by_key, prior_perturbation_tau_by_key = self.prior_perturber(
            true_pre_event_calibration,
            generator=generator,
        )

        # Sample true physical calibration events from the true pre-event
        # calibration, never from the deliberately imperfect model prior.
        events_by_key = self.event_sampler(
            working_window,
            true_pre_event_calibration,
            generator=generator,
        )

        # Each trajectory combines the true pre-event calibration with its
        # sampled event. It is the authoritative description of calibration
        # truth throughout the complete window.
        calibration_truth = {
            calibration_key: CalibrationTrajectory(
                pre_event=state,
                event=events_by_key[calibration_key],
            )
            for calibration_key, state in true_pre_event_calibration.items()
        }

        for trajectory in calibration_truth.values():
            trajectory.validate()

        # Render the true time-varying calibration into the raw measurements
        # and measured timestamps. No-event trajectories naturally reduce to
        # the identity rendering.
        working_window = self.event_renderer(
            working_window,
            calibration_truth,
        )

        # Construct fresh supervision from the synthetic physical truth. These targets
        # replace the source-window targets invalidated at the beginning of augmentation.
        augmented_targets = build_augmented_targets(
            current_calibration=current_prior,
            calibration_truth=calibration_truth,
        )

        # The rendered data follows the true physical trajectory, while the model
        # receives the independently perturbed current calibration prior and targets
        # describing the corresponding true final state.
        working_window = WindowBatch(
            streams=dict(working_window.streams),
            current_calibration=current_prior,
            metadata=dict(working_window.metadata),
            targets=augmented_targets,
        )

        # Add measurement noise only after all deterministic calibration geometry and
        # temporal effects have been rendered. Noise does not alter calibration truth,
        # so the freshly constructed targets remain valid.
        augmented_window, noise_bias_by_stream = self.noise_augmenter(
            working_window,
            generator=generator,
        )

        record = AugmentationRecord(
            calibration_keys=tuple(true_pre_event_calibration),
            frame_randomization_by_key=frame_randomization_by_key,
            prior_perturbation_xi_by_key=prior_perturbation_xi_by_key,
            prior_perturbation_tau_by_key=prior_perturbation_tau_by_key,
            events_by_key=events_by_key,
            noise_bias_by_stream=noise_bias_by_stream,
        )

        result = AugmentationResult(
            augmented_window=augmented_window,
            calibration_truth=calibration_truth,
            record=record,
        )

        result.validate()

        return result


def _copy_window_without_targets(
    window: WindowBatch,
) -> WindowBatch:
    """
    Copy a raw window for augmentation and discard existing supervision.

    Synthetic calibration augmentation can invalidate targets attached to the
    original window. New targets should instead be constructed later from the
    returned calibration_truth.
    """

    return WindowBatch(
        streams=dict(window.streams),
        current_calibration=dict(window.current_calibration),
        metadata=dict(window.metadata),
        targets=None,
    )


def _validate_window(window: WindowBatch) -> None:
    """Validate the common raw-window contract required by augmentation."""

    if not window.streams:
        raise ValueError(
            "Augmentation requires at least one sensor stream."
        )

    if set(window.streams) != set(window.metadata):
        raise ValueError(
            "Window streams and metadata must have identical keys."
        )

    if not window.current_calibration:
        raise ValueError(
            "Augmentation requires at least one calibration state."
        )

    calibration_batch_sizes: set[int] = set()

    for calibration_key, state in window.current_calibration.items():
        if not calibration_key:
            raise ValueError(
                "Calibration keys must be non-empty."
            )

        try:
            state.validate()
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid calibration state {calibration_key!r}."
            ) from error

        if not torch.is_floating_point(state.transform):
            raise TypeError(
                f"Calibration transform {calibration_key!r} must have floating-point dtype."
            )

        if not torch.is_floating_point(state.time_offset):
            raise TypeError(
                f"Calibration time offset {calibration_key!r} must have floating-point dtype."
            )

        if state.transform.device != state.time_offset.device:
            raise ValueError(
                f"Calibration transform and time offset {calibration_key!r} must be on the same device."
            )

        if state.transform.dtype != state.time_offset.dtype:
            raise TypeError(
                f"Calibration transform and time offset {calibration_key!r} must have the same dtype."
            )

        calibration_batch_sizes.add(
            state.transform.shape[0]
        )

    if len(calibration_batch_sizes) != 1:
        raise ValueError(
            "All calibration states must share batch size."
        )

    batch_size = next(
        iter(calibration_batch_sizes)
    )

    for stream_key, stream in window.streams.items():
        stream.validate()

        metadata = window.metadata[stream_key]
        calibration_key = metadata.calibration_key

        if calibration_key not in window.current_calibration:
            raise KeyError(
                f"Stream {stream_key!r} requires missing calibration key {calibration_key!r}."
            )

        calibration_state = window.current_calibration[
            calibration_key
        ]

        if stream.values.shape[0] != batch_size:
            raise ValueError(
                f"Stream {stream_key!r} and calibration states must share batch size."
            )

        if stream.values.device != calibration_state.transform.device:
            raise ValueError(
                f"Stream {stream_key!r} and calibration {calibration_key!r} must be on the same device."
            )

        if stream.timestamps.device != calibration_state.transform.device:
            raise ValueError(
                f"Stream {stream_key!r} timestamps and calibration {calibration_key!r} must be on the same device."
            )

        if stream.values.dtype != calibration_state.transform.dtype:
            raise TypeError(
                f"Stream {stream_key!r} values and calibration {calibration_key!r} must have the same dtype."
            )

        if stream.timestamps.dtype != calibration_state.transform.dtype:
            raise TypeError(
                f"Stream {stream_key!r} timestamps and calibration {calibration_key!r} must have the same dtype."
            )


def _validate_generator(
    generator: torch.Generator | None,
) -> None:
    """Validate an optional reproducible random-number generator."""

    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError(
            "generator must be a torch.Generator or None."
        )