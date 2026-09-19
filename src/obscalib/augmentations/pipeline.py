"""Orchestration for synthetic calibration augmentation stages."""

from __future__ import annotations

import torch

from obscalib.augmentations.calibration_event import CalibrationEventSampler
from obscalib.augmentations.config import AugmentationConfig
from obscalib.augmentations.frame_randomization import SensorFrameRandomizer
from obscalib.augmentations.noise import MeasurementNoiseAugmenter
from obscalib.augmentations.prior_perturbation import CalibrationPriorPerturber
from obscalib.augmentations.rendering import CalibrationEventRenderer
from obscalib.augmentations.sampling_rate import SamplingRateAugmenter
from obscalib.augmentations.structures import AugmentationRecord, AugmentationResult, CalibrationTrajectory
from obscalib.augmentations.targets import build_augmented_targets
from obscalib.data.structures import WindowBatch


class AugmentationPipeline:
    """Compose sampling, calibration, rendering, and measurement-noise augmentation stages."""

    def __init__(self, config: AugmentationConfig) -> None:
        self.config = config
        self.sampling_rate_augmenter = SamplingRateAugmenter(config.sampling_rate)
        self.frame_randomizer = SensorFrameRandomizer(config.frame_randomization)
        self.prior_perturber = CalibrationPriorPerturber(config.prior_perturbation)
        self.event_sampler = CalibrationEventSampler(config.calibration_event)
        self.event_renderer = CalibrationEventRenderer()
        self.noise_augmenter = MeasurementNoiseAugmenter(config.noise)

    def __call__(self, window: WindowBatch, generator: torch.Generator | None = None) -> AugmentationResult:
        """Apply configured augmentation stages and return augmented data, truth, and sampled metadata."""

        _validate_window(window)

        # Source supervision describes the original window and becomes stale as
        # soon as synthetic calibration physics is introduced.
        working_window = _copy_window_without_targets(window)

        # Sampling-rate augmentation operates on the clean raw measurements.
        # IMU streams are linearly resampled and LiDAR relative poses are
        # reduced by SE3 composition.
        working_window, sampling_target_frequency_hz, sampling_rate_applied = self.sampling_rate_augmenter(working_window, generator=generator)

        # Randomize the physical sensor frames before sampling calibration
        # priors or true calibration events.
        working_window, true_pre_event_calibration, frame_randomization = self.frame_randomizer(working_window, generator=generator)

        # The model prior is independent of the true physical calibration event.
        current_prior, prior_perturbation_xi, prior_perturbation_tau = self.prior_perturber(true_pre_event_calibration, generator=generator)

        # Sample true physical calibration changes using the already
        # sampling-rate-augmented observation window.
        events = self.event_sampler(working_window, true_pre_event_calibration, generator=generator)

        calibration_truth = {
            calibration_key: CalibrationTrajectory(
                pre_event=state,
                event=events[calibration_key],
            )
            for calibration_key, state in true_pre_event_calibration.items()
        }

        for trajectory in calibration_truth.values():
            trajectory.validate()

        # Render the true time-varying spatial and temporal calibration into the
        # clean raw measurements. This uses the true pre-event calibration, not
        # the independently perturbed model prior.
        rendered_window = self.event_renderer(working_window, calibration_truth)

        # Construct fresh supervision from the final true synthetic calibration.
        augmented_targets = build_augmented_targets(current_calibration=current_prior, calibration_truth=calibration_truth)

        working_window = WindowBatch(
            streams=dict(rendered_window.streams),
            current_calibration=current_prior,
            metadata=dict(rendered_window.metadata),
            targets=augmented_targets,
        )

        # Measurement noise is intentionally last. Frequency reduction therefore
        # does not smooth or otherwise alter the synthetic noise realization.
        augmented_window, noise_bias = self.noise_augmenter(working_window, generator=generator)

        record = AugmentationRecord(
            calibration_keys=tuple(true_pre_event_calibration),
            sampling_target_frequency_hz_by_stream=sampling_target_frequency_hz,
            sampling_rate_applied_by_stream=sampling_rate_applied,
            frame_randomization_by_key=frame_randomization,
            prior_perturbation_xi_by_key=prior_perturbation_xi,
            prior_perturbation_tau_by_key=prior_perturbation_tau,
            events_by_key=events,
            noise_bias_by_stream=noise_bias,
        )

        return AugmentationResult(
            augmented_window=augmented_window,
            calibration_truth=calibration_truth,
            record=record,
        )


def _copy_window_without_targets(window: WindowBatch) -> WindowBatch:
    """Copy a raw window while invalidating source supervision before synthetic augmentation."""

    return WindowBatch(
        streams=dict(window.streams),
        current_calibration=dict(window.current_calibration),
        metadata=dict(window.metadata),
        targets=None,
    )


def _validate_window(window: WindowBatch) -> None:
    if not window.streams:
        raise ValueError("Augmentation requires at least one sensor stream.")

    if set(window.streams) != set(window.metadata):
        raise ValueError("Window streams and metadata must have identical keys.")

    if not window.current_calibration:
        raise ValueError("Augmentation requires at least one calibration state.")

    for calibration_key, state in window.current_calibration.items():
        try:
            state.validate()
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid calibration state {calibration_key!r}.") from error

    for stream_key, stream in window.streams.items():
        stream.validate()

        calibration_key = window.metadata[stream_key].calibration_key

        if calibration_key not in window.current_calibration:
            raise KeyError(f"Stream {stream_key!r} requires missing calibration key {calibration_key!r}.")

        if stream.values.shape[0] != window.current_calibration[calibration_key].transform.shape[0]:
            raise ValueError(f"Stream {stream_key!r} and calibration {calibration_key!r} must share batch size.")