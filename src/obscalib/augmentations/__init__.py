"""Public API for synthetic sensor-calibration augmentation."""

from obscalib.augmentations.calibration_event import CalibrationEventSampler
from obscalib.augmentations.config import (
    AugmentationConfig,
    CalibrationEventConfig,
    FrameRandomizationConfig,
    NoiseAugmentationConfig,
    PerturbationDistribution,
    PerturbationMagnitudeConfig,
    PriorPerturbationConfig,
    SamplingRateAugmentationConfig,
)
from obscalib.augmentations.frame_randomization import SensorFrameRandomizer
from obscalib.augmentations.noise import MeasurementNoiseAugmenter
from obscalib.augmentations.pipeline import AugmentationPipeline
from obscalib.augmentations.prior_perturbation import CalibrationPriorPerturber
from obscalib.augmentations.profiles import TransitionProfile
from obscalib.augmentations.structures import (
    AugmentationRecord,
    AugmentationResult,
    CalibrationEvent,
    CalibrationTrajectory,
)
from obscalib.augmentations.sampling_rate import SamplingRateAugmenter

__all__ = [
    "AugmentationConfig",
    "AugmentationPipeline",
    "AugmentationRecord",
    "AugmentationResult",
    "CalibrationEvent",
    "CalibrationEventConfig",
    "CalibrationEventSampler",
    "CalibrationPriorPerturber",
    "CalibrationTrajectory",
    "FrameRandomizationConfig",
    "MeasurementNoiseAugmenter",
    "NoiseAugmentationConfig",
    "PerturbationDistribution",
    "PerturbationMagnitudeConfig",
    "PriorPerturbationConfig",
    "SensorFrameRandomizer",
    "TransitionProfile",
    "SamplingRateAugmentationConfig",
    "SamplingRateAugmenter",
]
