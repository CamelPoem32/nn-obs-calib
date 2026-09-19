"""End-to-end tests for the synthetic calibration augmentation pipeline."""

from __future__ import annotations

import torch

from obscalib.augmentations.config import AugmentationConfig, CalibrationEventConfig, FrameRandomizationConfig, NoiseAugmentationConfig, PerturbationMagnitudeConfig, PriorPerturbationConfig, VectorNoiseConfig
from obscalib.augmentations.pipeline import AugmentationPipeline
from obscalib.augmentations.profiles import TransitionProfile
from obscalib.calibration.state import CalibrationState
from obscalib.data.structures import GeometryType, MeasurementType, SensorMetadata, SensorStreamBatch, WindowBatch
from obscalib.geometry.lie import se3_exp


DTYPE = torch.float64
ATOL = 1e-9
RTOL = 1e-7


def _identity_state(
    batch_size: int = 1,
    time_offset: float = 0.0,
) -> CalibrationState:
    """Construct an identity calibration state."""

    return CalibrationState(
        transform=torch.eye(
            4,
            dtype=DTYPE,
        ).unsqueeze(0).repeat(
            batch_size,
            1,
            1,
        ),
        time_offset=torch.full(
            (batch_size, 1),
            time_offset,
            dtype=DTYPE,
        ),
    )


def _make_two_stream_imu_window() -> WindowBatch:
    """Construct gyro and accelerometer streams sharing one physical IMU key."""

    timestamps = torch.tensor(
        [[0.0, 2.0, 4.0, 6.0, 8.0, 10.0]],
        dtype=DTYPE,
    )

    gyro_values = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.8, 0.2, 0.1],
                [0.7, 0.3, 0.1],
                [0.6, 0.4, 0.2],
                [0.5, 0.5, 0.2],
            ]
        ],
        dtype=DTYPE,
    )

    accel_values = torch.tensor(
        [
            [
                [0.0, 0.0, -9.81],
                [0.1, 0.0, -9.80],
                [0.1, 0.1, -9.79],
                [0.2, 0.1, -9.78],
                [0.2, 0.2, -9.77],
                [0.3, 0.2, -9.76],
            ]
        ],
        dtype=DTYPE,
    )

    sample_mask = torch.ones(
        1,
        timestamps.shape[1],
        dtype=torch.bool,
    )

    return WindowBatch(
        streams={
            "gyro": SensorStreamBatch(
                values=gyro_values,
                timestamps=timestamps.clone(),
                sample_mask=sample_mask.clone(),
            ),
            "accel": SensorStreamBatch(
                values=accel_values,
                timestamps=timestamps.clone(),
                sample_mask=sample_mask.clone(),
            ),
        },
        current_calibration={
            "imu": _identity_state(
                time_offset=0.03,
            ),
        },
        metadata={
            "gyro": SensorMetadata(
                measurement_type=MeasurementType.IMU_GYROSCOPE,
                geometry_type=GeometryType.VECTOR,
                calibration_key="imu",
            ),
            "accel": SensorMetadata(
                measurement_type=MeasurementType.IMU_ACCELEROMETER,
                geometry_type=GeometryType.VECTOR,
                calibration_key="imu",
            ),
        },
        targets={},
    )


def test_pipeline_with_all_augmentations_disabled_is_identity() -> None:
    """The disabled augmentation pipeline must preserve raw data and calibration."""

    window = _make_two_stream_imu_window()

    pipeline = AugmentationPipeline(
        AugmentationConfig()
    )

    result = pipeline(
        window,
        generator=torch.Generator().manual_seed(1),
    )

    for stream_key in window.streams:
        torch.testing.assert_close(
            result.augmented_window.streams[stream_key].values,
            window.streams[stream_key].values,
        )

        torch.testing.assert_close(
            result.augmented_window.streams[stream_key].timestamps,
            window.streams[stream_key].timestamps,
        )

        assert torch.equal(
            result.augmented_window.streams[stream_key].sample_mask,
            window.streams[stream_key].sample_mask,
        )

    torch.testing.assert_close(
        result.augmented_window.current_calibration["imu"].transform,
        window.current_calibration["imu"].transform,
    )

    torch.testing.assert_close(
        result.augmented_window.current_calibration["imu"].time_offset,
        window.current_calibration["imu"].time_offset,
    )

    trajectory = result.calibration_truth["imu"]

    torch.testing.assert_close(
        trajectory.pre_event.transform,
        window.current_calibration["imu"].transform,
    )

    torch.testing.assert_close(
        trajectory.pre_event.time_offset,
        window.current_calibration["imu"].time_offset,
    )

    assert not torch.any(
        trajectory.event.occurred
    )

    assert result.record.frame_randomization_by_key == {}
    assert result.record.prior_perturbation_xi_by_key == {}
    assert result.record.prior_perturbation_tau_by_key == {}
    assert result.record.noise_bias_by_stream == {}


def test_full_pipeline_keeps_physics_truth_prior_and_noise_separate() -> None:
    """
    Exercise frame randomization, prior perturbation, a true temporal event,
    rendering, and vector bias in one augmentation call.

    The test verifies the scientific separation between:

        true pre-event calibration,
        model input prior,
        true event trajectory,
        rendered timestamps,
        measurement noise.
    """

    window = _make_two_stream_imu_window()

    config = AugmentationConfig(
        frame_randomization=FrameRandomizationConfig(
            enabled=True,
        ),
        prior_perturbation=PriorPerturbationConfig(
            enabled=True,
            probability=1.0,
            rotation=PerturbationMagnitudeConfig(
                maximum=0.2,
            ),
            time_offset=PerturbationMagnitudeConfig(
                maximum=0.05,
            ),
        ),
        calibration_event=CalibrationEventConfig(
            enabled=True,
            event_probability=1.0,
            rotation_probability=0.0,
            translation_probability=0.0,
            time_offset_probability=1.0,
            time_offset=PerturbationMagnitudeConfig(
                maximum=0.1,
            ),
            profile_weights={
                TransitionProfile.STEP: 1.0,
                TransitionProfile.LINEAR: 0.0,
                TransitionProfile.SMOOTHSTEP: 0.0,
            },
            minimum_pre_event_fraction=0.2,
            minimum_post_event_fraction=0.2,
        ),
        noise=NoiseAugmentationConfig(
            enabled=True,
            vector_by_type={
                MeasurementType.IMU_GYROSCOPE: VectorNoiseConfig(
                    gaussian_std=0.0,
                    window_bias_std=0.01,
                )
            },
        ),
    )

    pipeline = AugmentationPipeline(
        config
    )

    result = pipeline(
        window,
        generator=torch.Generator().manual_seed(42),
    )

    result.validate()

    trajectory = result.calibration_truth["imu"]
    event = trajectory.event

    A = result.record.frame_randomization_by_key["imu"]

    prior_delta_xi = result.record.prior_perturbation_xi_by_key["imu"]
    prior_delta_tau = result.record.prior_perturbation_tau_by_key["imu"]

    gyro_bias = result.record.noise_bias_by_stream["gyro"]

    # The frame-randomized true pre-event calibration follows
    #
    #     T'_WS = T_WS @ A.
    expected_true_pre_event_transform = (
        window.current_calibration["imu"].transform
        @ A
    )

    torch.testing.assert_close(
        trajectory.pre_event.transform,
        expected_true_pre_event_transform,
        atol=ATOL,
        rtol=RTOL,
    )

    torch.testing.assert_close(
        trajectory.pre_event.time_offset,
        window.current_calibration["imu"].time_offset,
    )

    # The model receives an imperfect prior generated independently from the
    # true physical event.
    expected_prior_transform = (
        se3_exp(prior_delta_xi)
        @ trajectory.pre_event.transform
    )

    expected_prior_tau = (
        trajectory.pre_event.time_offset
        + prior_delta_tau
    )

    torch.testing.assert_close(
        result.augmented_window.current_calibration["imu"].transform,
        expected_prior_transform,
        atol=ATOL,
        rtol=RTOL,
    )

    torch.testing.assert_close(
        result.augmented_window.current_calibration["imu"].time_offset,
        expected_prior_tau,
        atol=ATOL,
        rtol=RTOL,
    )

    # The configured event is temporal only.
    assert torch.all(
        event.occurred
    )

    assert event.profiles == (
        TransitionProfile.STEP,
    )

    torch.testing.assert_close(
        event.delta_xi,
        torch.zeros_like(
            event.delta_xi
        ),
    )

    assert torch.any(
        event.delta_tau != 0.0
    )

    # STEP temporal rendering shifts only measurements at or after the true
    # change time:
    #
    #     t_rendered = t_original - delta_tau.
    original_timestamps = window.streams["gyro"].timestamps

    post_event = (
        original_timestamps
        >= event.change_time_s
    )

    expected_timestamps = torch.where(
        post_event,
        original_timestamps - event.delta_tau,
        original_timestamps,
    )

    torch.testing.assert_close(
        result.augmented_window.streams["gyro"].timestamps,
        expected_timestamps,
        atol=ATOL,
        rtol=RTOL,
    )

    torch.testing.assert_close(
        result.augmented_window.streams["accel"].timestamps,
        expected_timestamps,
        atol=ATOL,
        rtol=RTOL,
    )

    # Gyroscope and accelerometer share the same physical IMU key, so frame
    # randomization must use the same R_A for both streams.
    R_A = A[..., :3, :3]

    expected_gyro_values = (
        window.streams["gyro"].values
        @ R_A
    )

    expected_accel_values = (
        window.streams["accel"].values
        @ R_A
    )

    # Only the gyroscope has an additional constant window bias.
    expected_gyro_values = (
        expected_gyro_values
        + gyro_bias[:, None, :]
    )

    torch.testing.assert_close(
        result.augmented_window.streams["gyro"].values,
        expected_gyro_values,
        atol=ATOL,
        rtol=RTOL,
    )

    torch.testing.assert_close(
        result.augmented_window.streams["accel"].values,
        expected_accel_values,
        atol=ATOL,
        rtol=RTOL,
    )

    # The true final calibration is determined by the physical event, not by
    # the intentionally incorrect model prior.
    final_state = trajectory.final_state()

    torch.testing.assert_close(
        final_state.transform,
        trajectory.pre_event.transform,
        atol=ATOL,
        rtol=RTOL,
    )

    torch.testing.assert_close(
        final_state.time_offset,
        trajectory.pre_event.time_offset + event.delta_tau,
        atol=ATOL,
        rtol=RTOL,
    )