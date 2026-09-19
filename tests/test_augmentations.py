"""Semantic tests for synthetic sensor-calibration augmentations."""

from __future__ import annotations

import math

import pytest
import torch

from obscalib.augmentations.calibration_event import CalibrationEventSampler
from obscalib.augmentations.config import CalibrationEventConfig, FrameRandomizationConfig, NoiseAugmentationConfig, PerturbationDistribution, PerturbationMagnitudeConfig, PriorPerturbationConfig, SE3NoiseConfig, SO3NoiseConfig, VectorNoiseConfig
from obscalib.augmentations.frame_randomization import SensorFrameRandomizer
from obscalib.augmentations.noise import MeasurementNoiseAugmenter
from obscalib.augmentations.prior_perturbation import CalibrationPriorPerturber
from obscalib.augmentations.profiles import TransitionProfile, evaluate_transition_profiles
from obscalib.augmentations.rendering import CalibrationEventRenderer
from obscalib.augmentations.sampling import sample_isotropic_perturbation, sample_perturbation_magnitudes, sample_signed_scalar_perturbation, sample_uniform_so3
from obscalib.augmentations.structures import CalibrationEvent, CalibrationTrajectory
from obscalib.augmentations.trajectory import evaluate_calibration_trajectory
from obscalib.calibration.state import CalibrationState
from obscalib.data.structures import GeometryType, MeasurementType, SensorMetadata, SensorStreamBatch, WindowBatch
from obscalib.geometry.lie import se3_exp, so3_exp
from obscalib.geometry.processing import GeometryProcessor
from obscalib.augmentations.config import SamplingRateAugmentationConfig
from obscalib.augmentations.sampling_rate import SamplingRateAugmenter, _linear_resample, _reduce_relative_se3_scan_rate


DTYPE = torch.float64
ATOL = 1e-9
RTOL = 1e-7


def _identity_state(
    batch_size: int = 1,
    time_offset: float = 0.0,
) -> CalibrationState:
    """Construct an identity calibration state."""

    transform = torch.eye(
        4,
        dtype=DTYPE,
    ).unsqueeze(0).repeat(batch_size, 1, 1)

    tau = torch.full(
        (batch_size, 1),
        time_offset,
        dtype=DTYPE,
    )

    return CalibrationState(
        transform=transform,
        time_offset=tau,
    )


def _state_from_xi(
    xi: torch.Tensor,
    time_offset: float = 0.0,
) -> CalibrationState:
    """Construct a calibration state from one or more SE3 tangents."""

    if xi.ndim == 1:
        xi = xi.unsqueeze(0)

    batch_size = xi.shape[0]

    return CalibrationState(
        transform=se3_exp(xi),
        time_offset=torch.full(
            (batch_size, 1),
            time_offset,
            dtype=xi.dtype,
            device=xi.device,
        ),
    )


def _make_stream(values: torch.Tensor, timestamps: torch.Tensor, sample_mask=None, interval_start_timestamps: torch.Tensor | None = None) -> SensorStreamBatch:
    """Construct one fully valid padded sensor stream for augmentation tests."""

    if sample_mask is None:
        sample_mask = torch.ones(timestamps.shape, dtype=torch.bool, device=timestamps.device)

    return SensorStreamBatch(
        values=values,
        timestamps=timestamps,
        sample_mask=sample_mask,
        interval_start_timestamps=interval_start_timestamps,
    )


def _make_vector_window(
    *,
    batch_size: int = 1,
    timestamps: torch.Tensor | None = None,
    calibration_key: str = "imu",
) -> WindowBatch:
    """Construct a simple vector-measurement window."""

    if timestamps is None:
        timestamps = torch.tensor(
            [[0.0, 2.0, 4.0, 6.0, 8.0, 10.0]],
            dtype=DTYPE,
        ).repeat(batch_size, 1)

    num_samples = timestamps.shape[1]

    values = torch.zeros(
        batch_size,
        num_samples,
        3,
        dtype=DTYPE,
    )

    values[..., 0] = 1.0
    values[..., 1] = 0.5

    return WindowBatch(
        streams={
            "gyro": _make_stream(
                values=values,
                timestamps=timestamps,
            )
        },
        current_calibration={
            calibration_key: _identity_state(batch_size),
        },
        metadata={
            "gyro": SensorMetadata(
                measurement_type=MeasurementType.IMU_GYROSCOPE,
                geometry_type=GeometryType.VECTOR,
                calibration_key=calibration_key,
            )
        },
        targets=None,
    )


def _make_event(
    *,
    delta_xi: torch.Tensor,
    delta_tau: torch.Tensor,
    change_time_s: float,
    profile: TransitionProfile,
    transition_duration_s: float = 0.0,
) -> CalibrationEvent:
    """Construct one deterministic B=1 calibration event."""

    event = CalibrationEvent(
        occurred=torch.tensor(
            [[True]],
            dtype=torch.bool,
        ),
        profiles=(profile,),
        change_time_s=torch.tensor(
            [[change_time_s]],
            dtype=DTYPE,
        ),
        transition_duration_s=torch.tensor(
            [[transition_duration_s]],
            dtype=DTYPE,
        ),
        delta_xi=delta_xi.reshape(1, 6).to(dtype=DTYPE),
        delta_tau=delta_tau.reshape(1, 1).to(dtype=DTYPE),
    )

    event.validate()

    return event


def test_sample_uniform_so3_returns_valid_rotations() -> None:
    """Haar SO3 sampling must produce proper orthogonal rotation matrices."""

    generator = torch.Generator().manual_seed(10)

    R = sample_uniform_so3(
        batch_size=128,
        device=torch.device("cpu"),
        dtype=DTYPE,
        generator=generator,
    )

    identity = torch.eye(
        3,
        dtype=DTYPE,
    ).expand_as(R)

    assert R.shape == (128, 3, 3)

    torch.testing.assert_close(
        R.transpose(-1, -2) @ R,
        identity,
        atol=1e-10,
        rtol=1e-10,
    )

    torch.testing.assert_close(
        torch.linalg.det(R),
        torch.ones(128, dtype=DTYPE),
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.parametrize(
    "distribution",
    [
        PerturbationDistribution.UNIFORM,
        PerturbationDistribution.TRUNCATED_NORMAL,
        PerturbationDistribution.TRUNCATED_STUDENT_T,
    ],
)
def test_perturbation_magnitudes_respect_bound(
    distribution: PerturbationDistribution,
) -> None:
    """All perturbation distributions must respect their configured hard bound."""

    config = PerturbationMagnitudeConfig(
        maximum=0.2,
        distribution=distribution,
        scale=None if distribution == PerturbationDistribution.UNIFORM else 0.05,
        degrees_of_freedom=3,
    )

    magnitudes = sample_perturbation_magnitudes(
        config=config,
        batch_size=512,
        device=torch.device("cpu"),
        dtype=DTYPE,
        generator=torch.Generator().manual_seed(4),
    )

    assert magnitudes.shape == (512, 1)
    assert torch.all(magnitudes >= 0.0)
    assert torch.all(magnitudes <= config.maximum)


def test_isotropic_and_signed_perturbation_shapes_and_bounds() -> None:
    """Vector and scalar samplers must preserve their configured magnitudes."""

    config = PerturbationMagnitudeConfig(
        maximum=0.3,
    )

    generator = torch.Generator().manual_seed(8)

    vectors = sample_isotropic_perturbation(
        config=config,
        batch_size=64,
        dimension=3,
        device=torch.device("cpu"),
        dtype=DTYPE,
        generator=generator,
    )

    scalars = sample_signed_scalar_perturbation(
        config=config,
        batch_size=64,
        device=torch.device("cpu"),
        dtype=DTYPE,
        generator=generator,
    )

    assert vectors.shape == (64, 3)
    assert scalars.shape == (64, 1)

    assert torch.all(
        torch.linalg.vector_norm(vectors, dim=-1) <= 0.3
    )

    assert torch.all(
        torch.abs(scalars) <= 0.3
    )


def test_transition_profiles_have_expected_values() -> None:
    """STEP, LINEAR, and SMOOTHSTEP must follow their agreed timing semantics."""

    timestamps_s = torch.tensor(
        [
            [4.0, 4.5, 5.0, 5.5, 6.0],
            [4.0, 4.5, 5.0, 5.5, 6.0],
            [4.0, 4.5, 5.0, 5.5, 6.0],
            [4.0, 4.5, 5.0, 5.5, 6.0],
        ],
        dtype=DTYPE,
    )

    progress = evaluate_transition_profiles(
        timestamps_s=timestamps_s,
        change_time_s=torch.tensor(
            [[5.0], [5.0], [5.0], [0.0]],
            dtype=DTYPE,
        ),
        transition_duration_s=torch.tensor(
            [[0.0], [2.0], [2.0], [0.0]],
            dtype=DTYPE,
        ),
        profiles=(
            TransitionProfile.STEP,
            TransitionProfile.LINEAR,
            TransitionProfile.SMOOTHSTEP,
            None,
        ),
    )

    expected = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [0.0, 0.15625, 0.5, 0.84375, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=DTYPE,
    )

    torch.testing.assert_close(
        progress,
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_frame_randomization_preserves_canonical_world_measurements() -> None:
    """
    Randomizing a sensor coordinate convention must not change the physical
    canonical measurement after applying the correspondingly randomized
    calibration.
    """

    batch_size = 2

    timestamps = torch.tensor(
        [[1.0, 2.0, 3.0]],
        dtype=DTYPE,
    ).repeat(batch_size, 1)

    vector_values = torch.tensor(
        [[[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0], [0.2, -0.3, 0.7]]],
        dtype=DTYPE,
    ).repeat(batch_size, 1, 1)

    so3_tangents = torch.tensor(
        [
            [0.1, 0.0, 0.0],
            [0.0, -0.15, 0.05],
            [0.05, 0.03, -0.02],
        ],
        dtype=DTYPE,
    )

    so3_values = so3_exp(
        so3_tangents
    ).unsqueeze(0).repeat(batch_size, 1, 1, 1)

    se3_tangents = torch.tensor(
        [
            [0.05, 0.00, 0.00, 0.3, 0.0, 0.0],
            [0.00, 0.08, 0.00, 0.0, 0.2, 0.0],
            [0.00, 0.00, -0.04, 0.0, 0.0, 0.1],
        ],
        dtype=DTYPE,
    )

    se3_values = se3_exp(
        se3_tangents
    ).unsqueeze(0).repeat(batch_size, 1, 1, 1)

    window = WindowBatch(
        streams={
            "gyro": _make_stream(
                vector_values,
                timestamps,
            ),
            "camera": _make_stream(
                so3_values,
                timestamps,
            ),
            "lidar": _make_stream(
                se3_values,
                timestamps,
            ),
        },
        current_calibration={
            "imu": _state_from_xi(
                torch.tensor(
                    [0.1, -0.05, 0.02, 0.3, -0.2, 0.1],
                    dtype=DTYPE,
                ).repeat(batch_size, 1)
            ),
            "camera": _state_from_xi(
                torch.tensor(
                    [-0.04, 0.08, 0.03, -0.2, 0.1, 0.4],
                    dtype=DTYPE,
                ).repeat(batch_size, 1)
            ),
            "lidar": _state_from_xi(
                torch.tensor(
                    [0.02, 0.06, -0.07, 0.5, 0.2, -0.1],
                    dtype=DTYPE,
                ).repeat(batch_size, 1)
            ),
        },
        metadata={
            "gyro": SensorMetadata(
                measurement_type=MeasurementType.IMU_GYROSCOPE,
                geometry_type=GeometryType.VECTOR,
                calibration_key="imu",
            ),
            "camera": SensorMetadata(
                measurement_type=MeasurementType.CAMERA_POSE,
                geometry_type=GeometryType.SO3,
                calibration_key="camera",
            ),
            "lidar": SensorMetadata(
                measurement_type=MeasurementType.LIDAR_POSE,
                geometry_type=GeometryType.SE3,
                calibration_key="lidar",
            ),
        },
        targets=None,
    )

    geometry_processor = GeometryProcessor()

    canonical_before = geometry_processor(
        window.streams,
        window.metadata,
        window.current_calibration,
    )

    randomizer = SensorFrameRandomizer(
        FrameRandomizationConfig(
            enabled=True,
        )
    )

    randomized_window, randomized_calibration, randomization_by_key = randomizer(
        window,
        generator=torch.Generator().manual_seed(19),
    )

    canonical_after = geometry_processor(
        randomized_window.streams,
        randomized_window.metadata,
        randomized_calibration,
    )

    for stream_key in window.streams:
        torch.testing.assert_close(
            canonical_after[stream_key].values,
            canonical_before[stream_key].values,
            atol=1e-8,
            rtol=1e-7,
        )

    for A in randomization_by_key.values():
        torch.testing.assert_close(
            A[..., :3, 3],
            torch.zeros_like(A[..., :3, 3]),
        )

        torch.testing.assert_close(
            A[..., 3, :],
            torch.tensor(
                [0.0, 0.0, 0.0, 1.0],
                dtype=DTYPE,
            ).expand_as(A[..., 3, :]),
        )


def test_prior_perturbation_matches_recorded_left_update() -> None:
    """Recorded prior perturbations must exactly reconstruct the supplied prior."""

    true_state = _state_from_xi(
        torch.tensor(
            [
                [0.10, -0.04, 0.03, 0.2, -0.1, 0.4],
                [-0.05, 0.06, 0.02, -0.3, 0.2, 0.1],
            ],
            dtype=DTYPE,
        ),
        time_offset=0.03,
    )

    config = PriorPerturbationConfig(
        enabled=True,
        probability=1.0,
        rotation=PerturbationMagnitudeConfig(
            maximum=0.2,
        ),
        translation=PerturbationMagnitudeConfig(
            maximum=0.3,
        ),
        time_offset=PerturbationMagnitudeConfig(
            maximum=0.05,
        ),
    )

    perturber = CalibrationPriorPerturber(
        config,
    )

    priors, delta_xi_by_key, delta_tau_by_key = perturber(
        {"imu": true_state},
        generator=torch.Generator().manual_seed(30),
    )

    delta_xi = delta_xi_by_key["imu"]
    delta_tau = delta_tau_by_key["imu"]

    expected_transform = (
        se3_exp(delta_xi)
        @ true_state.transform
    )

    expected_tau = (
        true_state.time_offset
        + delta_tau
    )

    torch.testing.assert_close(
        priors["imu"].transform,
        expected_transform,
    )

    torch.testing.assert_close(
        priors["imu"].time_offset,
        expected_tau,
    )

    assert torch.all(
        torch.linalg.vector_norm(delta_xi[:, :3], dim=-1)
        <= 0.2
    )

    assert torch.all(
        torch.linalg.vector_norm(delta_xi[:, 3:], dim=-1)
        <= 0.3
    )

    assert torch.all(
        torch.abs(delta_tau) <= 0.05
    )


def test_zero_probability_prior_perturbation_is_exact_identity() -> None:
    """An enabled stage with zero occurrence probability must leave the prior exact."""

    state = _identity_state(
        batch_size=2,
        time_offset=0.1,
    )

    perturber = CalibrationPriorPerturber(
        PriorPerturbationConfig(
            enabled=True,
            probability=0.0,
            rotation=PerturbationMagnitudeConfig(
                maximum=0.3,
            ),
        )
    )

    priors, delta_xi_by_key, delta_tau_by_key = perturber(
        {"imu": state},
        generator=torch.Generator().manual_seed(7),
    )

    torch.testing.assert_close(
        priors["imu"].transform,
        state.transform,
    )

    torch.testing.assert_close(
        priors["imu"].time_offset,
        state.time_offset,
    )

    torch.testing.assert_close(
        delta_xi_by_key["imu"],
        torch.zeros(2, 6, dtype=DTYPE),
    )

    torch.testing.assert_close(
        delta_tau_by_key["imu"],
        torch.zeros(2, 1, dtype=DTYPE),
    )


def test_event_sampler_step_event_respects_component_and_window_constraints() -> None:
    """STEP events must select configured components and stay inside context bounds."""

    batch_size = 4

    window = _make_vector_window(
        batch_size=batch_size,
    )

    config = CalibrationEventConfig(
        enabled=True,
        event_probability=1.0,
        rotation_probability=1.0,
        translation_probability=0.0,
        time_offset_probability=0.0,
        rotation=PerturbationMagnitudeConfig(
            maximum=0.2,
        ),
        profile_weights={
            TransitionProfile.STEP: 1.0,
            TransitionProfile.LINEAR: 0.0,
            TransitionProfile.SMOOTHSTEP: 0.0,
        },
        minimum_pre_event_fraction=0.2,
        minimum_post_event_fraction=0.2,
    )

    sampler = CalibrationEventSampler(
        config,
    )

    events = sampler(
        window,
        window.current_calibration,
        generator=torch.Generator().manual_seed(12),
    )

    event = events["imu"]

    assert torch.all(event.occurred)
    assert event.profiles == (TransitionProfile.STEP,) * batch_size

    torch.testing.assert_close(
        event.transition_duration_s,
        torch.zeros(
            batch_size,
            1,
            dtype=DTYPE,
        ),
    )

    assert torch.all(
        event.change_time_s >= 2.0
    )

    assert torch.all(
        event.change_time_s <= 8.0
    )

    assert torch.all(
        torch.linalg.vector_norm(event.delta_xi[:, :3], dim=-1)
        <= 0.2
    )

    torch.testing.assert_close(
        event.delta_xi[:, 3:],
        torch.zeros(
            batch_size,
            3,
            dtype=DTYPE,
        ),
    )

    torch.testing.assert_close(
        event.delta_tau,
        torch.zeros(
            batch_size,
            1,
            dtype=DTYPE,
        ),
    )


def test_event_sampler_finite_transition_fits_inside_window() -> None:
    """A gradual transition must leave the requested stable context on both sides."""

    window = _make_vector_window()

    config = CalibrationEventConfig(
        enabled=True,
        event_probability=1.0,
        rotation_probability=1.0,
        rotation=PerturbationMagnitudeConfig(
            maximum=0.2,
        ),
        profile_weights={
            TransitionProfile.STEP: 0.0,
            TransitionProfile.LINEAR: 1.0,
            TransitionProfile.SMOOTHSTEP: 0.0,
        },
        minimum_transition_duration_s=2.0,
        maximum_transition_duration_s=2.0,
        minimum_pre_event_fraction=0.2,
        minimum_post_event_fraction=0.2,
    )

    event = CalibrationEventSampler(config)(
        window,
        window.current_calibration,
        generator=torch.Generator().manual_seed(5),
    )["imu"]

    torch.testing.assert_close(
        event.transition_duration_s,
        torch.tensor(
            [[2.0]],
            dtype=DTYPE,
        ),
    )

    transition_start = (
        event.change_time_s
        - 0.5 * event.transition_duration_s
    )

    transition_end = (
        event.change_time_s
        + 0.5 * event.transition_duration_s
    )

    assert torch.all(
        transition_start >= 2.0
    )

    assert torch.all(
        transition_end <= 8.0
    )


def test_calibration_trajectory_uses_same_progress_for_se3_and_time_offset() -> None:
    """Spatial and temporal event components must share one transition profile."""

    pre_event = _state_from_xi(
        torch.tensor(
            [0.02, -0.03, 0.01, 0.1, 0.2, -0.1],
            dtype=DTYPE,
        ),
        time_offset=0.1,
    )

    delta_xi = torch.tensor(
        [[0.1, -0.05, 0.08, 0.3, -0.2, 0.4]],
        dtype=DTYPE,
    )

    delta_tau = torch.tensor(
        [[0.4]],
        dtype=DTYPE,
    )

    event = _make_event(
        delta_xi=delta_xi,
        delta_tau=delta_tau,
        change_time_s=5.0,
        profile=TransitionProfile.LINEAR,
        transition_duration_s=2.0,
    )

    timestamps = torch.tensor(
        [[4.0, 5.0, 6.0]],
        dtype=DTYPE,
    )

    transforms, time_offsets = evaluate_calibration_trajectory(
        pre_event=pre_event,
        event=event,
        timestamps_s=timestamps,
    )

    progress = torch.tensor(
        [0.0, 0.5, 1.0],
        dtype=DTYPE,
    )

    expected_delta_transform = se3_exp(
        progress[:, None] * delta_xi[0]
    )

    expected_transform = (
        expected_delta_transform
        @ pre_event.transform[0]
    )

    expected_tau = (
        pre_event.time_offset[0, 0]
        + progress * delta_tau[0, 0]
    )

    torch.testing.assert_close(
        transforms[0],
        expected_transform,
    )

    torch.testing.assert_close(
        time_offsets[0, :, 0],
        expected_tau,
    )


def test_renderer_vector_step_event_changes_frame_and_timestamp() -> None:
    """A STEP event must re-express vector data and synthesize the new measured time."""

    angle = math.pi / 2.0

    delta_xi = torch.tensor(
        [[0.0, 0.0, angle, 0.0, 0.0, 0.0]],
        dtype=DTYPE,
    )

    delta_tau = torch.tensor(
        [[0.2]],
        dtype=DTYPE,
    )

    event = _make_event(
        delta_xi=delta_xi,
        delta_tau=delta_tau,
        change_time_s=1.5,
        profile=TransitionProfile.STEP,
    )

    values = torch.tensor(
        [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
        dtype=DTYPE,
    )

    timestamps = torch.tensor(
        [[1.0, 2.0, 3.0]],
        dtype=DTYPE,
    )

    window = WindowBatch(
        streams={
            "gyro": _make_stream(
                values,
                timestamps,
            )
        },
        current_calibration={
            "imu": _identity_state(),
        },
        metadata={
            "gyro": SensorMetadata(
                measurement_type=MeasurementType.IMU_GYROSCOPE,
                geometry_type=GeometryType.VECTOR,
                calibration_key="imu",
            )
        },
        targets=None,
    )

    trajectory = CalibrationTrajectory(
        pre_event=window.current_calibration["imu"],
        event=event,
    )

    rendered = CalibrationEventRenderer()(
        window,
        {
            "imu": trajectory,
        },
    )

    R_A = se3_exp(
        delta_xi
    )[0, :3, :3]

    expected_values = values.clone()

    expected_values[:, 1:] = (
        values[:, 1:].unsqueeze(-2)
        @ R_A
    ).squeeze(-2)

    expected_timestamps = torch.tensor(
        [[1.0, 1.8, 2.8]],
        dtype=DTYPE,
    )

    torch.testing.assert_close(
        rendered.streams["gyro"].values,
        expected_values,
    )

    torch.testing.assert_close(
        rendered.streams["gyro"].timestamps,
        expected_timestamps,
    )


def test_renderer_relative_se3_uses_start_and_end_calibration() -> None:
    """
    A relative update crossing an event must use A_start^-1 DeltaT A_end.

    Identity updates before and after a STEP remain identity, while the update
    that crosses the step contains the calibration-frame change.
    """

    delta_xi = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
        dtype=DTYPE,
    )

    interval_start_timestamps = torch.tensor(
        [[0.0, 1.0, 2.0]],
        dtype=DTYPE,
    )

    event = _make_event(
        delta_xi=delta_xi,
        delta_tau=torch.zeros(
            1,
            1,
            dtype=DTYPE,
        ),
        change_time_s=1.5,
        profile=TransitionProfile.STEP,
    )

    identity = torch.eye(
        4,
        dtype=DTYPE,
    )

    values = identity.reshape(
        1,
        1,
        4,
        4,
    ).repeat(
        1,
        3,
        1,
        1,
    )

    timestamps = torch.tensor(
        [[1.0, 2.0, 3.0]],
        dtype=DTYPE,
    )

    state = _identity_state()

    window = WindowBatch(
        streams={
            "lidar": _make_stream(
                values,
                timestamps,
                interval_start_timestamps=interval_start_timestamps,
            )
        },
        current_calibration={
            "lidar": state,
        },
        metadata={
            "lidar": SensorMetadata(
                measurement_type=MeasurementType.LIDAR_POSE,
                geometry_type=GeometryType.SE3,
                calibration_key="lidar",
            )
        },
        targets=None,
    )

    trajectory = CalibrationTrajectory(
        pre_event=state,
        event=event,
    )

    rendered = CalibrationEventRenderer()(
        window,
        {
            "lidar": trajectory,
        },
    )

    A_final = se3_exp(
        delta_xi
    )[0]

    expected = values.clone()
    expected[0, 1] = A_final

    torch.testing.assert_close(
        rendered.streams["lidar"].values,
        expected,
    )
    torch.testing.assert_close(
        rendered.streams["lidar"].interval_start_timestamps,
        interval_start_timestamps,
    )

    torch.testing.assert_close(
        rendered.streams["lidar"].timestamps,
        timestamps,
    )
    


def test_vector_noise_bias_is_constant_inside_window_and_padding_is_unchanged() -> None:
    """VECTOR window bias must be constant per batch item rather than a random walk."""

    values = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [9.0, 9.0, 9.0],
            ]
        ],
        dtype=DTYPE,
    )

    timestamps = torch.tensor(
        [[0.0, 1.0, 0.0]],
        dtype=DTYPE,
    )

    sample_mask = torch.tensor(
        [[True, True, False]],
    )

    window = WindowBatch(
        streams={
            "gyro": _make_stream(
                values,
                timestamps,
                sample_mask=sample_mask,
            )
        },
        current_calibration={
            "imu": _identity_state(),
        },
        metadata={
            "gyro": SensorMetadata(
                measurement_type=MeasurementType.IMU_GYROSCOPE,
                geometry_type=GeometryType.VECTOR,
                calibration_key="imu",
            )
        },
        targets=None,
    )

    augmenter = MeasurementNoiseAugmenter(
        NoiseAugmentationConfig(
            enabled=True,
            vector_by_type={
                MeasurementType.IMU_GYROSCOPE: VectorNoiseConfig(
                    gaussian_std=0.0,
                    window_bias_std=0.2,
                )
            },
        )
    )

    augmented, bias_by_stream = augmenter(
        window,
        generator=torch.Generator().manual_seed(14),
    )

    bias = bias_by_stream["gyro"]

    difference = (
        augmented.streams["gyro"].values
        - values
    )

    torch.testing.assert_close(
        difference[0, 0],
        bias[0],
    )

    torch.testing.assert_close(
        difference[0, 1],
        bias[0],
    )

    torch.testing.assert_close(
        augmented.streams["gyro"].values[0, 2],
        values[0, 2],
    )


def test_so3_and_se3_noise_preserve_group_structure() -> None:
    """Manifold-valued noise must produce valid SO3 and homogeneous SE3 matrices."""

    timestamps = torch.tensor(
        [[1.0, 2.0, 3.0]],
        dtype=DTYPE,
    )

    R = torch.eye(
        3,
        dtype=DTYPE,
    ).reshape(
        1,
        1,
        3,
        3,
    ).repeat(
        1,
        3,
        1,
        1,
    )

    T = torch.eye(
        4,
        dtype=DTYPE,
    ).reshape(
        1,
        1,
        4,
        4,
    ).repeat(
        1,
        3,
        1,
        1,
    )

    window = WindowBatch(
        streams={
            "camera": _make_stream(
                R,
                timestamps,
            ),
            "lidar": _make_stream(
                T,
                timestamps,
            ),
        },
        current_calibration={
            "camera": _identity_state(),
            "lidar": _identity_state(),
        },
        metadata={
            "camera": SensorMetadata(
                measurement_type=MeasurementType.CAMERA_POSE,
                geometry_type=GeometryType.SO3,
                calibration_key="camera",
            ),
            "lidar": SensorMetadata(
                measurement_type=MeasurementType.LIDAR_POSE,
                geometry_type=GeometryType.SE3,
                calibration_key="lidar",
            ),
        },
        targets=None,
    )

    augmenter = MeasurementNoiseAugmenter(
        NoiseAugmentationConfig(
            enabled=True,
            so3_by_type={
                MeasurementType.CAMERA_POSE: SO3NoiseConfig(
                    rotation_std=0.05,
                )
            },
            se3_by_type={
                MeasurementType.LIDAR_POSE: SE3NoiseConfig(
                    rotation_std=0.05,
                    translation_std=0.1,
                )
            },
        )
    )

    augmented, _ = augmenter(
        window,
        generator=torch.Generator().manual_seed(123),
    )

    noisy_R = augmented.streams["camera"].values
    noisy_T = augmented.streams["lidar"].values

    identity_3 = torch.eye(
        3,
        dtype=DTYPE,
    ).expand_as(noisy_R)

    torch.testing.assert_close(
        noisy_R.transpose(-1, -2) @ noisy_R,
        identity_3,
        atol=1e-10,
        rtol=1e-10,
    )

    torch.testing.assert_close(
        torch.linalg.det(noisy_R),
        torch.ones(
            noisy_R.shape[:2],
            dtype=DTYPE,
        ),
        atol=1e-10,
        rtol=1e-10,
    )

    expected_last_row = torch.tensor(
        [0.0, 0.0, 0.0, 1.0],
        dtype=DTYPE,
    ).expand_as(
        noisy_T[..., 3, :]
    )

    torch.testing.assert_close(
        noisy_T[..., 3, :],
        expected_last_row,
    )

    noisy_T_R = noisy_T[..., :3, :3]

    torch.testing.assert_close(
        noisy_T_R.transpose(-1, -2) @ noisy_T_R,
        torch.eye(
            3,
            dtype=DTYPE,
        ).expand_as(noisy_T_R),
        atol=1e-10,
        rtol=1e-10,
    )

def test_sampling_rate_linear_imu_resampling_preserves_linear_signal() -> None:
    """Linear interpolation must reproduce a linear IMU signal exactly."""

    timestamps = torch.arange(0.0, 1.0 + 1e-12, 0.01, dtype=DTYPE)

    values = torch.stack(
        (
            2.0 * timestamps + 1.0,
            -3.0 * timestamps + 0.5,
            0.25 * timestamps - 2.0,
        ),
        dim=-1,
    )

    resampled_values, resampled_timestamps = _linear_resample(values, timestamps, target_frequency_hz=37.0)

    expected_values = torch.stack(
        (
            2.0 * resampled_timestamps + 1.0,
            -3.0 * resampled_timestamps + 0.5,
            0.25 * resampled_timestamps - 2.0,
        ),
        dim=-1,
    )

    torch.testing.assert_close(resampled_values, expected_values, rtol=1e-10, atol=1e-10)

    assert resampled_timestamps[0].item() == pytest.approx(0.0)
    assert resampled_timestamps[-1].item() == pytest.approx(1.0)
    assert resampled_timestamps.shape[0] < timestamps.shape[0]

def test_sampling_rate_shared_imu_calibration_key_uses_same_target_frequency() -> None:
    """Gyroscope and accelerometer streams from one IMU must share the sampled target frequency."""

    timestamps = torch.arange(0.0, 1.0 + 1e-12, 0.01, dtype=DTYPE).reshape(1, -1)
    sample_mask = torch.ones_like(timestamps, dtype=torch.bool)

    gyro_values = torch.stack(
        (
            timestamps,
            2.0 * timestamps,
            3.0 * timestamps,
        ),
        dim=-1,
    )

    accel_values = torch.stack(
        (
            -timestamps,
            4.0 * timestamps,
            0.5 * timestamps,
        ),
        dim=-1,
    )

    window = WindowBatch(
        streams={
            "imu_gyro": SensorStreamBatch(values=gyro_values, timestamps=timestamps, sample_mask=sample_mask),
            "imu_accel": SensorStreamBatch(values=accel_values, timestamps=timestamps, sample_mask=sample_mask),
        },
        current_calibration={
            "imu": _identity_state(),
        },
        metadata={
            "imu_gyro": SensorMetadata(measurement_type=MeasurementType.IMU_GYROSCOPE, geometry_type=GeometryType.VECTOR, calibration_key="imu"),
            "imu_accel": SensorMetadata(measurement_type=MeasurementType.IMU_ACCELEROMETER, geometry_type=GeometryType.VECTOR, calibration_key="imu"),
        },
        targets=None,
    )

    augmenter = SamplingRateAugmenter(
        SamplingRateAugmentationConfig(
            enabled=True,
            minimum_imu_frequency_hz=40.0,
            imu_probability=1.0,
        )
    )

    generator = torch.Generator().manual_seed(12345)

    augmented_window, target_frequency_hz_by_stream, applied_by_stream = augmenter(window, generator=generator)

    torch.testing.assert_close(target_frequency_hz_by_stream["imu_gyro"], target_frequency_hz_by_stream["imu_accel"])

    assert applied_by_stream["imu_gyro"].item()
    assert applied_by_stream["imu_accel"].item()

    gyro = augmented_window.streams["imu_gyro"]
    accel = augmented_window.streams["imu_accel"]

    assert torch.equal(gyro.sample_mask, accel.sample_mask)
    torch.testing.assert_close(gyro.timestamps, accel.timestamps)

def test_sampling_rate_lidar_10hz_to_7hz_selects_expected_scans() -> None:
    """A 10 Hz LiDAR sequence reduced to 7 Hz must retain the nearest realizable scan pattern."""

    scan_timestamps = torch.arange(0.0, 1.0 + 1e-12, 0.1, dtype=DTYPE)

    values = torch.eye(4, dtype=DTYPE).reshape(1, 4, 4).repeat(10, 1, 1)
    values[:, 0, 3] = 1.0

    reduced_values, reduced_end_timestamps, reduced_start_timestamps = _reduce_relative_se3_scan_rate(
        values,
        scan_timestamps[:-1],
        scan_timestamps[1:],
        target_frequency_hz=7.0,
    )

    expected_scan_timestamps = torch.tensor(
        [0.0, 0.1, 0.3, 0.4, 0.6, 0.7, 0.9, 1.0],
        dtype=DTYPE,
    )

    torch.testing.assert_close(reduced_start_timestamps, expected_scan_timestamps[:-1], rtol=0.0, atol=1e-12)
    torch.testing.assert_close(reduced_end_timestamps, expected_scan_timestamps[1:], rtol=0.0, atol=1e-12)

    assert reduced_values.shape == (7, 4, 4)

    expected_x_translations = torch.tensor(
        [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0],
        dtype=DTYPE,
    )

    torch.testing.assert_close(reduced_values[:, 0, 3], expected_x_translations)

def test_sampling_rate_lidar_augmenter_preserves_interval_contract() -> None:
    """LiDAR scan-rate augmentation must return valid explicit relative intervals."""

    scan_timestamps = torch.arange(0.0, 1.0 + 1e-12, 0.1, dtype=DTYPE)

    values = torch.eye(4, dtype=DTYPE).reshape(1, 1, 4, 4).repeat(1, 10, 1, 1)
    values[0, :, 0, 3] = 0.1

    window = WindowBatch(
        streams={
            "lidar": SensorStreamBatch(
                values=values,
                timestamps=scan_timestamps[1:].reshape(1, -1),
                sample_mask=torch.ones(1, 10, dtype=torch.bool),
                interval_start_timestamps=scan_timestamps[:-1].reshape(1, -1),
            )
        },
        current_calibration={
            "lidar": _identity_state(),
        },
        metadata={
            "lidar": SensorMetadata(
                measurement_type=MeasurementType.LIDAR_POSE,
                geometry_type=GeometryType.SE3,
                calibration_key="lidar",
            )
        },
        targets=None,
    )

    augmenter = SamplingRateAugmenter(
        SamplingRateAugmentationConfig(
            enabled=True,
            minimum_lidar_frequency_hz=5.0,
            lidar_probability=1.0,
        )
    )

    generator = torch.Generator().manual_seed(4321)

    augmented_window, target_frequency_hz_by_stream, applied_by_stream = augmenter(window, generator=generator)

    stream = augmented_window.streams["lidar"]
    stream.validate()

    valid = stream.sample_mask[0]

    assert applied_by_stream["lidar"].item()
    assert 5.0 <= target_frequency_hz_by_stream["lidar"].item() <= 10.0
    assert stream.interval_start_timestamps is not None
    assert torch.all(stream.interval_start_timestamps[0, valid] < stream.timestamps[0, valid])

    if valid.sum() > 1:
        torch.testing.assert_close(stream.timestamps[0, valid][:-1], stream.interval_start_timestamps[0, valid][1:], rtol=0.0, atol=1e-9)

def test_sampling_rate_augmentation_disabled_preserves_window() -> None:
    """Disabled sampling-rate augmentation must leave streams unchanged."""

    timestamps = torch.tensor([[0.0, 0.01, 0.02, 0.03]], dtype=DTYPE)
    values = torch.randn(1, 4, 3, dtype=DTYPE)

    window = WindowBatch(
        streams={
            "imu_gyro": SensorStreamBatch(
                values=values,
                timestamps=timestamps,
                sample_mask=torch.ones_like(timestamps, dtype=torch.bool),
            )
        },
        current_calibration={
            "imu": _identity_state(),
        },
        metadata={
            "imu_gyro": SensorMetadata(
                measurement_type=MeasurementType.IMU_GYROSCOPE,
                geometry_type=GeometryType.VECTOR,
                calibration_key="imu",
            )
        },
        targets=None,
    )

    augmenter = SamplingRateAugmenter(SamplingRateAugmentationConfig())

    augmented_window, target_frequency_hz_by_stream, applied_by_stream = augmenter(window)

    assert augmented_window is window
    assert target_frequency_hz_by_stream == {}
    assert applied_by_stream == {}