"""Tests for carried calibration-state updates."""

import math

import pytest
import torch

from obscalib.calibration import CalibrationState, CalibrationUpdater
from obscalib.geometry import se3_exp
from obscalib.models.structures import CalibrationPrediction


def _make_prediction(
    delta_xi: torch.Tensor,
    delta_tau: torch.Tensor,
    *,
    change_event_logit: torch.Tensor | None = None,
    change_time: torch.Tensor | None = None,
) -> CalibrationPrediction:
    """Build a complete prediction while focusing tests on state corrections."""

    batch_size = delta_xi.shape[0]
    dtype = delta_xi.dtype
    device = delta_xi.device

    if change_event_logit is None:
        change_event_logit = torch.zeros(
            batch_size,
            1,
            dtype=dtype,
            device=device,
        )

    if change_time is None:
        change_time = torch.zeros(
            batch_size,
            1,
            dtype=dtype,
            device=device,
        )

    return CalibrationPrediction(
        change_event_logit=change_event_logit,
        change_time=change_time,
        delta_xi=delta_xi,
        delta_tau=delta_tau,
    )


def _identity_state(
    batch_size: int,
    *,
    dtype: torch.dtype = torch.float64,
) -> CalibrationState:
    """Create a batched identity calibration state."""

    transform = torch.eye(
        4,
        dtype=dtype,
    ).expand(batch_size, -1, -1).clone()

    time_offset = torch.zeros(
        batch_size,
        1,
        dtype=dtype,
    )

    return CalibrationState(
        transform=transform,
        time_offset=time_offset,
    )


def test_zero_prediction_preserves_calibration_state() -> None:
    updater = CalibrationUpdater()
    state = _identity_state(batch_size=3)

    prediction = _make_prediction(
        delta_xi=torch.zeros(3, 6, dtype=torch.float64),
        delta_tau=torch.zeros(3, 1, dtype=torch.float64),
    )

    updated = updater.update(state, prediction)

    torch.testing.assert_close(
        updated.transform,
        state.transform,
    )
    torch.testing.assert_close(
        updated.time_offset,
        state.time_offset,
    )


def test_update_applies_independent_batched_corrections() -> None:
    updater = CalibrationUpdater()
    state = _identity_state(batch_size=3)

    delta_xi = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, -3.0],
        ],
        dtype=torch.float64,
    )
    delta_tau = torch.tensor(
        [[0.1], [-0.2], [0.3]],
        dtype=torch.float64,
    )

    prediction = _make_prediction(
        delta_xi,
        delta_tau,
    )

    updated = updater.update(state, prediction)

    expected_transform = se3_exp(delta_xi)

    torch.testing.assert_close(
        updated.transform,
        expected_transform,
    )
    torch.testing.assert_close(
        updated.time_offset,
        delta_tau,
    )


def test_update_is_left_multiplicative() -> None:
    updater = CalibrationUpdater()

    # Current state has a 90-degree rotation around z and translation [1, 2, 3].
    angle = math.pi / 2.0
    rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    transform = torch.eye(4, dtype=torch.float64).unsqueeze(0)
    transform[0, :3, :3] = rotation
    transform[0, :3, 3] = torch.tensor(
        [1.0, 2.0, 3.0],
        dtype=torch.float64,
    )

    state = CalibrationState(
        transform=transform,
        time_offset=torch.zeros(1, 1, dtype=torch.float64),
    )

    # Pure +x translational tangent correction.
    delta_xi = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )

    prediction = _make_prediction(
        delta_xi,
        torch.zeros(1, 1, dtype=torch.float64),
    )

    updated = updater.update(state, prediction)

    expected = se3_exp(delta_xi) @ state.transform

    torch.testing.assert_close(
        updated.transform,
        expected,
    )

    # This explicitly distinguishes left from right multiplication:
    #
    #     Exp(delta_xi) @ T
    #
    # adds [1, 0, 0] on the spatial/output side, giving [2, 2, 3].
    expected_translation = torch.tensor(
        [[2.0, 2.0, 3.0]],
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        updated.transform[:, :3, 3],
        expected_translation,
    )


def test_time_offset_update_is_additive() -> None:
    updater = CalibrationUpdater()

    state = CalibrationState(
        transform=torch.eye(
            4,
            dtype=torch.float64,
        ).unsqueeze(0).repeat(2, 1, 1),
        time_offset=torch.tensor(
            [[0.25], [-0.50]],
            dtype=torch.float64,
        ),
    )

    prediction = _make_prediction(
        delta_xi=torch.zeros(2, 6, dtype=torch.float64),
        delta_tau=torch.tensor(
            [[0.10], [0.20]],
            dtype=torch.float64,
        ),
    )

    updated = updater.update(state, prediction)

    expected_time_offset = torch.tensor(
        [[0.35], [-0.30]],
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        updated.time_offset,
        expected_time_offset,
    )


def test_change_outputs_do_not_gate_calibration_update() -> None:
    updater = CalibrationUpdater()
    state = _identity_state(batch_size=2)

    delta_xi = torch.tensor(
        [
            [0.1, 0.0, 0.0, 0.2, 0.0, 0.0],
            [0.0, -0.1, 0.0, 0.0, 0.3, 0.0],
        ],
        dtype=torch.float64,
    )
    delta_tau = torch.tensor(
        [[0.05], [-0.02]],
        dtype=torch.float64,
    )

    prediction_a = _make_prediction(
        delta_xi,
        delta_tau,
        change_event_logit=torch.tensor(
            [[-100.0], [100.0]],
            dtype=torch.float64,
        ),
        change_time=torch.tensor(
            [[-10.0], [500.0]],
            dtype=torch.float64,
        ),
    )

    prediction_b = _make_prediction(
        delta_xi,
        delta_tau,
        change_event_logit=torch.tensor(
            [[0.0], [0.0]],
            dtype=torch.float64,
        ),
        change_time=torch.tensor(
            [[0.0], [0.0]],
            dtype=torch.float64,
        ),
    )

    updated_a = updater.update(state, prediction_a)
    updated_b = updater.update(state, prediction_b)

    # change_event_logit and change_time are auxiliary supervised outputs only.
    torch.testing.assert_close(
        updated_a.transform,
        updated_b.transform,
    )
    torch.testing.assert_close(
        updated_a.time_offset,
        updated_b.time_offset,
    )


def test_calibration_update_remains_differentiable() -> None:
    updater = CalibrationUpdater()
    state = _identity_state(batch_size=4)

    delta_xi = (
        0.1 * torch.randn(
            4,
            6,
            dtype=torch.float64,
        )
    ).requires_grad_()

    delta_tau = (
        0.1 * torch.randn(
            4,
            1,
            dtype=torch.float64,
        )
    ).requires_grad_()

    prediction = _make_prediction(
        delta_xi,
        delta_tau,
    )

    updated = updater.update(state, prediction)

    # Differentiability is retained for possible future predicted-rollout or
    # consistency losses, even though normal teacher-forced training need not
    # backpropagate through the carried calibration update.
    loss = (
        updated.transform[..., 0, 1].sum()
        + updated.transform[..., :3, 3].sum()
        + updated.time_offset.sum()
    )
    loss.backward()

    assert delta_xi.grad is not None
    assert delta_tau.grad is not None

    assert torch.isfinite(delta_xi.grad).all()
    assert torch.isfinite(delta_tau.grad).all()


def test_prediction_batch_size_must_match_state() -> None:
    updater = CalibrationUpdater()
    state = _identity_state(batch_size=2)

    prediction = _make_prediction(
        delta_xi=torch.zeros(3, 6, dtype=torch.float64),
        delta_tau=torch.zeros(3, 1, dtype=torch.float64),
    )

    with pytest.raises(
        ValueError,
        match="share batch size",
    ):
        updater.update(state, prediction)