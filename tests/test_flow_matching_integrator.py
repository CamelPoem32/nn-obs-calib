"""ODE integration and FM velocity-loss tests with the canonical SE(3) API."""

import math

import pytest
import torch

from obscalib.flow_matching.lie_integrator import integrate_calibration_flow
from obscalib.flow_matching.loss import CalibrationFlowLossWeights, calibration_velocity_loss
from obscalib.geometry.lie import se3_exp, se3_log


@pytest.mark.parametrize("method", ("euler", "midpoint"))
@pytest.mark.parametrize("steps", (1, 2, 4, 8, 16))
def test_constant_left_trivialized_velocity_is_exact(method, steps):
    reference = torch.zeros(3, 2, 8, dtype=torch.float64)
    target = torch.tensor([0.3, -0.1, 0.2, 0.12, -0.05, 0.08, 0.03], dtype=reference.dtype).expand(3, 2, 7)
    calls = []

    def field(current_xi, current_tau, time):
        calls.append(float(time.reshape(-1)[0]))
        assert current_xi.shape == (3, 2, 6) and current_tau.shape == time.shape == (3, 2, 1)
        return target

    result = integrate_calibration_flow(field, reference, num_steps=steps, method=method, record_trajectory=True)
    expected = se3_exp(target[..., :6].reshape(-1, 6)).reshape(3, 2, 4, 4)
    torch.testing.assert_close(result.correction_transform, expected, atol=2e-7, rtol=2e-7)
    torch.testing.assert_close(result.delta_tau, target[..., 6:], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(result.delta_xi, target[..., :6], atol=3e-6, rtol=3e-6)
    assert result.trajectory is not None and len(result.trajectory) == steps + 1
    assert result.trajectory[0].time == 0.0 and result.trajectory[-1].time == 1.0
    assert len(calls) == steps * (1 if method == "euler" else 2)


def test_zero_velocity_keeps_identity_and_zero_offset():
    reference = torch.ones(2, 3, 16)

    def zero_field(xi, tau, time):
        return xi.new_zeros(*xi.shape[:-1], 7)

    result = integrate_calibration_flow(zero_field, reference, num_steps=4, method="midpoint")
    torch.testing.assert_close(result.correction_transform, torch.eye(4).expand(2, 3, 4, 4))
    torch.testing.assert_close(result.delta_xi, torch.zeros(2, 3, 6), atol=1e-6, rtol=0)
    torch.testing.assert_close(result.delta_tau, torch.zeros(2, 3, 1))
    assert result.trajectory is None


def test_midpoint_integrates_linearly_time_varying_velocity_more_accurately():
    reference = torch.zeros(2, 1, 4, dtype=torch.float64)
    direction = torch.tensor([0, 0, 0.45, 0, 0, 0, 0.02], dtype=reference.dtype)

    def field(xi, tau, time):
        return 2 * time * direction

    euler = integrate_calibration_flow(field, reference, num_steps=4, method="euler")
    midpoint = integrate_calibration_flow(field, reference, num_steps=4, method="midpoint")
    expected = se3_exp(direction[:6].expand(2, 1, 6).reshape(-1, 6)).reshape(2, 1, 4, 4)
    euler_error = (euler.correction_transform - expected).abs().max().item()
    midpoint_error = (midpoint.correction_transform - expected).abs().max().item()
    assert midpoint_error < 1e-7 and midpoint_error < euler_error / 100
    torch.testing.assert_close(midpoint.delta_tau, torch.full((2, 1, 1), 0.02, dtype=reference.dtype), atol=1e-10, rtol=0)


def test_noncommuting_time_varying_field_converges_with_step_count():
    reference = torch.zeros(1, 3, dtype=torch.float64)

    def field(xi, tau, time):
        omega = torch.cat((0.3 * torch.sin(2 * math.pi * time), 0.3 * torch.cos(2 * math.pi * time), torch.zeros_like(time)), dim=-1)
        return torch.cat((omega, torch.zeros_like(omega), torch.zeros_like(time)), dim=-1)

    # A very fine midpoint trajectory supplies a numerical reference; no true endpoint is assumed.
    reference_solution = integrate_calibration_flow(field, reference, num_steps=200, method="midpoint")
    coarse = integrate_calibration_flow(field, reference, num_steps=4, method="midpoint")
    fine = integrate_calibration_flow(field, reference, num_steps=32, method="midpoint")
    coarse_error = (coarse.correction_transform - reference_solution.correction_transform).abs().max()
    fine_error = (fine.correction_transform - reference_solution.correction_transform).abs().max()
    assert fine_error < coarse_error


@pytest.mark.parametrize("steps, method", ((0, "midpoint"), (True, "midpoint"), (2, "rk4")))
def test_invalid_integration_settings_raise(steps, method):
    with pytest.raises(ValueError):
        integrate_calibration_flow(lambda xi, tau, t: xi.new_zeros(*xi.shape[:-1], 7), torch.zeros(2, 4), num_steps=steps, method=method)


def test_velocity_shape_and_dtype_validation():
    reference = torch.zeros(2, 4)
    with pytest.raises(ValueError, match="shape"):
        integrate_calibration_flow(lambda xi, tau, t: xi.new_zeros(2, 6), reference)
    with pytest.raises(ValueError, match="dtype"):
        integrate_calibration_flow(lambda xi, tau, t: xi.new_zeros(2, 7).double(), reference)


def test_weighted_velocity_loss_is_differentiable_and_matches_manual_components():
    target = torch.zeros(2, 3, 7, dtype=torch.float64)
    prediction = torch.ones_like(target).requires_grad_()
    weights = CalibrationFlowLossWeights(rotation_scale_rad=2, translation_scale_m=4, time_offset_scale_s=5, rotation_weight=2, translation_weight=3, time_offset_weight=4)
    output = calibration_velocity_loss(prediction, target, weights=weights)
    expected = 2 / 4 + 3 / 16 + 4 / 25
    torch.testing.assert_close(output.total, torch.tensor(expected, dtype=prediction.dtype))
    assert output.num_active_rotations == 0 and output.active_rotation.item() == 0
    output.total.backward()
    assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum() > 0
    assert "loss_fm_total" in output.detached_metrics()


def test_active_rotation_loss_masks_and_zero_active_case():
    target = torch.zeros(2, 7, dtype=torch.float64)
    target[0, 0] = 0.3
    prediction = torch.zeros_like(target, requires_grad=True)
    weights = CalibrationFlowLossWeights(active_rotation_weight=0.75)
    output = calibration_velocity_loss(prediction, target, weights=weights)
    assert output.num_active_rotations == 1
    assert output.active_rotation.item() > 0
    output.total.backward()
    assert torch.isfinite(prediction.grad).all()

    empty = calibration_velocity_loss(torch.zeros(2, 7), torch.zeros(2, 7), weights=weights)
    assert empty.num_active_rotations == 0
    assert empty.total.item() == 0

    with pytest.raises(ValueError, match="active_rotation_mask"):
        calibration_velocity_loss(torch.zeros(2, 7), torch.zeros(2, 7), active_rotation_mask=torch.ones(2))


@pytest.mark.parametrize("value", (0.0, -1.0, float("nan"), float("inf")))
def test_invalid_loss_scales(value):
    with pytest.raises(ValueError):
        CalibrationFlowLossWeights(rotation_scale_rad=value)


@pytest.mark.parametrize("angle_degrees", (60.0, 170.0))
def test_large_rotation_correction_remains_valid_on_se3(angle_degrees):
    """Test the group element rather than comparing ambiguous 180-degree log branches."""
    angle = math.radians(angle_degrees)
    reference = torch.zeros(2, 12, dtype=torch.float64)
    direction = reference.new_tensor([0.0, angle, 0.0, 0.03, -0.02, 0.08, -0.015]).expand(2, 7)

    result = integrate_calibration_flow(lambda xi, tau, t: direction, reference, num_steps=16, method="midpoint")
    expected = se3_exp(direction[:, :6])
    torch.testing.assert_close(result.correction_transform, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(se3_exp(result.delta_xi), expected, atol=2e-6, rtol=2e-6)
    assert torch.isfinite(result.delta_xi).all()
