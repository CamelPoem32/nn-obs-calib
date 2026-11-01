"""Property tests for SE(3) correction-to-go supervision.

Place under tests/ and run: pytest -q tests/test_correction_to_go_path.py
"""

import math

import pytest
import torch

from obscalib.flow_matching.calibration_path import make_calibration_flow_batch, make_correction_to_go_flow_batch
from obscalib.geometry.lie import se3_exp


def _truth(dtype=torch.float64):
    xi = torch.tensor([[[0.12, -0.18, 0.10, 0.03, 0.02, -0.01],
                        [-0.55, 0.15, 0.25, -0.04, 0.02, 0.03]]], dtype=dtype)
    tau = torch.tensor([[[0.15], [-0.08]]], dtype=dtype)
    return xi, tau


def _check_endpoint(batch, xi, tau, atol=1e-7):
    recovered = batch.correction_transform @ se3_exp(((1.0 - batch.time) * batch.target_velocity[..., :6]).reshape(-1, 6)).reshape(*xi.shape[:-1], 4, 4)
    expected = se3_exp(xi.reshape(-1, 6)).reshape(*xi.shape[:-1], 4, 4)
    torch.testing.assert_close(recovered, expected, atol=atol, rtol=atol)
    torch.testing.assert_close(batch.correction_tau + (1.0 - batch.time) * batch.target_velocity[..., 6:], tau, atol=atol, rtol=atol)


def test_on_path_matches_original_constant_velocity():
    xi, tau = _truth()
    t = tau.new_full(tau.shape, 0.4)
    batch = make_correction_to_go_flow_batch(xi, tau, time=t, mode="on_path")
    torch.testing.assert_close(batch.target_velocity, torch.cat((xi, tau), dim=-1), atol=1e-7, rtol=1e-7)
    _check_endpoint(batch, xi, tau)
    old = make_calibration_flow_batch(xi, tau, time=t)
    torch.testing.assert_close(batch.correction_transform, old.correction_transform, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("mode", ["off_path", "identity", "mixed"])
def test_oracle_exactly_recovers_endpoint(mode):
    xi, tau = _truth()
    t = tau.new_full(tau.shape, 0.5)
    batch = make_correction_to_go_flow_batch(xi, tau, time=t, mode=mode,
        off_path_probability=0.6, identity_probability=0.2, rotation_noise_std_rad=0.1,
        translation_noise_std_m=0.02, tau_noise_std_s=0.025,
        generator=torch.Generator().manual_seed(1234))
    _check_endpoint(batch, xi, tau, atol=2e-6)
    if mode == "identity":
        assert bool((batch.time == 0).all())
        assert bool((batch.correction_tau == 0).all())
        torch.testing.assert_close(batch.target_velocity, torch.cat((xi, tau), dim=-1), atol=1e-7, rtol=1e-7)


def test_healthy_zero_target_and_off_path_recovery():
    xi, tau = _truth()
    xi.zero_()
    tau.zero_()
    batch = make_correction_to_go_flow_batch(xi, tau, time=tau.new_full(tau.shape, 0.7), mode="off_path",
        generator=torch.Generator().manual_seed(17))
    _check_endpoint(batch, xi, tau, atol=2e-6)


def test_time_grid_and_mode_broadcast_across_sensor_keys():
    xi, tau = _truth()
    xi, tau = xi.expand(7, 5, -1, -1), tau.expand(7, 5, -1, -1)
    t = tau.new_full(tau.shape, 0.45)
    out = make_correction_to_go_flow_batch(xi, tau, time=t, mode="mixed",
        off_path_probability=0.6, identity_probability=0.2,
        generator=torch.Generator().manual_seed(32))
    assert out.target_velocity.shape == (7, 5, 2, 7)
    assert out.state_mode.shape == t.shape
    assert bool((out.state_mode[..., 0, :] == out.state_mode[..., 1, :]).all())
    _check_endpoint(out, xi, tau, atol=3e-6)


def test_invalid_times_and_probabilities_raise():
    xi, tau = _truth()
    for t in (-0.01, 0.95, 1.0):
        with pytest.raises(ValueError):
            make_correction_to_go_flow_batch(xi, tau, time=tau.new_full(tau.shape, t), max_time=0.875)
    with pytest.raises(ValueError):
        make_correction_to_go_flow_batch(xi, tau, time=tau.new_full(tau.shape, 0.5), off_path_probability=0.8, identity_probability=0.3)
