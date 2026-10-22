"""Analytic SE(3) x R calibration-correction paths and velocity targets.

These are corrections from the identity, not absolute sensor extrinsics.  The
canonical obscalib convention is [rotation, translation] and left application
of the final correction to the old extrinsic.
"""

import math

import pytest
import torch

from obscalib.flow_matching.calibration_path import make_calibration_flow_batch, sample_flow_times
from obscalib.geometry.lie import se3_exp, se3_log


def _targets(batch_size=4, num_calibrations=3, *, dtype=torch.float64):
    torch.manual_seed(41)
    delta_xi = torch.randn(batch_size, num_calibrations, 6, dtype=dtype) * 0.13
    delta_tau = torch.randn(batch_size, num_calibrations, 1, dtype=dtype) * 0.012
    return delta_xi, delta_tau


def test_calibration_path_exact_exponential_and_constant_velocity():
    delta_xi, delta_tau = _targets()
    t = torch.linspace(0, 1, delta_xi.shape[0], dtype=delta_xi.dtype)[:, None, None].expand_as(delta_tau).clone()
    batch = make_calibration_flow_batch(delta_xi, delta_tau, time=t)

    assert batch.correction_xi.shape == (4, 3, 6)
    assert batch.correction_tau.shape == (4, 3, 1)
    assert batch.correction_transform.shape == (4, 3, 4, 4)
    assert batch.time.shape == (4, 3, 1)
    assert batch.target_velocity.shape == (4, 3, 7)
    torch.testing.assert_close(batch.correction_xi, t * delta_xi)
    torch.testing.assert_close(batch.correction_tau, t * delta_tau)
    torch.testing.assert_close(batch.target_velocity, torch.cat((delta_xi, delta_tau), -1))
    torch.testing.assert_close(batch.correction_transform, se3_exp((t * delta_xi).reshape(-1, 6)).reshape(4, 3, 4, 4), atol=1e-9, rtol=1e-9)

    # The conditional field along D_t=Exp(t*delta) is constant in the left-trivialized frame.
    t_mid = torch.full_like(delta_tau, 0.6)
    mid = make_calibration_flow_batch(delta_xi, delta_tau, time=t_mid)
    end = se3_exp(delta_xi.reshape(-1, 6)).reshape(4, 3, 4, 4)
    remaining = torch.linalg.solve(mid.correction_transform.reshape(-1, 4, 4), end.reshape(-1, 4, 4))
    conditional_velocity = se3_log(remaining).reshape(4, 3, 6) / (1 - t_mid)
    torch.testing.assert_close(conditional_velocity, delta_xi, atol=2e-6, rtol=2e-6)


def test_identity_start_endpoint_and_left_applied_update():
    delta_xi, delta_tau = _targets(batch_size=2, num_calibrations=2)
    zeros = torch.zeros_like(delta_tau)
    ones = torch.ones_like(delta_tau)
    beginning = make_calibration_flow_batch(delta_xi, delta_tau, time=zeros)
    ending = make_calibration_flow_batch(delta_xi, delta_tau, time=ones)

    identity = torch.eye(4, dtype=delta_xi.dtype).expand(2, 2, 4, 4)
    torch.testing.assert_close(beginning.correction_transform, identity, atol=1e-10, rtol=0)
    torch.testing.assert_close(beginning.correction_xi, torch.zeros_like(delta_xi))
    torch.testing.assert_close(beginning.correction_tau, zeros)
    torch.testing.assert_close(ending.correction_tau, delta_tau)
    torch.testing.assert_close(ending.correction_transform, se3_exp(delta_xi.reshape(-1, 6)).reshape(2, 2, 4, 4))

    current = se3_exp(torch.randn(4, 6, dtype=delta_xi.dtype) * 0.2).reshape(2, 2, 4, 4)
    truth = ending.correction_transform @ current
    recovered = se3_log((truth @ torch.linalg.inv(current)).reshape(-1, 4, 4)).reshape(2, 2, 6)
    torch.testing.assert_close(recovered, delta_xi, atol=2e-6, rtol=2e-6)


def test_sampled_times_are_reproducible_and_can_be_shared_across_calibrations():
    xi, tau = _targets(batch_size=6, num_calibrations=3, dtype=torch.float32)
    t1 = sample_flow_times(xi, generator=torch.Generator().manual_seed(17), shared_across_calibrations=True)
    t2 = sample_flow_times(xi, generator=torch.Generator().manual_seed(17), shared_across_calibrations=True)
    assert t1.shape == tau.shape
    torch.testing.assert_close(t1, t2)
    torch.testing.assert_close(t1[:, :1], t1[:, 2:])
    assert bool(((t1 >= 0) & (t1 < 1)).all())

    per_calibration = sample_flow_times(xi, generator=torch.Generator().manual_seed(17))
    assert not torch.equal(per_calibration[:, 0], per_calibration[:, 1])


def test_path_is_supervision_only_and_does_not_differentiate_through_geometry():
    xi, tau = _targets()
    xi.requires_grad_()
    tau.requires_grad_()
    output = make_calibration_flow_batch(xi, tau)
    assert not output.target_velocity.requires_grad
    assert not output.correction_transform.requires_grad
    assert not output.correction_xi.requires_grad


@pytest.mark.parametrize("bad_time", [-0.1, 1.01, float("nan"), float("inf")])
def test_reject_invalid_times(bad_time):
    xi, tau = _targets()
    with pytest.raises(ValueError, match="time"):
        make_calibration_flow_batch(xi, tau, time=torch.full_like(tau, bad_time))


def test_reject_incompatible_shapes_and_dtypes():
    xi, tau = _targets()
    with pytest.raises(ValueError, match="target_delta_xi"):
        make_calibration_flow_batch(xi[..., :5], tau)
    with pytest.raises(ValueError, match="target_delta_tau"):
        make_calibration_flow_batch(xi, tau[..., 0])
    with pytest.raises(ValueError, match="dtype"):
        make_calibration_flow_batch(xi.float(), tau)
    with pytest.raises(ValueError, match="time"):
        make_calibration_flow_batch(xi, tau, time=torch.rand(4, 1, dtype=xi.dtype))
