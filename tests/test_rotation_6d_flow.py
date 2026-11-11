"""Consolidated tests of 6D FM geometry, CTG paths, integration and model interface."""

from __future__ import annotations

import math

import pytest
import torch

from obscalib.calibration.state import CalibrationState
from obscalib.config import CalibrationPoolerConfig, MLPConfig, PooledCalibrationHeadConfig, PooledCalibrationModelConfig, TransformerConfig
from obscalib.data.structures import CalibrationTargetBatch, TokenBatch
from obscalib.flow_matching.rotation_6d_path import identity_6d_flow_state, make_rotation_6d_flow_batch, stack_rotation_6d_flow_target
from obscalib.flow_matching.rotation_6d_integrator import integrate_rotation_6d_flow
from obscalib.geometry.rotation_6d import matrix_to_rotation_6d, rotation_6d_to_matrix
from obscalib.models.pooled_calibration_6d_model import PooledCalibration6DModel
from obscalib.models.pooled_calibration_flow_6d_model import PooledCalibrationFlow6DModel
from obscalib.training.rotation_6d_flow_loss import Rotation6DFlowLossWeights, rotation_6d_flow_loss, rotation_6d_targets_from_calibration


def _rotation(angle: float, axis=(0.38, -0.48, 0.75), dtype=torch.float64):
    axis = torch.tensor(axis, dtype=dtype)
    axis = axis / axis.norm()
    x, y, z = axis.unbind()
    K = torch.stack((torch.stack((x * 0., -z, y)), torch.stack((z, x * 0., -x)), torch.stack((-y, x, x * 0.))))
    return torch.matrix_exp(K * math.radians(angle))


def _config():
    return PooledCalibrationModelConfig(
        transformer=TransformerConfig(input_dim=8, d_model=32, n_heads=4, num_layers=2, dim_feedforward=64, dropout=0., num_summary_tokens=1, activation='gelu'),
        pooler=CalibrationPoolerConfig(n_heads=4, dropout=0., calibration_context_dim=0),
        mlp=MLPConfig(hidden_dims=(32,), output_dim=24, activation='gelu', dropout=0.),
        head=PooledCalibrationHeadConfig(calibration_context_dim=0, hidden_dims=(32, 16), activation='gelu', dropout=0.),
    )


def _tokens():
    return TokenBatch(x=torch.randn(2, 10, 8), token_mask=torch.ones(2, 10, dtype=torch.bool), calibration_ids=torch.tensor([[0] * 5 + [1] * 5] * 2), calibration_keys=('imu', 'lidar'))


def _target(angle=179., b=2, axis=(0.38, -0.48, 0.75)):
    R = _rotation(angle, axis=axis).expand(b, -1, -1)
    rotation = matrix_to_rotation_6d(R)
    translation = torch.tensor([[0.02, -0.04, 0.01]], dtype=torch.float64).expand(b, -1)
    tau = torch.full((b, 1), 0.03, dtype=torch.float64)
    return stack_rotation_6d_flow_target(rotation, translation, tau)


@pytest.mark.parametrize('angle', [0., 30., 90., 150., 179., 180.])
def test_6d_target_roundtrip_at_full_range(angle):
    target = _target(angle)
    torch.testing.assert_close(rotation_6d_to_matrix(target[:, :6]), _rotation(angle).expand(2, -1, -1), atol=1e-10, rtol=1e-10)


def test_identity_is_no_correction_for_every_calibration():
    reference = torch.randn(3, 2, 5)
    initial = identity_6d_flow_state(reference)
    assert initial.shape == (3, 2, 10)
    torch.testing.assert_close(rotation_6d_to_matrix(initial[..., :6]), torch.eye(3).expand(3, 2, 3, 3))
    assert initial[..., 6:].abs().max() == 0


def test_analytic_path_at_zero_and_near_one():
    target = _target().unsqueeze(1).expand(2, 3, -1).clone()
    time = torch.tensor([[[0.0]] * 3, [[0.949]] * 3], dtype=torch.float64)
    batch = make_rotation_6d_flow_batch(target, time=time, mode='on_path', max_time=.95)
    initial = identity_6d_flow_state(target)
    torch.testing.assert_close(batch.state, (1.0 - time) * initial + time * target)
    torch.testing.assert_close(batch.target_velocity, target - initial)
    assert batch.state.shape == (2, 3, 10)


def test_offpath_velocity_is_exact_correction_to_go():
    target = _target(179.).unsqueeze(1).expand(2, 3, -1).clone()
    time = torch.full((2, 3, 1), 0.7, dtype=torch.float64)
    g = torch.Generator().manual_seed(123)
    batch = make_rotation_6d_flow_batch(target, time=time, mode='off_path', max_time=.95, generator=g)
    torch.testing.assert_close(batch.state + (1.0 - time) * batch.target_velocity, target, atol=1e-12, rtol=1e-12)
    assert torch.all(batch.state_mode == 1)


def test_identity_mode_and_shared_randomness():
    target = _target().unsqueeze(1).expand(2, 4, -1).clone()
    a = make_rotation_6d_flow_batch(target, mode='identity')
    assert torch.all(a.time == 0)
    assert torch.all(a.state_mode == 2)
    torch.testing.assert_close(a.state, identity_6d_flow_state(target))
    b = make_rotation_6d_flow_batch(target, generator=torch.Generator().manual_seed(42), shared_across_calibrations=True)
    c = make_rotation_6d_flow_batch(target, generator=torch.Generator().manual_seed(42), shared_across_calibrations=True)
    torch.testing.assert_close(b.time, c.time)
    torch.testing.assert_close(b.state, c.state)
    assert torch.all(b.time[:, 0] == b.time[:, 1])
    assert torch.all(b.state_mode[:, 0] == b.state_mode[:, 1])


def test_bad_path_inputs_rejected():
    with pytest.raises(ValueError):
        make_rotation_6d_flow_batch(torch.zeros(2, 9))
    with pytest.raises(ValueError):
        make_rotation_6d_flow_batch(torch.zeros(2, 10), time=torch.full((2, 1), 1.0))
    with pytest.raises(ValueError):
        stack_rotation_6d_flow_target(torch.zeros(2, 6), torch.zeros(2, 2), torch.zeros(2, 1))


@pytest.mark.parametrize('method', ['euler', 'midpoint'])
@pytest.mark.parametrize('angle', [30., 150., 179., 180.])
def test_integrator_recovers_constant_analytic_velocity_and_endpoint(method, angle):
    target = _target(angle).unsqueeze(1)
    initial = identity_6d_flow_state(target)
    velocity = target - initial
    result = integrate_rotation_6d_flow(lambda state, t: velocity, target, num_steps=8, method=method, record_trajectory=True)
    torch.testing.assert_close(result.state, target, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(result.delta_rotation, _rotation(angle).expand(2, 1, -1, -1), atol=1e-10, rtol=1e-10)
    assert len(result.trajectory) == 9


def test_euclidean_path_is_allowed_to_be_degenerate_at_intermediate_time():
    target = _target(180., axis=(0., 1., 0.), b=1)
    midpoint = make_rotation_6d_flow_batch(target, time=torch.full((1, 1), 0.5, dtype=torch.float64), mode='on_path')
    # 180 around Y sends x -> -x. Its linear column interpolation crosses zero.
    assert midpoint.state[0, :3].norm() < 1e-10
    assert torch.isfinite(midpoint.state).all()
    assert torch.isfinite(midpoint.target_velocity).all()


def test_endpoint_gradients_flow_through_full_integrator():
    target = _target(179., b=1)
    initial = identity_6d_flow_state(target)
    weight = torch.nn.Parameter(torch.zeros(10, dtype=torch.float64))
    optimizer = torch.optim.SGD([weight], lr=1e-2)
    result = integrate_rotation_6d_flow(lambda state, t: (target - initial) + weight, target, num_steps=4, method='midpoint')
    loss = (result.state - target).square().mean()
    optimizer.zero_grad()
    loss.backward()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
    # Test nonzero derivative away from the exact target.
    shifted = integrate_rotation_6d_flow(lambda state, t: (target - initial) + weight + 0.1, target, num_steps=4)
    shifted_loss = (shifted.state - target).square().mean()
    shifted_loss.backward()
    assert weight.grad.norm() > 0


def test_velocity_loss_and_validation():
    target = torch.ones(2, 3, 10)
    zero = rotation_6d_flow_loss(target, target, weights=Rotation6DFlowLossWeights())
    assert float(zero.total) == 0
    pred = torch.nn.Parameter(torch.zeros_like(target))
    loss = rotation_6d_flow_loss(pred, target)
    assert float(loss.total) > 0
    loss.total.backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        rotation_6d_flow_loss(pred, target[..., :7])


def test_model_training_inference_and_backbone_transfer():
    torch.manual_seed(31)
    config = _config()
    source = PooledCalibration6DModel(config)
    model = PooledCalibrationFlow6DModel(config, integration_steps=2)
    model.initialize_backbone_from_regression(source, freeze=False)
    for name in ('transformer_encoder', 'pooler', 'mlp_body'):
        for key, value in getattr(source, name).state_dict().items():
            torch.testing.assert_close(getattr(model, name).state_dict()[key], value)
    tokens = _tokens()
    output = model(tokens, num_steps=2, method='euler')
    assert set(output.predictions) == {'imu', 'lidar'}
    for p in output.predictions.values():
        assert p.rotation_6d.shape == (2, 6) and p.delta_translation.shape == (2, 3) and p.delta_tau.shape == (2, 1)
    target_R = matrix_to_rotation_6d(_rotation(179., dtype=torch.float32).expand(2, -1, -1))
    goal_r = {'imu': target_R, 'lidar': target_R}
    goal_t = {key: torch.zeros(2, 3) for key in goal_r}
    goal_tau = {key: torch.zeros(2, 1) for key in goal_r}
    training = model.forward_flow_matching(tokens, goal_r, goal_t, goal_tau, mode='on_path')
    assert training.predicted_velocity.shape == (2, 2, 10)
    objective = rotation_6d_flow_loss(training.predicted_velocity, training.flow_batch.target_velocity).total
    objective.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.head_6d_flow.parameters())
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.transformer_encoder.parameters())
    model.freeze_feature_extractor(True)
    assert not any(p.requires_grad for p in model.transformer_encoder.parameters())
    assert all(p.requires_grad for p in model.head_6d_flow.parameters())


def test_truth_targets_match_separate_rotation_translation_convention():
    identity = torch.eye(4, dtype=torch.float64).expand(2, -1, -1).clone()
    prior = CalibrationState(transform=identity, time_offset=torch.zeros(2, 1, dtype=torch.float64))
    gt = identity.clone()
    gt[:, :3, :3] = _rotation(179.)
    gt[:, :3, 3] = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64)
    goals = {'lidar': CalibrationTargetBatch(next_transform=gt, next_time_offset=torch.full((2, 1), 0.03, dtype=torch.float64), change_label=None, change_time=None)}
    r, t, tau = rotation_6d_targets_from_calibration({'lidar': prior}, goals)
    torch.testing.assert_close(rotation_6d_to_matrix(r['lidar']), gt[:, :3, :3], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(t['lidar'], gt[:, :3, 3])
    torch.testing.assert_close(tau['lidar'], goals['lidar'].next_time_offset)
