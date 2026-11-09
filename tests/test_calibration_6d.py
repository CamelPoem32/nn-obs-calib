"""Consolidated geometry, prediction, updates, loss and checkpoint tests for 6D regression."""

from __future__ import annotations

import math

import pytest
import torch

from obscalib.calibration.rotation_6d_update import apply_calibration_6d_update
from obscalib.calibration.state import CalibrationState
from obscalib.config import CalibrationPoolerConfig, MLPConfig, PooledCalibrationHeadConfig, PooledCalibrationModelConfig, TransformerConfig
from obscalib.data.structures import CalibrationTargetBatch, TokenBatch
from obscalib.geometry.rotation_6d import matrix_to_rotation_6d, rotation_6d_to_matrix
from obscalib.models.calibration_6d_head import Calibration6DHead, Calibration6DPrediction
from obscalib.models.pooled_calibration_6d_model import PooledCalibration6DModel, load_6d_pretrained_features
from obscalib.models.pooled_calibration_model import PooledCalibrationModel
from obscalib.training.rotation_6d_loss import Calibration6DLossWeights, calibration_6d_loss


def _rotation(axis: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Use a matrix exponential in tests to avoid depending on 6D reconstruction."""
    axis = axis / axis.norm()
    x, y, z = axis.unbind()
    skew = torch.stack((torch.stack((x * 0, -z, y)), torch.stack((z, x * 0, -x)), torch.stack((-y, x, x * 0))))
    return torch.matrix_exp(skew * math.radians(angle_deg))


def _transform(rotation: torch.Tensor, translation: torch.Tensor | None = None) -> torch.Tensor:
    T = torch.eye(4, device=rotation.device, dtype=rotation.dtype).expand(rotation.shape[:-2] + (4, 4)).clone()
    T[..., :3, :3] = rotation
    if translation is not None:
        T[..., :3, 3] = translation
    return T


def _identity_state(batch_size: int, *, dtype: torch.dtype = torch.float64) -> CalibrationState:
    return CalibrationState(transform=torch.eye(4, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1), time_offset=torch.zeros(batch_size, 1, dtype=dtype))


def _target(T: torch.Tensor, tau: torch.Tensor) -> CalibrationTargetBatch:
    return CalibrationTargetBatch(next_transform=T, next_time_offset=tau, change_label=None, change_time=None)


def _config(context_dim: int = 0) -> PooledCalibrationModelConfig:
    return PooledCalibrationModelConfig(
        transformer=TransformerConfig(input_dim=8, d_model=32, n_heads=4, num_layers=2, dim_feedforward=64, dropout=0.0, num_summary_tokens=1, activation="gelu"),
        pooler=CalibrationPoolerConfig(n_heads=4, dropout=0.0, calibration_context_dim=context_dim),
        mlp=MLPConfig(hidden_dims=(32,), output_dim=24, activation="gelu", dropout=0.0),
        head=PooledCalibrationHeadConfig(calibration_context_dim=context_dim, hidden_dims=(32, 16), activation="gelu", dropout=0.0),
    )


def _tokens() -> TokenBatch:
    batch_size, n = 2, 10
    return TokenBatch(x=torch.randn(batch_size, n, 8), token_mask=torch.ones(batch_size, n, dtype=torch.bool), calibration_ids=torch.tensor([[0] * 5 + [1] * 5] * batch_size), calibration_keys=("imu", "lidar"))


@pytest.mark.parametrize("angle", [0.0, 15.0, 90.0, 150.0, 175.0, 179.0, 180.0])
def test_6d_matrix_roundtrip(angle: float) -> None:
    R = _rotation(torch.tensor([0.3, -0.7, 1.1], dtype=torch.float64), angle)
    restored = rotation_6d_to_matrix(matrix_to_rotation_6d(R))
    torch.testing.assert_close(restored, R, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(restored.T @ restored, torch.eye(3, dtype=R.dtype), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(torch.det(restored), R.new_tensor(1.0), rtol=1e-10, atol=1e-10)


def test_6d_rotation_targets_are_continuous_across_180_degrees() -> None:
    axis = torch.tensor([0.4, 0.2, -0.6], dtype=torch.float64)
    left = matrix_to_rotation_6d(_rotation(axis, 179.99))
    right = matrix_to_rotation_6d(_rotation(axis, 180.01))
    assert torch.linalg.vector_norm(left - right) < 0.001


def test_decoder_batches_and_finite_gradients_near_cut_locus() -> None:
    R = torch.stack((_rotation(torch.tensor([1., 0.2, -0.3], dtype=torch.float64), 179.9), _rotation(torch.tensor([0.1, 1., 0.2], dtype=torch.float64), 179.99)))
    six = matrix_to_rotation_6d(R).detach().clone().requires_grad_()
    reconstructed = rotation_6d_to_matrix(six)
    reconstructed.square().sum().backward()
    assert six.grad is not None and torch.isfinite(six.grad).all()
    assert reconstructed.shape == (2, 3, 3)


def test_decoder_fallback_for_degenerate_inputs() -> None:
    six = torch.tensor([[0., 0., 0., 0., 0., 0.], [1., 0., 0., 3., 0., 0.]])
    matrices = rotation_6d_to_matrix(six)
    eye = torch.eye(3).expand(2, -1, -1)
    torch.testing.assert_close(matrices.transpose(-1, -2) @ matrices, eye, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(torch.det(matrices), torch.ones(2), rtol=1e-6, atol=1e-6)


def test_geometry_invalid_shapes() -> None:
    with pytest.raises(ValueError):
        rotation_6d_to_matrix(torch.zeros(2, 5))
    with pytest.raises(ValueError):
        matrix_to_rotation_6d(torch.zeros(2, 4, 4))


def test_head_identity_initialization_and_context() -> None:
    config = _config(context_dim=3).head
    head = Calibration6DHead(config, shared_feature_dim=24)
    features = torch.zeros(5, 24)
    output = head(features, torch.zeros(5, 3))
    torch.testing.assert_close(output.rotation_6d, torch.tensor([[1., 0., 0., 0., 1., 0.]]).expand(5, -1), atol=2e-3, rtol=0)
    assert output.delta_translation.shape == (5, 3) and output.delta_tau.shape == (5, 1)
    with pytest.raises(ValueError):
        head(features)


def test_full_rotation_translation_and_time_update_and_healthy_isolation() -> None:
    state = _identity_state(2)
    R_delta = _rotation(torch.tensor([0., 0., 1.], dtype=torch.float64), 179.0).expand(2, -1, -1)
    six = matrix_to_rotation_6d(R_delta).detach().clone().requires_grad_()
    pred = Calibration6DPrediction(rotation_6d=six, delta_translation=torch.tensor([[0.2, 0.3, -0.5]] * 2, dtype=torch.float64), delta_tau=torch.full((2, 1), 0.04, dtype=torch.float64))
    states = {"lidar": state, "imu": _identity_state(2)}
    predictions = {"lidar": pred, "imu": pred}
    updated = apply_calibration_6d_update(states, predictions, keys_to_update=("lidar",))
    torch.testing.assert_close(updated["lidar"].transform[:, :3, :3], R_delta, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(updated["lidar"].transform[:, :3, 3], pred.delta_translation)
    torch.testing.assert_close(updated["lidar"].time_offset, pred.delta_tau)
    assert updated["imu"] is states["imu"]
    updated["lidar"].transform[:, 0, 0].sum().backward()
    assert six.grad is not None and torch.isfinite(six.grad).all()


def test_calibration_loss_zero_at_exact_ground_truth_and_nonzero_at_identity() -> None:
    state = _identity_state(3)
    R = _rotation(torch.tensor([0.7, -0.4, 0.2], dtype=torch.float64), 179.0).expand(3, -1, -1)
    dt = torch.tensor([[0.1, 0.2, -0.1]] * 3, dtype=torch.float64)
    tau = torch.full((3, 1), 0.025, dtype=torch.float64)
    target = {"lidar": _target(_transform(R, dt), tau)}
    prior = {"lidar": state}
    correct = Calibration6DPrediction(rotation_6d=matrix_to_rotation_6d(R), delta_translation=dt, delta_tau=tau)
    exact = calibration_6d_loss({"lidar": correct}, prior, target)
    assert float(exact.total) < 1e-15

    six = matrix_to_rotation_6d(torch.eye(3, dtype=R.dtype)).expand(3, -1).clone().requires_grad_()
    wrong = Calibration6DPrediction(rotation_6d=six, delta_translation=torch.zeros_like(dt, requires_grad=True), delta_tau=torch.zeros_like(tau, requires_grad=True))
    loss = calibration_6d_loss({"lidar": wrong}, prior, target, weights=Calibration6DLossWeights())
    assert float(loss.total.detach()) > 0
    loss.total.backward()
    assert six.grad is not None and torch.isfinite(six.grad).all()
    assert wrong.delta_translation.grad is not None and torch.isfinite(wrong.delta_translation.grad).all()
    assert wrong.delta_tau.grad is not None and torch.isfinite(wrong.delta_tau.grad).all()


def test_model_pooled_forward_and_backpropagation() -> None:
    torch.manual_seed(10)
    model = PooledCalibration6DModel(_config())
    tokens = _tokens()
    output = model(tokens)
    assert set(output.predictions) == {"imu", "lidar"}
    assert output.shared_features.shape == (2, 2, 24)
    for pred in output.predictions.values():
        assert pred.rotation_6d.shape == (2, 6)
        assert pred.delta_rotation.shape == (2, 3, 3)
        assert pred.delta_translation.shape == (2, 3)
        assert pred.delta_tau.shape == (2, 1)
    objective = sum((pred.rotation_6d.square().mean() + pred.delta_translation.square().mean() + pred.delta_tau.square().mean()) for pred in output.predictions.values())
    objective.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.head_6d.parameters())
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.transformer_encoder.parameters())


def test_pretrained_regression_or_fm_features_transfer_without_old_head() -> None:
    config = _config()
    source = PooledCalibrationModel(config)
    target = PooledCalibration6DModel(config)
    before_head = {key: value.detach().clone() for key, value in target.head_6d.state_dict().items()}
    target.freeze_feature_extractor(True)
    assert not any(p.requires_grad for p in target.transformer_encoder.parameters())
    assert all(p.requires_grad for p in target.head_6d.parameters())
    target.freeze_feature_extractor(False)
    loaded = load_6d_pretrained_features(target, {"model_state_dict": source.state_dict()})
    assert loaded
    for prefix in ("transformer_encoder", "pooler", "mlp_body"):
        source_state = getattr(source, prefix).state_dict()
        target_state = getattr(target, prefix).state_dict()
        for key in source_state:
            torch.testing.assert_close(target_state[key], source_state[key])
    for key, previous in before_head.items():
        torch.testing.assert_close(target.head_6d.state_dict()[key], previous)
    with pytest.raises(KeyError):
        load_6d_pretrained_features(target, {"model_state_dict": {"transformer_encoder.nonexistent": torch.zeros(1)}}, strict=True)


def test_head_can_overfit_one_fixed_179_degree_correction() -> None:
    """Geometry/head smoke test, not a claim about sensor-data generalization."""
    torch.manual_seed(2026)
    target_R = _rotation(torch.tensor([1., 2., 3.]), 179.0)
    target_6d = matrix_to_rotation_6d(target_R)
    head = Calibration6DHead(_config().head, shared_feature_dim=24)
    features = torch.randn(1, 24)
    optimizer = torch.optim.Adam(head.parameters(), lr=5e-3)
    for _ in range(100):
        prediction = head(features)
        loss = (prediction.rotation_6d - target_6d).square().mean() + 0.5 * (prediction.delta_rotation - target_R).square().sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    rotation_error = prediction.delta_rotation.detach()[0] @ target_R.T
    angle_deg = torch.rad2deg(torch.acos(((rotation_error.trace() - 1.0) / 2.0).clamp(-1.0, 1.0)))
    assert angle_deg < 2.0
