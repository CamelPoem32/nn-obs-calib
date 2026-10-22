"""Configuration, feature-extractor initialization, FM-head training and routing.

These tests use the *real* pooled and flow model classes. For focused model
unit tests, the expensive token encoder is replaced with a deterministic
feature source; existing tokenizer/Transformer tests should exercise raw tokens.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from obscalib.config import (
    CalibrationFlowHeadSettings,
    CalibrationFlowIntegratorConfig,
    CalibrationFlowLossConfig,
    CalibrationPoolerConfig,
    MLPConfig,
    PooledCalibrationFlowConfig,
    PooledCalibrationHeadConfig,
    PooledCalibrationModelConfig,
    TransformerConfig,
)
from obscalib.flow_matching.loss import CalibrationFlowLossWeights, calibration_velocity_loss
from obscalib.models.pooled_calibration_flow_head import PooledCalibrationFlowHead
from obscalib.models.pooled_calibration_flow_model import PooledCalibrationFlowModel
from obscalib.models.pooled_calibration_model import PooledCalibrationModel


@pytest.fixture
def model_config():
    return PooledCalibrationModelConfig(
        transformer=TransformerConfig(input_dim=8, d_model=16, n_heads=4, num_layers=1, dim_feedforward=32, dropout=0.0),
        pooler=CalibrationPoolerConfig(n_heads=4, dropout=0.0, calibration_context_dim=0),
        mlp=MLPConfig(hidden_dims=(24,), output_dim=12, activation="gelu", dropout=0.0),
        head=PooledCalibrationHeadConfig(hidden_dims=(16, 8), activation="gelu", dropout=0.0),
    )


def make_test_inputs(keys, *, batch_size=3, dtype=torch.float32):
    """Keys are intentionally dynamic; no topology order is hard-coded."""
    tokens = SimpleNamespace(calibration_keys=tuple(keys))
    target_xi = {key: torch.tensor([[0.04 * (i + 1), -0.02, 0.08, 0.01, -0.03, 0.02]], dtype=dtype).expand(batch_size, 6).clone() for i, key in enumerate(keys)}
    target_tau = {key: torch.full((batch_size, 1), 0.005 * (i + 1), dtype=dtype) for i, key in enumerate(keys)}
    return tokens, target_xi, target_tau


def install_fake_feature_encoder(monkeypatch, model, *, keys, batch_size=3, context_dim=0):
    """Retain the actual shared MLPBody for parameter-freeze/gradient tests."""
    torch.manual_seed(13)
    pooled_input = torch.randn(batch_size, len(keys), model.config.transformer.d_model)
    context = torch.randn(batch_size, len(keys), context_dim) if context_dim else None

    def encode(tokens, calibration_context=None):
        assert tuple(tokens.calibration_keys) == tuple(keys)
        shared = model.mlp_body(pooled_input.reshape(-1, pooled_input.shape[-1])).reshape(batch_size, len(keys), -1)
        return shared, context

    monkeypatch.setattr(model, "encode_calibration_features", encode)
    return context


def test_flow_configuration_wires_model_and_loss_without_changing_regression_config(model_config):
    flow_config = PooledCalibrationFlowConfig(
        backbone=model_config,
        head=CalibrationFlowHeadSettings(rotation_state_scale_rad=0.7, translation_state_scale_m=0.2),
        integrator=CalibrationFlowIntegratorConfig(num_steps=4, method="euler", ablation_steps=(1, 2, 4, 8)),
        loss=CalibrationFlowLossConfig(active_rotation_weight=0.75),
    )
    model = PooledCalibrationFlowModel(flow_config.backbone, **flow_config.model_kwargs())
    assert model.integration_steps == 4 and model.integration_method == "euler"
    assert model.head.state_scales.tolist()[:3] == pytest.approx([0.7] * 3)
    assert model.head.state_scales.tolist()[3:6] == pytest.approx([0.2] * 3)
    assert flow_config.integrator.num_field_evaluations() == 4
    assert CalibrationFlowIntegratorConfig(num_steps=4, method="midpoint").num_field_evaluations() == 8
    assert isinstance(flow_config.loss.to_loss_weights(), CalibrationFlowLossWeights)
    assert flow_config.loss.to_loss_weights().active_rotation_weight == pytest.approx(0.75)
    assert flow_config.freeze_backbone_initially


@pytest.mark.parametrize("kwargs", ({"num_steps": 0}, {"num_steps": True}, {"method": "rk4"}, {"ablation_steps": ()}, {"ablation_steps": (1, 1)}, {"ablation_steps": (0, 4)}))
def test_integration_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        CalibrationFlowIntegratorConfig(**kwargs)


@pytest.mark.parametrize("kwargs", ({"rotation_state_scale_rad": 0}, {"translation_state_scale_m": -1}, {"time_offset_state_scale_s": float("nan")}))
def test_flow_head_settings_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        CalibrationFlowHeadSettings(**kwargs)


@pytest.mark.parametrize("kwargs", ({"rotation_weight": -1}, {"translation_scale_m": 0}, {"time_offset_weight": float("inf")}))
def test_flow_loss_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        CalibrationFlowLossConfig(**kwargs)


def test_head_initially_ignores_correction_state_and_time_but_can_learn_them(model_config):
    torch.manual_seed(33)
    head = PooledCalibrationFlowHead(model_config.head, shared_feature_dim=model_config.mlp.output_dim)
    features = torch.randn(5, model_config.mlp.output_dim)
    zeros = torch.zeros(5, 6)
    tau = torch.zeros(5, 1)
    t = torch.zeros(5, 1)
    other_xi = torch.randn_like(zeros) * 0.3
    other_tau = torch.full_like(tau, 0.07)
    other_t = torch.ones_like(t)

    original = head(features, zeros, tau, t)
    changed = head(features, other_xi, other_tau, other_t)
    torch.testing.assert_close(original, changed, atol=1e-7, rtol=1e-7)
    first_linear = next(layer for layer in head.net if isinstance(layer, nn.Linear))
    assert torch.count_nonzero(first_linear.weight[:, head.condition_dim:]) == 0

    loss = (changed - torch.randn_like(changed)).square().mean()
    loss.backward()
    assert first_linear.weight.grad is not None
    assert first_linear.weight.grad[:, head.condition_dim:].abs().sum() > 0


@pytest.mark.parametrize("keys", (("imu",), ("lidar_right", "imu", "lidar")))
def test_fm_training_targets_align_with_dynamic_keys_and_train_new_head_only(model_config, monkeypatch, keys):
    torch.manual_seed(7)
    model = PooledCalibrationFlowModel(model_config)
    model.freeze_feature_extractor(True)
    model.train()
    assert not model.transformer_encoder.training and not model.pooler.training and not model.mlp_body.training
    install_fake_feature_encoder(monkeypatch, model, keys=keys)
    tokens, target_xi, target_tau = make_test_inputs(keys)
    time = torch.full((3, len(keys), 1), 0.35)

    output = model.forward_flow_matching(tokens, target_delta_xi=dict(reversed(list(target_xi.items()))), target_delta_tau=dict(reversed(list(target_tau.items()))), time=time)
    assert output.calibration_keys == keys
    assert output.predicted_velocity.shape == (3, len(keys), 7)
    assert output.flow_batch.target_velocity.shape == (3, len(keys), 7)
    for index, key in enumerate(keys):
        torch.testing.assert_close(output.flow_batch.target_velocity[:, index, :6], target_xi[key])
        torch.testing.assert_close(output.flow_batch.target_velocity[:, index, 6:], target_tau[key])
        torch.testing.assert_close(output.by_key()[key][0], output.predicted_velocity[:, index])
        torch.testing.assert_close(output.flow_batch.correction_xi[:, index], 0.35 * target_xi[key])

    model.zero_grad(set_to_none=True)
    loss = calibration_velocity_loss(output.predicted_velocity, output.flow_batch.target_velocity).total
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.head.parameters())
    assert all(p.grad is None for name in ("transformer_encoder", "pooler", "mlp_body") for p in getattr(model, name).parameters())


def test_unfreezing_restores_shared_mlp_gradient(model_config, monkeypatch):
    model = PooledCalibrationFlowModel(model_config)
    keys = ("imu", "lidar")
    install_fake_feature_encoder(monkeypatch, model, keys=keys)
    tokens, target_xi, target_tau = make_test_inputs(keys)
    model.freeze_feature_extractor(False)
    model.train()
    output = model.forward_flow_matching(tokens, target_xi, target_tau)
    calibration_velocity_loss(output.predicted_velocity, output.flow_batch.target_velocity).total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.mlp_body.parameters())


def test_pretrained_backbone_copy_is_exact_and_does_not_copy_old_head(model_config):
    regression_model = PooledCalibrationModel(model_config)
    flow_model = PooledCalibrationFlowModel(model_config)
    old_backbone = {name: {k: v.clone() for k, v in getattr(regression_model, name).state_dict().items()} for name in ("transformer_encoder", "pooler", "mlp_body")}
    old_head = {key: value.clone() for key, value in flow_model.head.state_dict().items()}
    flow_model.initialize_backbone_from_regression(regression_model, freeze=True)
    for name in old_backbone:
        for key, expected in old_backbone[name].items():
            torch.testing.assert_close(getattr(flow_model, name).state_dict()[key], expected)
    for key, expected in old_head.items():
        torch.testing.assert_close(flow_model.head.state_dict()[key], expected)
    assert flow_model.feature_extractor_frozen
    assert all(not parameter.requires_grad for module in (flow_model.transformer_encoder, flow_model.pooler, flow_model.mlp_body) for parameter in module.parameters())
    assert all(parameter.requires_grad for parameter in flow_model.head.parameters())
    assert "head.delta_xi_head.weight" in regression_model.state_dict()  # Regression checkpoint contract remains intact.
    flow_model.freeze_feature_extractor(False)
    assert all(parameter.requires_grad for module in (flow_model.transformer_encoder, flow_model.pooler, flow_model.mlp_body) for parameter in module.parameters())


@pytest.mark.parametrize("keys", (("imu",), ("lidar", "imu", "lidar_right")))
def test_inference_returns_original_output_contract_and_step_ablation(model_config, monkeypatch, keys):
    torch.manual_seed(66)
    model = PooledCalibrationFlowModel(model_config, integration_steps=4)
    install_fake_feature_encoder(monkeypatch, model, keys=keys)
    tokens, _, _ = make_test_inputs(keys)
    model.eval()
    with torch.no_grad():
        result1 = model(tokens, num_steps=1, method="midpoint")
        result8 = model(tokens, num_steps=8, method="midpoint")
    assert tuple(result8.predictions) == keys
    assert result8.shared_features.shape == (3, len(keys), model_config.mlp.output_dim)
    for key in keys:
        pred1, pred8 = result1.predictions[key], result8.predictions[key]
        assert pred8.delta_xi.shape == (3, 6)
        assert pred8.delta_tau.shape == (3, 1)
        assert pred8.change_time.shape == pred8.change_event_logit.shape == (3, 1)
        assert torch.count_nonzero(pred8.change_time) == torch.count_nonzero(pred8.change_event_logit) == 0
        torch.testing.assert_close(pred1.delta_xi, pred8.delta_xi, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(pred1.delta_tau, pred8.delta_tau, atol=2e-6, rtol=2e-6)


def test_context_is_propagated_to_flow_head(model_config, monkeypatch):
    config = PooledCalibrationModelConfig(
        transformer=model_config.transformer,
        pooler=model_config.pooler,
        mlp=model_config.mlp,
        head=PooledCalibrationHeadConfig(calibration_context_dim=7, hidden_dims=(16, 8), dropout=0),
    )
    model = PooledCalibrationFlowModel(config)
    keys = ("imu", "lidar")
    context = install_fake_feature_encoder(monkeypatch, model, keys=keys, context_dim=7)
    tokens, xi, tau = make_test_inputs(keys)
    output = model.forward_flow_matching(tokens, xi, tau)
    assert output.predicted_velocity.shape == (3, 2, 7)
    shared = output.shared_features
    expected = model.velocity_from_features(shared, output.flow_batch.correction_xi, output.flow_batch.correction_tau, output.flow_batch.time, context)
    torch.testing.assert_close(output.predicted_velocity, expected)
    with pytest.raises(ValueError, match="context"):
        model.velocity_from_features(shared, output.flow_batch.correction_xi, output.flow_batch.correction_tau, output.flow_batch.time)


def test_missing_or_malformed_training_targets_are_rejected(model_config, monkeypatch):
    model = PooledCalibrationFlowModel(model_config)
    keys = ("imu", "lidar")
    install_fake_feature_encoder(monkeypatch, model, keys=keys)
    tokens, xi, tau = make_test_inputs(keys)
    with pytest.raises(KeyError, match="Missing calibration targets"):
        model.forward_flow_matching(tokens, {"imu": xi["imu"]}, tau)
    xi["imu"] = torch.zeros(3, 5)
    with pytest.raises(ValueError, match="shape"):
        model.forward_flow_matching(tokens, xi, tau)
