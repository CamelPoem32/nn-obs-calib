"""Run with: pytest -q tests/test_pooled_calibration_classifier.py

Drop this file into the nn-obs-calib tests directory. No dataset is needed.
"""

from types import SimpleNamespace

import pytest
import torch

from obscalib.config import CalibrationPoolerConfig, MLPConfig, PooledCalibrationHeadConfig, PooledCalibrationModelConfig, TransformerConfig
from obscalib.models.pooled_calibration_6d_model import PooledCalibration6DModel
from obscalib.models.pooled_calibration_classifier_model import PooledCalibrationClassifierModel, calibration_fault_loss, initialize_classifier_from_regressor
from obscalib.models.pooled_calibration_model import PooledCalibrationModel


def small_config() -> PooledCalibrationModelConfig:
    """Same structural interfaces as 002.27, with fewer layers for a fast test."""
    return PooledCalibrationModelConfig(
        transformer=TransformerConfig(input_dim=8, d_model=32, n_heads=4, num_layers=2, dim_feedforward=64, dropout=0.0, num_summary_tokens=1, activation="gelu"),
        mlp=MLPConfig(hidden_dims=(64, 32), output_dim=32, activation="gelu", dropout=0.0),
        pooler=CalibrationPoolerConfig(n_heads=4, dropout=0.0, calibration_context_dim=0),
        head=PooledCalibrationHeadConfig(calibration_context_dim=0, hidden_dims=(32, 16), activation="gelu", dropout=0.0),
    )


@pytest.mark.parametrize("regressor_type", [PooledCalibration6DModel, PooledCalibrationModel])
def test_strict_backbone_transfer(regressor_type) -> None:
    source = regressor_type(small_config())
    target = PooledCalibrationClassifierModel(small_config())
    initialize_classifier_from_regressor(target, source)
    for name in ("transformer_encoder", "pooler", "mlp_body"):
        original = getattr(source, name).state_dict()
        copied = getattr(target, name).state_dict()
        assert original.keys() == copied.keys()
        for key in original:
            torch.testing.assert_close(original[key], copied[key], rtol=0.0, atol=0.0)


def test_classifier_outputs_and_gradients(monkeypatch) -> None:
    classifier = PooledCalibrationClassifierModel(small_config())
    classifier.set_backbone_trainable(False)
    assert not any(p.requires_grad for name in ("transformer_encoder", "pooler", "mlp_body") for p in getattr(classifier, name).parameters())
    assert all(p.requires_grad for p in classifier.head.parameters())

    features = torch.randn(5, 3, classifier.config.mlp.output_dim)
    monkeypatch.setattr(classifier, "encode_calibration_features", lambda tokens, calibration_context=None: (features, None))
    tokens = SimpleNamespace(calibration_keys=("imu", "lidar", "lidar_right"))
    output = classifier(tokens)
    assert output.logits.shape == (5, 3, 3)
    assert output.probabilities.shape == (5, 3, 3)
    assert set(output.predictions) == set(tokens.calibration_keys)
    assert output.predictions["lidar"].unreliable_logit.shape == (5, 1)

    rotation_labels = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.float32)
    time_labels = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=torch.float32)
    loss, components = calibration_fault_loss(output, rotation_labels, time_labels)
    assert set(components) == {"joint", "rotation", "time"}
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and bool(torch.any(p.grad != 0)) for p in classifier.head.parameters())
    classifier.set_backbone_trainable(True)
    assert all(p.requires_grad for name in ("transformer_encoder", "pooler", "mlp_body") for p in getattr(classifier, name).parameters())
