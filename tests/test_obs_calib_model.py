import torch
from torch import nn

from obscalib.config import (
    CalibrationHeadConfig,
    MLPConfig,
    ObsCalibModelConfig,
    TransformerConfig,
)
from obscalib.data import TokenBatch
from obscalib.models import ObsCalibModel


def test_complete_learned_path_shapes_derived_dimensions_and_gradients() -> None:
    batch_size = 2
    config = ObsCalibModelConfig(
        transformer=TransformerConfig(
            input_dim=5,
            d_model=8,
            n_heads=2,
            num_layers=1,
            dim_feedforward=16,
            dropout=0.0,
            num_summary_tokens=3,
        ),
        mlp=MLPConfig(
            hidden_dims=(12,),
            output_dim=10,
            activation="gelu",
            dropout=0.0,
        ),
        calibration_head=CalibrationHeadConfig(
            calibration_context_dim=4,
            hidden_dims=(8,),
            activation="silu",
            dropout=0.0,
        ),
        head_keys=("imu", "lidar"),
    )
    model = ObsCalibModel(config)

    # K * d_model and d_shared are composition-derived, not repeated choices.
    assert config.summary_feature_dim == 3 * 8
    assert config.shared_feature_dim == 10
    assert model.mlp_body.input_dim == 24
    assert all(head.shared_feature_dim == 10 for head in model.heads.values())

    tokens = TokenBatch(
        x=torch.randn(batch_size, 6, 5, requires_grad=True),
        token_mask=torch.tensor(
            [[True, True, True, False, False, False], [True] * 6]
        ),
        sensor_ids=torch.zeros(batch_size, 6, dtype=torch.long),
        measurement_types=torch.zeros(batch_size, 6, dtype=torch.long),
    )
    calibration_context = {
        "imu": torch.randn(batch_size, 4),
        "lidar": torch.randn(batch_size, 4),
    }

    output = model(tokens, calibration_context)

    assert set(output.predictions) == {"imu", "lidar"}
    assert output.shared_features is not None
    assert output.shared_features.shape == (batch_size, 10)
    for prediction in output.predictions.values():
        assert prediction.change_logit.shape == (batch_size, 1)
        assert prediction.change_time.shape == (batch_size, 1)
        assert prediction.delta_xi.shape == (batch_size, 6)
        assert prediction.delta_tau.shape == (batch_size, 1)

    # Include every deterministic output so gradients cross both heads and the
    # complete Transformer -> flattened summaries -> shared MLP path.
    loss = sum(
        prediction.change_logit.sum()
        + prediction.change_time.sum()
        + prediction.delta_xi.sum()
        + prediction.delta_tau.sum()
        for prediction in output.predictions.values()
    )
    loss.backward()

    assert tokens.x.grad is not None
    assert model.transformer_encoder.learned_summary_tokens.grad is not None
    assert isinstance(model.transformer_encoder.input_projection, nn.Linear)
    assert model.transformer_encoder.input_projection.weight.grad is not None
    encoder_layer = model.transformer_encoder.encoder.layers[0]
    assert encoder_layer.self_attn.in_proj_weight.grad is not None
    first_mlp_layer = next(
        layer for layer in model.mlp_body.layers if isinstance(layer, nn.Linear)
    )
    assert first_mlp_layer.weight.grad is not None
    for head in model.heads.values():
        assert head.change_logit_head.weight.grad is not None
        assert head.change_time_head.weight.grad is not None
        assert head.delta_xi_head.weight.grad is not None
        assert head.delta_tau_head.weight.grad is not None
