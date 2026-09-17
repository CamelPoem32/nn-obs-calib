import torch

from obscalib.config import (
    CalibrationHeadConfig,
    MLPConfig,
    ObsCalibModelConfig,
    TransformerConfig,
)
from obscalib.data import TokenBatch
from obscalib.models import CalibrationHead, ObsCalibModel


def test_calibration_head_named_output_shapes() -> None:
    head = CalibrationHead(
        CalibrationHeadConfig(
            calibration_context_dim=4,
            hidden_dims=(8, 6),
        ),
        shared_feature_dim=10,
    )

    prediction = head(torch.randn(3, 10), torch.randn(3, 4))

    assert prediction.change_logit.shape == (3, 1)
    assert prediction.change_time.shape == (3, 1)
    assert prediction.delta_xi.shape == (3, 6)
    assert prediction.delta_tau.shape == (3, 1)


def test_model_builds_and_evaluates_multiple_module_dict_heads() -> None:
    transformer = TransformerConfig(
        input_dim=5,
        d_model=8,
        n_heads=2,
        num_layers=1,
        dim_feedforward=16,
        num_summary_tokens=2,
    )
    mlp = MLPConfig(hidden_dims=(12,), output_dim=10)
    head = CalibrationHeadConfig(
        calibration_context_dim=4,
        hidden_dims=(8,),
    )
    model = ObsCalibModel(
        ObsCalibModelConfig(
            transformer=transformer,
            mlp=mlp,
            calibration_head=head,
            head_keys=("imu", "camera"),
        )
    )
    tokens = TokenBatch(
        x=torch.randn(3, 6, 5),
        token_mask=torch.ones(3, 6, dtype=torch.bool),
        sensor_ids=torch.zeros(3, 6, dtype=torch.long),
        measurement_types=torch.zeros(3, 6, dtype=torch.long),
    )

    output = model(
        tokens,
        {
            "imu": torch.randn(3, 4),
            "camera": torch.randn(3, 4),
        },
    )

    assert set(model.heads) == {"imu", "camera"}
    assert set(output.predictions) == {"imu", "camera"}
    assert output.shared_features is not None
    assert output.shared_features.shape == (3, 10)
