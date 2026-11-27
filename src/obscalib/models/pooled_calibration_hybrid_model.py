"""Absolute 6D calibration regression with per-sensor numerical feature fusion."""
from collections.abc import Mapping
import torch
from obscalib.config import PooledCalibrationHybridConfig
from obscalib.data.structures import TokenBatch
from obscalib.numerical.structures import NumericalFeatureBatch
from .numerical_encoder import NumericalEncoder
from .calibration_feature_fusion import CalibrationFeatureFusion
from .pooled_calibration_6d_model import PooledCalibration6DModel, PooledCalibration6DOutput


class PooledCalibrationHybridModel(PooledCalibration6DModel):
    """Preserve all pretrained parameter names, including the absolute head.

    Numerical features are computed outside forward. Only the corresponding
    source sensor receives each embedding; the reference receives exact zeros.
    """
    def __init__(self, config: PooledCalibrationHybridConfig) -> None:
        super().__init__(config.backbone)
        self.hybrid_config = config
        self.numerical_encoder = NumericalEncoder(config.numerical_feature_dim,config.numerical_encoder)
        self.feature_fusion = CalibrationFeatureFusion(config.backbone.mlp.output_dim,config.numerical_encoder.output_dim,config.fusion)

    def encode_numerical_features(self, shared: torch.Tensor, numerical_features: NumericalFeatureBatch, calibration_keys: tuple[str,...], *, use_numerical: bool = True) -> torch.Tensor:
        values,valid = numerical_features.aligned(calibration_keys)
        if values.shape[:2] != shared.shape[:2] or values.shape[-1] != self.hybrid_config.numerical_feature_dim:
            raise ValueError('Numerical and neural feature shapes disagree.')
        if values.device != shared.device:
            raise ValueError('Numerical and neural features must share device.')
        if not use_numerical:
            return shared.new_zeros((*shared.shape[:2],self.hybrid_config.numerical_encoder.output_dim))
        return self.numerical_encoder(values.to(shared.dtype),valid)

    def forward(self, tokens: TokenBatch, calibration_context: Mapping[str,torch.Tensor] | None = None, *, numerical_features: NumericalFeatureBatch, use_numerical: bool = True) -> PooledCalibration6DOutput:
        shared,stacked_context = self.encode_calibration_features(tokens,calibration_context)
        numerical = self.encode_numerical_features(shared,numerical_features,tuple(tokens.calibration_keys),use_numerical=use_numerical)
        fused = self.feature_fusion(torch.cat((shared,numerical),dim=-1))
        return self.decode_calibration_features(fused,stacked_context,tuple(tokens.calibration_keys))

    def initialize_from_regressor(self, regressor: PooledCalibration6DModel) -> None:
        """Strictly copy all original modules, including head_6d; leave new layers intact."""
        if type(regressor) is not PooledCalibration6DModel or self.config != regressor.config:
            raise ValueError('Expected original PooledCalibration6DModel with identical backbone config.')
        source = regressor.state_dict()
        own = self.state_dict()
        added = {k for k in own if k.startswith(('numerical_encoder.','feature_fusion.'))}
        if set(source) != set(own)-added:
            raise RuntimeError('Original checkpoint keys do not match hybrid backbone/head.')
        if any(source[k].shape != own[k].shape for k in source):
            raise ValueError('Original checkpoint tensor shapes differ.')
        self.load_state_dict({**own,**source},strict=True)
