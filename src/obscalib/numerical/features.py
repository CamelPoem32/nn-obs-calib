"""Extract numerical calibration features without model or target dependencies."""
from collections.abc import Mapping
import torch
from obscalib.geometry.lie import so3_log
from .angular_motion import build_angular_correspondences, select_angular_streams
from .config import NumericalFeatureConfig, WahbaConfig
from .structures import NumericalFeatureBatch, WahbaResult
from .wahba import solve_wahba


def encode_wahba_result(result: WahbaResult, config: NumericalFeatureConfig) -> torch.Tensor:
    """Convert SO(3) and reliability diagnostics to [B,F]. Invalid rows are zero.

    H and its eigenvalues share trace normalization. Raw motion energy remains
    separate. Residuals and energy are not artificially normalized by batch data.
    """
    R = result.rotation
    if config.rotation_mode == '3d':
        rotation = so3_log(R)
    elif config.rotation_6d_layout == 'columns':
        rotation = R[..., :, :2].transpose(-1,-2).reshape(R.shape[:-2]+(6,))
    else:
        rotation = R[..., :2, :].reshape(R.shape[:-2]+(6,))
    parts = [rotation]
    trace = result.information_matrix.diagonal(dim1=-2,dim2=-1).sum(-1).clamp_min(torch.finfo(R.dtype).tiny)
    if config.include_information_matrix:
        i,j = torch.triu_indices(3,3,device=R.device)
        parts.append(result.information_matrix[...,i,j]/trace[...,None])
    if config.include_eigenvalues:
        parts.append(result.eigenvalues/trace[...,None])
    if config.include_residuals:
        parts.extend((result.residual_before[...,None], result.residual_after[...,None]))
    if config.include_motion_energy:
        parts.append(result.motion_energy[...,None])
    values = torch.cat(parts,dim=-1)
    return torch.where(result.valid[...,None],values,0.)


class WahbaFeatureExtractor:
    """Compute relative LiDAR-to-reference corrections in the prior body frame.

    The output is ordered by calibration key, including an exact zero reference
    row. Raw streams are used exclusively to recover measurement interval lengths.
    No calibration targets, offset search, or model-dependent parameters enter.
    """
    def __init__(self, wahba_config: WahbaConfig | None = None, feature_config: NumericalFeatureConfig | None = None) -> None:
        self.wahba_config = wahba_config or WahbaConfig()
        self.feature_config = feature_config or NumericalFeatureConfig()

    @torch.no_grad()
    def __call__(self, processed_streams: Mapping, calibration_keys: tuple[str,...], *, raw_streams: Mapping) -> NumericalFeatureBatch:
        keys = tuple(calibration_keys)
        reference_key, ref_name, sources = select_angular_streams(processed_streams,raw_streams,keys,self.wahba_config)
        ref = processed_streams[ref_name]
        B = ref.values.shape[0]
        values = ref.values.new_zeros(B,len(keys),self.feature_config.feature_dim)
        valid = torch.zeros(B,len(keys),device=values.device,dtype=torch.bool)
        results = {}
        for key,name in sources.items():
            pairs = build_angular_correspondences(ref,processed_streams[name],raw_streams[name],self.wahba_config)
            result = solve_wahba(pairs.reference_vectors,pairs.source_vectors,pairs.sample_mask,pairs.weights,config=self.wahba_config)
            i = keys.index(key)
            values[:,i] = encode_wahba_result(result,self.feature_config)
            valid[:,i] = result.valid
            results[key] = result
        return NumericalFeatureBatch(values,valid,keys,reference_key,results)
