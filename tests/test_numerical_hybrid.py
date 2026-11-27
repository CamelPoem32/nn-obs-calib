"""Dataset-free tests for the numerical and hybrid model components."""
from types import SimpleNamespace
import pytest
import torch
from obscalib.config import (CalibrationPoolerConfig, MLPConfig, PooledCalibrationHeadConfig, PooledCalibrationModelConfig, TransformerConfig, PooledCalibrationHybridConfig, NumericalEncoderConfig, CalibrationFeatureFusionConfig)
from obscalib.geometry.lie import so3_exp
from obscalib.numerical import (WahbaConfig, NumericalFeatureConfig, NumericalFeatureBatch, WahbaFeatureExtractor, solve_wahba, build_angular_correspondences, encode_wahba_result)
from obscalib.models.numerical_encoder import NumericalEncoder
from obscalib.models.calibration_feature_fusion import CalibrationFeatureFusion
from obscalib.models.pooled_calibration_6d_model import PooledCalibration6DModel
from obscalib.models.pooled_calibration_hybrid_model import PooledCalibrationHybridModel


@pytest.mark.parametrize('angle',[0.,30.,90.,179.,180.])
def test_wahba_exact_rotation(angle):
    rng = torch.Generator().manual_seed(13)
    b = torch.randn(2,20,3,generator=rng,dtype=torch.float64)
    axis = torch.tensor([1.,2.,-3.],dtype=b.dtype); axis /= axis.norm()
    Q = so3_exp(torch.deg2rad(torch.tensor(angle,dtype=b.dtype))*axis).expand(2,3,3)
    a = b @ Q.transpose(-1,-2)
    mask = torch.ones(2,20,dtype=torch.bool); mask[:,0] = False
    a[:,0] = float('nan')
    result = solve_wahba(a,b,mask)
    assert result.valid.all() and (result.num_correspondences == 19).all()
    torch.testing.assert_close(result.rotation,Q,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(torch.linalg.det(result.rotation),torch.ones(2,dtype=b.dtype))
    assert result.residual_after.max() < 1e-10
    assert (result.eigenvalues[:,0] > 0).all()


def test_degenerate_and_empty_wahba():
    b = torch.zeros(3,10,3,dtype=torch.float64)
    b[0,:,0] = torch.arange(1,11,dtype=b.dtype)
    b[2,:,0] = 1.
    mask = torch.ones(3,10,dtype=torch.bool); mask[2,1:] = False
    result = solve_wahba(b,b,mask)
    assert result.valid.tolist() == [True,False,False]
    assert result.eigenvalues[0,0] < 1e-10
    torch.testing.assert_close(result.rotation[1:],torch.eye(3,dtype=b.dtype).expand(2,3,3))
    for mode,dim in [('3d',15),('6d',18)]:
        x = encode_wahba_result(result,NumericalFeatureConfig(rotation_mode=mode))
        assert x.shape == (3,dim) and torch.isfinite(x).all()
        assert not x[1:].any()
    empty = solve_wahba(b[:,:0],b[:,:0]); assert not empty.valid.any()


def test_reflection_is_not_returned():
    b = torch.eye(3,dtype=torch.float64)[None]
    a = b.clone(); a[...,0] *= -1
    result = solve_wahba(a,b)
    torch.testing.assert_close(torch.linalg.det(result.rotation),torch.ones(1,dtype=b.dtype))
    assert result.residual_after.item() > 0


def test_weighted_normalized_wahba_and_representation():
    b = torch.tensor([[[1.,0.,0.],[0.,2.,0.],[0.,0.,3.],[5.,5.,5.]]],dtype=torch.float64)
    Q = so3_exp(torch.tensor([[.4,-.7,1.2]],dtype=b.dtype))
    a = b @ Q.transpose(-1,-2); a[:,3] = float('nan')
    result = solve_wahba(a,b,weights=torch.tensor([[1.,2.,3.,0.]],dtype=b.dtype),config=WahbaConfig(normalize_vectors=True))
    torch.testing.assert_close(result.rotation,Q)
    f = encode_wahba_result(result,NumericalFeatureConfig(rotation_mode='6d',include_information_matrix=False,include_eigenvalues=False,include_residuals=False,include_motion_energy=False))
    torch.testing.assert_close(f,torch.cat((Q[...,0],Q[...,1]),dim=-1))


def stream(values,timestamps,key,measurement_type,mask=None):
    return SimpleNamespace(values=values,timestamps=timestamps,calibration_key=key,measurement_type=measurement_type,sample_mask=torch.ones_like(timestamps,dtype=torch.bool) if mask is None else mask)


def sample_streams(shift=0.):
    # Piecewise-linear gyro with non-collinear excitation. All interval lengths
    # and endpoints are explicit, including an interval with fractional knots.
    dtype = torch.float64
    t = torch.linspace(0,4,41,dtype=dtype)[None]
    g = torch.stack((t,t.square(),torch.sin(t)),dim=-1)
    ref = stream(g,t,'imu','gyroscope')
    start = torch.tensor([[.25,1.05,2.15]],dtype=dtype)
    end = start + .5
    raw = SimpleNamespace(timestamps=end,interval_start_timestamps=start,values=torch.eye(4,dtype=dtype).expand(1,3,4,4).clone())
    tmp = stream(torch.ones(1,3,6,dtype=dtype),end,'lidar','lidar_odometry')
    vectors = build_angular_correspondences(ref,tmp,raw).reference_vectors
    Q = so3_exp(torch.tensor([[.4,-.6,2.7]],dtype=dtype))
    lidar_vectors = vectors @ Q
    lidar = stream(torch.cat((lidar_vectors,torch.zeros_like(lidar_vectors)),dim=-1),end+shift,'lidar','lidar_odometry')
    right = stream(torch.cat((vectors,torch.zeros_like(vectors)),dim=-1),end,'lidar_right','lidar_odometry')
    return {'gyroscope':ref,'lidar_odometry':lidar,'lidar_right_odometry':right},{'lidar_odometry':raw,'lidar_right_odometry':raw},Q


def test_timestamp_matching_and_no_temporal_search():
    streams,raw,Q = sample_streams()
    pairs = build_angular_correspondences(streams['gyroscope'],streams['lidar_odometry'],raw['lidar_odometry'])
    # First coordinate gyro(t)=t, integral from .25 to .75 is .25.
    torch.testing.assert_close(pairs.reference_vectors[0,0,0],torch.tensor(.25,dtype=Q.dtype))
    assert pairs.sample_mask.all()
    extractor = WahbaFeatureExtractor(WahbaConfig(reference_key='imu'),NumericalFeatureConfig(rotation_mode='6d'))
    result = extractor(streams,('lidar_right','imu','lidar'),raw_streams=raw)
    assert result.values.shape == (1,3,18)
    assert result.valid.tolist() == [[True,False,True]] and not result.values[:,1].any()
    assert 'imu' not in result.results
    torch.testing.assert_close(result.results['lidar'].rotation,Q,rtol=1e-9,atol=1e-9)
    shifted,raw,_ = sample_streams(.3)
    shifted_result = extractor(shifted,('imu','lidar','lidar_right'),raw_streams=raw)
    assert shifted_result.results['lidar'].residual_after.item() > .01
    torch.testing.assert_close(shifted_result.results['lidar_right'].rotation,torch.eye(3,dtype=Q.dtype)[None],atol=1e-9,rtol=1e-9)


def test_partial_coverage_and_masked_gyro_hole():
    streams,raw,_ = sample_streams()
    streams['gyroscope'].sample_mask[:,5] = False
    p = build_angular_correspondences(streams['gyroscope'],streams['lidar_odometry'],raw['lidar_odometry'])
    assert p.sample_mask.tolist() == [[False,True,True]]
    streams['lidar_odometry'].timestamps += 10.
    p = build_angular_correspondences(streams['gyroscope'],streams['lidar_odometry'],raw['lidar_odometry'])
    assert not p.sample_mask.any() and not p.reference_vectors.any()


def test_fusion_identity_and_encoder_gradients():
    torch.manual_seed(13)
    encoder = NumericalEncoder(18,NumericalEncoderConfig(hidden_dim=16,output_dim=8))
    fusion = CalibrationFeatureFusion(16,8)
    x = torch.randn(4,3,18); h = torch.randn(4,3,16)
    valid = torch.tensor([[False,True,True]]*4)
    z = encoder(x,valid); assert not z.any()
    fused = fusion(torch.cat((h,z),dim=-1))
    torch.testing.assert_close(fused,h,atol=1e-6,rtol=1e-6)
    fused.square().mean().backward()
    assert encoder.mlp[-1].weight.grad.abs().sum() > 0
    assert fusion.mlp[0].weight.grad.abs().sum() > 0
    optimizer = torch.optim.SGD(encoder.parameters(),lr=.1); optimizer.step()
    z = encoder(x,valid)
    assert not z[:,0].any() and z[:,1:].abs().sum() > 0
    optimizer.zero_grad(); fusion(torch.cat((h,z),dim=-1)).square().mean().backward()
    assert encoder.mlp[0].weight.grad.abs().sum() > 0


def small_config():
    return PooledCalibrationModelConfig(transformer=TransformerConfig(input_dim=8,d_model=32,n_heads=4,num_layers=2,dim_feedforward=64),pooler=CalibrationPoolerConfig(n_heads=4),mlp=MLPConfig(hidden_dims=(64,32),output_dim=32),head=PooledCalibrationHeadConfig(hidden_dims=(32,16)))


def test_hybrid_checkpoint_transfer_and_forward(monkeypatch):
    torch.manual_seed(14)
    config = small_config()
    base = PooledCalibration6DModel(config).eval()
    hybrid = PooledCalibrationHybridModel(PooledCalibrationHybridConfig(backbone=config,numerical_feature_dim=18)).eval()
    hybrid.initialize_from_regressor(base)
    for name,p in base.state_dict().items():
        torch.testing.assert_close(p,hybrid.state_dict()[name],rtol=0.,atol=0.)
    keys = ('imu','lidar','lidar_right')
    tokens = SimpleNamespace(calibration_keys=keys)
    h = torch.randn(4,3,32)
    for net in (base,hybrid):
        monkeypatch.setattr(net,'encode_calibration_features',lambda tokens,calibration_context=None: (h,None))
    numeric = NumericalFeatureBatch(torch.randn(4,3,18),torch.ones(4,3,dtype=torch.bool),('lidar_right','imu','lidar'),'imu')
    old = base(tokens); new = hybrid(tokens,numerical_features=numeric)
    for key in keys:
        for field in ('rotation_6d','delta_translation','delta_tau','delta_rotation'):
            torch.testing.assert_close(getattr(old.predictions[key],field),getattr(new.predictions[key],field),atol=3e-6,rtol=3e-6)
    loss = sum(p.delta_tau.square().mean()+p.rotation_6d.square().mean() for p in new.predictions.values()); loss.backward()
    assert hybrid.numerical_encoder.mlp[-1].weight.grad.abs().sum() > 0
    assert any(p.grad is not None for p in hybrid.head_6d.parameters())
    with torch.no_grad():
        hybrid.numerical_encoder.mlp[-1].bias.fill_(.2)
        hybrid.numerical_encoder.mlp[-1].weight.normal_()
    z = hybrid.encode_numerical_features(h,numeric,keys)
    assert not z[:,0].any()
    vals,valid = numeric.aligned(keys)
    torch.testing.assert_close(vals[:,1],numeric.values[:,2])
    assert not valid[:,0].any()
    assert z[:,1:].abs().sum() > 0
    assert not hybrid.encode_numerical_features(h,numeric,keys,use_numerical=False).any()
    # All hybrid checkpoint tensors round-trip strictly.
    clone = PooledCalibrationHybridModel(hybrid.hybrid_config)
    clone.load_state_dict(hybrid.state_dict(),strict=True)
