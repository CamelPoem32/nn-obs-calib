'''Synthetic regression tests for optional right-LiDAR ingestion in KAIST datasets.'''

from pathlib import Path

import numpy as np
import pytest
import torch

from kaist_dataset.data import IMUData, LidarData
from kaist_dataset.dataset import KAISTCalibrationDataset, KAISTRigCalibration, KAISTSequenceSpec, build_kaist_sequence_specs
from obscalib.augmentations.config import PerturbationMagnitudeConfig, PriorPerturbationConfig
from obscalib.augmentations.prior_perturbation import CalibrationPriorPerturber
from obscalib.calibration.state import CalibrationState
from obscalib.config import WindowingConfig
from obscalib.data.collation import collate_windows
from obscalib.data.structures import GeometryType, MeasurementType


CALIBRATION_KEYS_2 = ('imu', 'lidar')
CALIBRATION_KEYS_3 = ('imu', 'lidar', 'lidar_right')


def _make_rig(*, include_right: bool) -> KAISTRigCalibration:
    T_B_I = np.eye(4)
    T_B_L = np.eye(4)
    T_B_L[:3, 3] = [-0.4, 0.3, 1.9]
    T_B_L_right = np.eye(4)
    T_B_L_right[:3, 3] = [-0.4, -0.3, 1.9]
    return KAISTRigCalibration(T_B_I=T_B_I, T_B_L=T_B_L, tau_imu_s=0.001, tau_lidar_s=-0.002, T_B_L_right=T_B_L_right if include_right else None, tau_lidar_right_s=0.003)


def _make_spec(tmp_path: Path, *, name: str = 'UrbanSynthetic', include_right: bool = True) -> KAISTSequenceSpec:
    sensor_data_root = tmp_path / name / 'sensor_data'
    sensor_data_root.mkdir(parents=True, exist_ok=True)
    lidar_pose_csv = tmp_path / name / 'lidar_map_poses_vlp_left.csv'
    lidar_pose_csv.touch()
    lidar_right_pose_csv = tmp_path / name / 'lidar_map_poses_vlp_right.csv'
    if include_right:
        lidar_right_pose_csv.touch()
    return KAISTSequenceSpec(name=name, sensor_data_root=sensor_data_root, lidar_pose_csv=lidar_pose_csv, lidar_right_pose_csv=lidar_right_pose_csv if include_right else None)


def _make_lidar(*, right: bool) -> LidarData:
    # Right LiDAR has a shorter and differently phased valid scan time range.
    timestamps = np.arange(1.0, 13.0 + 1e-10, 0.5) if right else np.arange(0.0, 14.0 + 1e-10, 0.5)
    relative_poses = np.repeat(np.eye(4)[None], len(timestamps) - 1, axis=0)
    relative_poses[:, :3, 3] = [0.02, 0.01, 0.0] if right else [0.05, 0.0, 0.0]
    return LidarData(timestamps_s=timestamps[1:], relative_poses_se3=relative_poses, scan_timestamps_s=timestamps)


@pytest.fixture
def fake_kaist_io(monkeypatch):
    '''Patch only disk readers; use the real adapter, windowing, and collator.'''
    import kaist_dataset.dataset as dataset_module

    imu_times = np.arange(0.0, 14.0 + 1e-10, 0.01)
    imu = IMUData(timestamps_s=imu_times, accel_mps2=np.tile([0.0, 0.0, 9.81], (len(imu_times), 1)), gyro_radps=np.tile([0.1, 0.2, 0.3], (len(imu_times), 1)), name='xsens')
    loaded_paths = []

    def load_imus(sensor_data_root, target_frequency_hz=None):
        return {'xsens': imu}

    def load_lidar(path):
        loaded_paths.append(Path(path))
        return _make_lidar(right='right' in Path(path).stem)

    monkeypatch.setattr(dataset_module, 'load_imus', load_imus)
    monkeypatch.setattr(dataset_module, 'load_lidar_relative_poses', load_lidar)
    return loaded_paths


def _make_dataset(spec: KAISTSequenceSpec, *, include_right: bool) -> KAISTCalibrationDataset:
    return KAISTCalibrationDataset([spec], _make_rig(include_right=include_right), WindowingConfig(window_duration_s=5.0, window_stride_s=5.0, max_samples_per_sensor=700))


def test_three_sensor_spec_builder_is_explicit_opt_in(tmp_path):
    kwargs = dict(sequence_names=['Urban16'], kaist_dataset_root=tmp_path / 'raw', lidar_pose_root=tmp_path / 'poses')
    legacy = build_kaist_sequence_specs(**kwargs)[0]
    assert legacy.lidar_right_pose_csv is None
    assert legacy.lidar_pose_csv == tmp_path / 'poses' / 'Urban16' / 'lidar_map_poses_vlp_left.csv'

    three_sensor = build_kaist_sequence_specs(**kwargs, lidar_right_pose_filename='lidar_map_poses_vlp_right.csv')[0]
    assert three_sensor.lidar_right_pose_csv == tmp_path / 'poses' / 'Urban16' / 'lidar_map_poses_vlp_right.csv'
    assert three_sensor.lidar_pose_csv == legacy.lidar_pose_csv


def test_legacy_two_sensor_import_unchanged(tmp_path, fake_kaist_io):
    spec = _make_spec(tmp_path, include_right=False)
    dataset = _make_dataset(spec, include_right=False)

    assert dataset.calibration_keys == CALIBRATION_KEYS_2
    assert len(fake_kaist_io) == 1
    assert len(dataset) == 2
    assert set(dataset[0].streams) == {'gyro', 'accel', 'lidar'}
    assert tuple(dataset[0].current_calibration) == CALIBRATION_KEYS_2
    assert dataset.sequence_info[0].num_lidar_right_relative_poses == 0
    assert 'right LiDAR=' not in dataset.summary()


def test_three_sensor_import_routing_overlap_and_batch(tmp_path, fake_kaist_io):
    spec = _make_spec(tmp_path, include_right=True)
    dataset = _make_dataset(spec, include_right=True)

    assert dataset.calibration_keys == CALIBRATION_KEYS_3
    assert fake_kaist_io == [spec.lidar_pose_csv, spec.lidar_right_pose_csv]
    assert len(dataset) == 2
    assert dataset.sequence_info[0].num_lidar_relative_poses == 28
    assert dataset.sequence_info[0].num_lidar_right_relative_poses == 24
    assert dataset.sequence_info[0].common_start_time_s == pytest.approx(1.0)
    assert dataset.sequence_info[0].common_end_time_s == pytest.approx(13.0)
    assert 'right LiDAR=24' in dataset.summary()

    sample = dataset[0]
    assert tuple(sample.current_calibration) == CALIBRATION_KEYS_3
    assert set(sample.streams) == {'gyro', 'accel', 'lidar', 'lidar_right'}
    assert sample.metadata['lidar'].calibration_key == 'lidar'
    assert sample.metadata['lidar_right'].calibration_key == 'lidar_right'
    assert sample.metadata['lidar_right'].geometry_type == GeometryType.SE3
    assert sample.metadata['lidar_right'].measurement_type == MeasurementType.LIDAR_POSE
    np.testing.assert_allclose(sample.current_calibration['lidar_right'].transform[0].numpy(), _make_rig(include_right=True).T_B_L_right)
    assert sample.current_calibration['lidar_right'].time_offset.item() == pytest.approx(0.003)
    assert sample.streams['lidar_right'].interval_start_timestamps is not None
    assert sample.streams['lidar_right'].values.shape[-2:] == (4, 4)
    assert not torch.allclose(sample.streams['lidar'].values[:2], sample.streams['lidar_right'].values[:2])

    batch = collate_windows([dataset[0], dataset[1]])
    assert tuple(batch.current_calibration) == CALIBRATION_KEYS_3
    assert batch.streams['lidar_right'].values.shape[0] == 2
    assert batch.current_calibration['lidar_right'].transform.shape == (2, 4, 4)
    batch.streams['lidar_right'].validate()


def test_three_sensor_requires_right_extrinsic_and_csv(tmp_path, fake_kaist_io):
    spec = _make_spec(tmp_path, include_right=True)
    with pytest.raises(ValueError, match='T_B_L_right'):
        _make_dataset(spec, include_right=False)

    spec.lidar_right_pose_csv.unlink()
    with pytest.raises(FileNotFoundError, match='right LiDAR'):
        _make_dataset(spec, include_right=True)


def test_three_sensor_optional_extrinsic_validation():
    with pytest.raises(ValueError, match='T_B_L_right'):
        KAISTRigCalibration(T_B_I=np.eye(4), T_B_L=np.eye(4), T_B_L_right=np.eye(3))
    with pytest.raises(ValueError, match='temporal calibrations'):
        KAISTRigCalibration(T_B_I=np.eye(4), T_B_L=np.eye(4), T_B_L_right=np.eye(4), tau_lidar_right_s=float('nan'))


def test_three_sensor_pair_prior_selects_one_calibration_per_sample():
    batch_size = 128
    truth = {key: CalibrationState(transform=torch.eye(4).expand(batch_size, 4, 4).clone(), time_offset=torch.zeros(batch_size, 1)) for key in CALIBRATION_KEYS_3}
    sampler = CalibrationPriorPerturber(PriorPerturbationConfig(enabled=True, probability=1.0, rotation=PerturbationMagnitudeConfig(maximum=np.deg2rad(5.0)), translation=None, time_offset=PerturbationMagnitudeConfig(maximum=0.01), one_calibration_per_sample=True, one_rotation_axis_per_sample=False, one_perturbation_type_per_sample=False))
    priors, deltas_xi, deltas_tau = sampler(truth, generator=torch.Generator().manual_seed(12345))

    assert tuple(priors) == CALIBRATION_KEYS_3
    active_mask = torch.stack([(deltas_xi[key][:, :3].abs().amax(dim=1) > 1e-8) | (deltas_tau[key][:, 0].abs() > 1e-8) for key in CALIBRATION_KEYS_3], dim=1)
    assert torch.all(active_mask.sum(dim=1) == 1)
    assert torch.all(active_mask.sum(dim=0) > 0)
    for key in CALIBRATION_KEYS_3:
        torch.testing.assert_close(priors[key].time_offset, deltas_tau[key])
        assert torch.all(torch.linalg.vector_norm(deltas_xi[key][:, :3], dim=1) <= np.deg2rad(5.0) + 1e-7)
        assert torch.all(deltas_tau[key].abs() <= 0.01 + 1e-7)


def test_three_sensor_reads_distinct_real_lidar_pose_csvs(tmp_path, monkeypatch):
    '''Exercise the actual CSV -> relative SE3 conversion for both LiDARs.'''
    import csv
    import kaist_dataset.dataset as dataset_module

    spec = _make_spec(tmp_path, include_right=True)
    columns = ['timestamp', 'r11', 'r12', 'r13', 't11', 'r21', 'r22', 'r23', 't21', 'r31', 'r32', 'r33', 't31', 'success']

    for right, path in ((False, spec.lidar_pose_csv), (True, spec.lidar_right_pose_csv)):
        scan_timestamps = _make_lidar(right=right).scan_timestamps_s
        with path.open('w', newline='') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=columns)
            writer.writeheader()
            for i, timestamp in enumerate(scan_timestamps):
                writer.writerow(dict(zip(columns, [timestamp, 1, 0, 0, i * (0.02 if right else 0.05), 0, 1, 0, i * (0.01 if right else 0.0), 0, 0, 1, 0, 1])))

    imu_timestamps = np.arange(0.0, 14.0 + 1e-10, 0.01)
    imu = IMUData(timestamps_s=imu_timestamps, accel_mps2=np.tile([0, 0, 9.81], (len(imu_timestamps), 1)), gyro_radps=np.zeros((len(imu_timestamps), 3)), name='xsens')
    monkeypatch.setattr(dataset_module, 'load_imus', lambda root, target_frequency_hz=None: {'xsens': imu})

    dataset = _make_dataset(spec, include_right=True)
    left_increment = dataset[0].streams['lidar'].values[0, :3, 3].numpy()
    right_increment = dataset[0].streams['lidar_right'].values[0, :3, 3].numpy()
    np.testing.assert_allclose(left_increment, [0.05, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(right_increment, [0.02, 0.01, 0.0], atol=1e-6)


def test_three_sensor_true_tau_augmentation_keeps_measurement_times_consistent(tmp_path, fake_kaist_io):
    from obscalib.augmentations.config import TrueCalibrationAugmentationConfig
    from obscalib.augmentations.true_calibration import TrueCalibrationAugmenter

    dataset = _make_dataset(_make_spec(tmp_path, include_right=True), include_right=True)
    base_window = collate_windows([dataset[0], dataset[1]])
    augmenter = TrueCalibrationAugmenter(TrueCalibrationAugmentationConfig(enabled=True, probability=1.0, rotation_probability=0.0, time_offset_probability=1.0, rotation=None, translation=None, time_offset=PerturbationMagnitudeConfig(maximum=0.03)))
    augmented_window, augmented_truth, _, delta_tau = augmenter(base_window, generator=torch.Generator().manual_seed(123))

    for key in CALIBRATION_KEYS_3:
        torch.testing.assert_close(augmented_truth[key].time_offset, base_window.current_calibration[key].time_offset + delta_tau[key])

    for stream_key, stream in base_window.streams.items():
        calibration_key = base_window.metadata[stream_key].calibration_key
        expected_shift = delta_tau[calibration_key]
        augmented_stream = augmented_window.streams[stream_key]
        torch.testing.assert_close(augmented_stream.timestamps[stream.sample_mask], (stream.timestamps - expected_shift)[stream.sample_mask])
        if stream.interval_start_timestamps is not None:
            torch.testing.assert_close(augmented_stream.interval_start_timestamps[stream.sample_mask], (stream.interval_start_timestamps - expected_shift)[stream.sample_mask])
