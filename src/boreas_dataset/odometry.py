"""Independent Boreas scan odometry adapted from the supplied KAIST importer.

Retains its Open3D Tensor multiscale point-to-plane registration, robust kernels,
bounded recent-scan local map, motion initialization, CPU/CUDA selection and
optional independent-pair workers. Only raw XYZ scans enter registration: no
reference poses, IMU measurements or calibration matrices are read.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from tqdm import tqdm

from .data import LidarData
from .lidar import discover_lidar_scans, load_lidar_scan
from .timing import timestamps_ns_to_s

import pandas as pd


def compute_lidar_odometry(sequence_root: str | Path, *, source: str = "velodyne",
        time_start=None, time_end=None, scan_step: int = 1, max_scans: int | None = None,
        voxel_size_m: float = 0.5, max_correspondence_distance_m: float = 1.5,
        max_iterations: int = 50, local_map_scans: int = 5,
        voxel_scale_factors: tuple[float, ...] = (2.0, 1.0, 0.5),
        use_previous_motion_initialization: bool = True, robust_kernel: str | None = "tukey",
        robust_kernel_scale_factor: float = 1.0, bidirectional_check: bool = False,
        device: str = "cpu", cuda_device_id: int = 0, number_of_workers: int = 1,
        normal_radius_factor: float = 2.0, normal_max_nn: int = 30) -> LidarData:
    """Register current XYZ scans into a recent local map in the previous scan frame.

    No reference trajectory is used. Relative transforms map current to previous;
    timestamps are interval endpoints, consistent with this repository's renderer.
    Only the bounded recent preprocessed cloud history is kept in memory.
    """
    config = _validated_registration_config(voxel_size_m=voxel_size_m,
        max_correspondence_distance_m=max_correspondence_distance_m, max_iterations=max_iterations,
        local_map_scans=local_map_scans, voxel_scale_factors=voxel_scale_factors,
        use_previous_motion_initialization=use_previous_motion_initialization, robust_kernel=robust_kernel,
        robust_kernel_scale_factor=robust_kernel_scale_factor, bidirectional_check=bidirectional_check,
        device=device, cuda_device_id=cuda_device_id, normal_radius_factor=normal_radius_factor,
        normal_max_nn=normal_max_nn)
    config = replace(config, source=source)
    if not isinstance(scan_step, int) or scan_step < 1:
        raise ValueError("scan_step must be a positive integer")
    if max_scans is not None and (not isinstance(max_scans, int) or max_scans < 2):
        raise ValueError("max_scans must be at least two")
    if not isinstance(number_of_workers, int) or number_of_workers < 1:
        raise ValueError("number_of_workers must be positive")
    entries = discover_lidar_scans(sequence_root, source=source, time_start=time_start, time_end=time_end)[::scan_step]
    entries = entries if max_scans is None else entries[:max_scans]
    if len(entries) < 2:
        raise ValueError(f"Need at least two selected {source} scans")
    scans = [scan.path for scan in entries]
    ns = np.asarray([scan.timestamp_ns for scan in entries], dtype=np.int64)
    open3d = _import_open3d()
    open3d_device = _resolve_open3d_device(open3d, config)
    sequential = device == "cuda" or local_map_scans > 1 or use_previous_motion_initialization or bidirectional_check
    workers = number_of_workers
    if workers > 1 and sequential:
        warnings.warn("Local-map/motion-initialized odometry is sequential; using one worker. "
                      "Independent pairs permit --local-map-scans 1 --no-previous-motion-initialization on CPU.",
                      RuntimeWarning, stacklevel=2)
        workers = 1
    if workers == 1:
        relative, fitness, rmse, diagnostics = _register_scan_sequence_serial(
            scans, config=config, open3d=open3d, open3d_device=open3d_device)
    else:
        relative, fitness, rmse, diagnostics = _register_scan_pairs_parallel_independent(
            scans, config=config, number_of_workers=workers)
    if any(value <= 0 for value in fitness):
        raise RuntimeError("ICP found no correspondences for at least one interval; no odometry cache was accepted")
    times = timestamps_ns_to_s(ns)
    metadata = {**vars(config), "registration": "open3d_tensor_multiscale_point_to_plane_icp",
        "motion_source": "raw scan registration", "independent_odometry": True, "uses_ground_truth": False,
        "relative_pose_convention": "T_previous_current", "relative_timestamp_semantics": "end_scan_timestamp",
        "scan_timestamp_reference": entries[0].timestamp_reference, "scan_step": scan_step,
        "max_scans": max_scans, "time_start": time_start, "time_end": time_end,
        "scan_timestamps_ns": ns.tolist(), "number_of_workers_requested": number_of_workers,
        "number_of_workers_used": workers, "deskewed": False, "open3d_version": open3d.__version__,
        "interval_diagnostics": diagnostics, **_summarize_registration_diagnostics(diagnostics)}
    return LidarData(times[1:], np.stack(relative), times, [str(path) for path in scans],
                     np.asarray(fitness), np.asarray(rmse), metadata)


@dataclass(frozen=True)
class _RegistrationConfig:
    '''Internal tensor-registration configuration.'''
    voxel_size_m: float
    max_correspondence_distance_m: float
    max_iterations: int
    local_map_scans: int
    voxel_scale_factors: tuple[float, ...]
    use_previous_motion_initialization: bool
    robust_kernel: str | None
    robust_kernel_scale_factor: float
    bidirectional_check: bool
    device: str
    cuda_device_id: int
    normal_radius_factor: float
    normal_max_nn: int
    source: str = "velodyne"


@dataclass
class _CachedScan:
    '''One preprocessed scan retained in the bounded local-map cache.'''
    cloud: Any
    T_O_L: np.ndarray


def _validated_registration_config(*, voxel_size_m: float, max_correspondence_distance_m: float, max_iterations: int, local_map_scans: int, voxel_scale_factors: tuple[float, ...], use_previous_motion_initialization: bool, robust_kernel: str | None, robust_kernel_scale_factor: float, bidirectional_check: bool, device: str, cuda_device_id: int, normal_radius_factor: float, normal_max_nn: int) -> _RegistrationConfig:
    '''Validate public registration controls and normalize them once.'''
    voxel_size_m = float(voxel_size_m)
    max_correspondence_distance_m = float(max_correspondence_distance_m)
    max_iterations = int(max_iterations)
    local_map_scans = int(local_map_scans)
    robust_kernel_scale_factor = float(robust_kernel_scale_factor)
    cuda_device_id = int(cuda_device_id)
    normal_radius_factor = float(normal_radius_factor)
    normal_max_nn = int(normal_max_nn)
    normalized_device = str(device).strip().lower()
    if normalized_device not in {'cpu', 'cuda'}:
        raise ValueError("device must be either 'cpu' or 'cuda'")
    if cuda_device_id < 0:
        raise ValueError('cuda_device_id must be non-negative')
    if not np.isfinite(voxel_size_m) or voxel_size_m <= 0.0:
        raise ValueError('voxel_size_m must be finite and positive')
    if not np.isfinite(max_correspondence_distance_m) or max_correspondence_distance_m <= 0.0:
        raise ValueError('max_correspondence_distance_m must be finite and positive')
    if max_iterations <= 0:
        raise ValueError('max_iterations must be positive')
    if local_map_scans <= 0:
        raise ValueError('local_map_scans must be at least one')
    if not np.isfinite(robust_kernel_scale_factor) or robust_kernel_scale_factor <= 0.0:
        raise ValueError('robust_kernel_scale_factor must be finite and positive')
    if not np.isfinite(normal_radius_factor) or normal_radius_factor <= 0.0:
        raise ValueError('normal_radius_factor must be finite and positive')
    if normal_max_nn <= 0:
        raise ValueError('normal_max_nn must be positive')
    scale_factors = tuple(float(value) for value in voxel_scale_factors)
    if not scale_factors:
        raise ValueError('voxel_scale_factors must contain at least one value')
    if any(not np.isfinite(value) or value <= 0.0 for value in scale_factors):
        raise ValueError('voxel_scale_factors must contain only finite positive values')
    if any(scale_factors[index] <= scale_factors[index + 1] for index in range(len(scale_factors) - 1)):
        raise ValueError('voxel_scale_factors must be strictly decreasing from coarse to fine')
    normalized_kernel = None if robust_kernel is None else str(robust_kernel).strip().lower()
    if normalized_kernel not in {None, 'tukey', 'huber', 'cauchy', 'l2'}:
        raise ValueError("robust_kernel must be one of None, 'l2', 'tukey', 'huber', or 'cauchy'")
    return _RegistrationConfig(voxel_size_m=voxel_size_m, max_correspondence_distance_m=max_correspondence_distance_m, max_iterations=max_iterations, local_map_scans=local_map_scans, voxel_scale_factors=scale_factors, use_previous_motion_initialization=bool(use_previous_motion_initialization), robust_kernel=normalized_kernel, robust_kernel_scale_factor=robust_kernel_scale_factor, bidirectional_check=bool(bidirectional_check), device=normalized_device, cuda_device_id=cuda_device_id, normal_radius_factor=normal_radius_factor, normal_max_nn=normal_max_nn)


def _resolve_open3d_device(open3d, config: _RegistrationConfig):
    '''Resolve ``CPU:0`` or ``CUDA:n`` and fail early when requested CUDA is unavailable.'''
    if config.device == 'cpu':
        return open3d.core.Device('CPU:0')
    if not open3d.core.cuda.is_available():
        raise RuntimeError("device='cuda' was requested, but Open3D reports that CUDA is unavailable")
    device_count = int(open3d.core.cuda.device_count())
    if config.cuda_device_id >= device_count:
        raise ValueError(f'cuda_device_id={config.cuda_device_id} is invalid because Open3D reports {device_count} CUDA device(s)')
    return open3d.core.Device(f'CUDA:{config.cuda_device_id}')


def _cache_voxel_size(config: _RegistrationConfig) -> float:
    '''Return the finest voxel size used to bound cached scan memory.'''
    return float(config.voxel_size_m * min(config.voxel_scale_factors))


def _import_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError('open3d is required for LiDAR ICP loading. Install it in the notebook environment with `pip install open3d`.') from exc
    return o3d


def _preprocess_scan_tensor(open3d, path: Path, open3d_device, config: _RegistrationConfig):
    '''Load one raw scan, move it to the selected device, downsample once at the finest requested scale, and estimate normals once.'''
    xyz = load_lidar_scan(path, source=config.source)[:, :3]
    positions = open3d.core.Tensor(np.ascontiguousarray(xyz, dtype=np.float32), dtype=open3d.core.Dtype.Float32, device=open3d_device)
    cloud = open3d.t.geometry.PointCloud(positions)
    cache_voxel_size = _cache_voxel_size(config)
    cloud = cloud.voxel_down_sample(cache_voxel_size)
    if int(cloud.point.positions.shape[0]) == 0:
        raise ValueError(f'Voxel downsampling produced an empty point cloud: {path}')
    cloud.estimate_normals(max_nn=config.normal_max_nn, radius=max(config.normal_radius_factor * cache_voxel_size, 1e-3))
    return cloud


def _transform_tensor_cloud(open3d, cloud, transformation: np.ndarray, open3d_device):
    '''Clone one cached tensor cloud and transform its points and normals on the selected device.'''
    transformed = cloud.clone()
    transformation_tensor = open3d.core.Tensor(np.asarray(transformation, dtype=np.float32), dtype=open3d.core.Dtype.Float32, device=open3d_device)
    transformed.transform(transformation_tensor)
    return transformed


def _build_local_map_tensor(open3d, history: deque[_CachedScan], T_O_L_reference: np.ndarray, open3d_device, config: _RegistrationConfig):
    '''Build a bounded target map from only the latest cached scans, expressed in the immediately previous LiDAR frame.'''

    if not history:
        raise ValueError('Cannot build a local map without previous scans')

    T_Lreference_O = np.linalg.inv(np.asarray(T_O_L_reference, dtype=float))

    transformed_positions = []
    transformed_normals = []

    for cached_scan in history:
        T_Lreference_Lscan = T_Lreference_O @ np.asarray(cached_scan.T_O_L, dtype=float)

        transformed_cloud = _transform_tensor_cloud(open3d, cached_scan.cloud, T_Lreference_Lscan, open3d_device)

        transformed_positions.append(transformed_cloud.point.positions)
        transformed_normals.append(transformed_cloud.point.normals)

    ##################################################
    # Open3D core.concatenate has special behavior for a
    # single input tensor and can flatten an (N, 3) tensor.
    # Preserve the tensor directly when the local map contains
    # only one scan.
    ##################################################

    if len(transformed_positions) == 1:
        positions = transformed_positions[0]
        normals = transformed_normals[0]
    else:
        positions = open3d.core.concatenate(transformed_positions, axis=0)
        normals = open3d.core.concatenate(transformed_normals, axis=0)

    if len(positions.shape) != 2 or int(positions.shape[1]) != 3:
        raise RuntimeError(f'Local-map positions have invalid shape {positions.shape}; expected (N, 3)')

    if len(normals.shape) != 2 or int(normals.shape[1]) != 3:
        raise RuntimeError(f'Local-map normals have invalid shape {normals.shape}; expected (N, 3)')

    if int(positions.shape[0]) != int(normals.shape[0]):
        raise RuntimeError(f'Local-map positions and normals have different lengths: {positions.shape[0]} vs {normals.shape[0]}')

    local_map = open3d.t.geometry.PointCloud({'positions': positions, 'normals': normals})

    local_map = local_map.voxel_down_sample(_cache_voxel_size(config))

    if int(local_map.point.positions.shape[0]) == 0:
        raise ValueError('Local-map voxel downsampling produced an empty point cloud')

    return local_map


def _iterations_per_scale(total_iterations: int, number_of_scales: int) -> tuple[int, ...]:
    '''Split one total ICP iteration budget from coarse to fine.'''
    if total_iterations < number_of_scales:
        raise ValueError('max_iterations must be at least the number of voxel scales')
    if number_of_scales == 1:
        return (int(total_iterations),)
    weights = np.linspace(number_of_scales, 1.0, number_of_scales)
    weights /= np.sum(weights)
    allocations = np.maximum(1, np.floor(weights * int(total_iterations)).astype(int))
    difference = int(total_iterations) - int(np.sum(allocations))
    allocation_index = 0
    while difference != 0:
        index = allocation_index % number_of_scales
        if difference > 0:
            allocations[index] += 1
            difference -= 1
        elif allocations[index] > 1:
            allocations[index] -= 1
            difference += 1
        allocation_index += 1
    return tuple(int(value) for value in allocations)


def _tensor_robust_kernel(open3d, config: _RegistrationConfig):
    '''Create the tensor robust kernel used by point-to-plane ICP.'''
    robust_kernel_module = open3d.t.pipelines.registration.robust_kernel
    method_map = {'l2': robust_kernel_module.RobustKernelMethod.L2Loss, 'huber': robust_kernel_module.RobustKernelMethod.HuberLoss, 'cauchy': robust_kernel_module.RobustKernelMethod.CauchyLoss, 'tukey': robust_kernel_module.RobustKernelMethod.TukeyLoss}
    method = robust_kernel_module.RobustKernelMethod.L2Loss if config.robust_kernel is None else method_map[config.robust_kernel]
    kernel_scale_m = max(config.robust_kernel_scale_factor * config.voxel_size_m, 1e-6)
    return robust_kernel_module.RobustKernel(method, kernel_scale_m, 1.0)


def _register_tensor_multiscale(open3d, *, source, target, initial_transform: np.ndarray, config: _RegistrationConfig) -> tuple[np.ndarray, float, float, dict[str, float]]:
    '''Register source into target with Open3D Tensor multi-scale point-to-plane ICP on the source/target device.'''
    registration = open3d.t.pipelines.registration
    voxel_sizes = open3d.utility.DoubleVector([config.voxel_size_m * scale_factor for scale_factor in config.voxel_scale_factors])
    max_correspondence_distances = open3d.utility.DoubleVector([config.max_correspondence_distance_m * scale_factor for scale_factor in config.voxel_scale_factors])
    criteria_list = [registration.ICPConvergenceCriteria(max_iteration=iterations) for iterations in _iterations_per_scale(config.max_iterations, len(config.voxel_scale_factors))]
    estimation = registration.TransformationEstimationPointToPlane(_tensor_robust_kernel(open3d, config))
    initial_transform_tensor = open3d.core.Tensor(np.asarray(initial_transform, dtype=np.float64), dtype=open3d.core.Dtype.Float64, device=open3d.core.Device('CPU:0'))
    result = registration.multi_scale_icp(source, target, voxel_sizes, criteria_list, max_correspondence_distances, initial_transform_tensor, estimation)
    transformation = np.asarray(result.transformation.cpu().numpy(), dtype=float)
    if transformation.shape != (4, 4) or not np.all(np.isfinite(transformation)):
        raise RuntimeError('Open3D Tensor ICP did not produce a finite 4x4 transformation')
    finest_correspondence_distance = float(config.max_correspondence_distance_m * min(config.voxel_scale_factors))
    information_function = getattr(registration, 'get_information_matrix', None)
    if information_function is None:
        information_function = getattr(registration, 'get_information_matrix_from_point_clouds')
    information_tensor = information_function(source, target, finest_correspondence_distance, result.transformation)
    information_matrix = np.asarray(information_tensor.cpu().numpy(), dtype=float)
    information_matrix = 0.5 * (information_matrix + information_matrix.T)
    information_eigenvalues = np.linalg.eigvalsh(information_matrix)
    positive_eigenvalues = information_eigenvalues[information_eigenvalues > 1e-12]
    if positive_eigenvalues.size == 0:
        information_min_eigenvalue = 0.0
        information_condition_number = np.inf
    else:
        information_min_eigenvalue = float(np.min(positive_eigenvalues))
        information_condition_number = float(np.max(positive_eigenvalues) / information_min_eigenvalue)
    diagnostics = {'information_min_eigenvalue': information_min_eigenvalue, 'information_condition_number': information_condition_number}
    return transformation, float(result.fitness), float(result.inlier_rmse), diagnostics


def _bidirectional_closure_diagnostics(open3d, *, current_cloud, target_map, forward_transform: np.ndarray, config: _RegistrationConfig) -> dict[str, float]:
    '''Independently solve the reverse registration and report SE(3) closure; this does not modify the accepted forward odometry.'''
    reverse_transform, _, _, _ = _register_tensor_multiscale(open3d, source=target_map, target=current_cloud, initial_transform=np.linalg.inv(forward_transform), config=config)
    closure = forward_transform @ reverse_transform
    translation_closure_m = float(np.linalg.norm(closure[:3, 3]))
    trace_value = np.clip((np.trace(closure[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
    rotation_closure_deg = float(np.rad2deg(np.arccos(trace_value)))
    return {'bidirectional_translation_closure_m': translation_closure_m, 'bidirectional_rotation_closure_deg': rotation_closure_deg}


def _register_scan_sequence_serial(scans: list[Path], *, config: _RegistrationConfig, open3d, open3d_device) -> tuple[list[np.ndarray], list[float], list[float], list[dict[str, float]]]:
    '''Register a scan sequence while retaining only the latest local-map scans on the chosen Open3D device.'''
    relative_poses: list[np.ndarray] = []
    fitness: list[float] = []
    inlier_rmse: list[float] = []
    diagnostics: list[dict[str, float]] = []
    first_cloud = _preprocess_scan_tensor(open3d, scans[0], open3d_device, config)
    T_O_L_previous = np.eye(4)
    previous_relative_pose: np.ndarray | None = None
    history: deque[_CachedScan] = deque(maxlen=config.local_map_scans)
    history.append(_CachedScan(cloud=first_cloud, T_O_L=T_O_L_previous.copy()))
    for scan_path in tqdm(scans[1:], desc=f'LiDAR ICP [{config.device}]'):
        current_cloud = _preprocess_scan_tensor(open3d, scan_path, open3d_device, config)
        target_map = _build_local_map_tensor(open3d, history, T_O_L_previous, open3d_device, config)
        initial_transform = previous_relative_pose if config.use_previous_motion_initialization and previous_relative_pose is not None else np.eye(4)
        relative_pose, pair_fitness, pair_rmse, pair_diagnostics = _register_tensor_multiscale(open3d, source=current_cloud, target=target_map, initial_transform=initial_transform, config=config)
        if config.bidirectional_check:
            pair_diagnostics.update(_bidirectional_closure_diagnostics(open3d, current_cloud=current_cloud, target_map=target_map, forward_transform=relative_pose, config=config))
        relative_poses.append(relative_pose)
        fitness.append(pair_fitness)
        inlier_rmse.append(pair_rmse)
        diagnostics.append(pair_diagnostics)
        T_O_L_current = T_O_L_previous @ relative_pose
        history.append(_CachedScan(cloud=current_cloud, T_O_L=T_O_L_current.copy()))
        previous_relative_pose = relative_pose
        T_O_L_previous = T_O_L_current
    if config.device == 'cuda':
        open3d.core.cuda.synchronize(open3d_device)
    return relative_poses, fitness, inlier_rmse, diagnostics


def _scan_pair_chunks(number_of_scans: int, number_of_workers: int) -> list[tuple[int, int]]:
    '''Split consecutive independent pair indices into overlapping worker chunks.'''
    number_of_pairs = number_of_scans - 1
    if number_of_pairs <= 0:
        return []
    worker_count = min(max(int(number_of_workers), 1), number_of_pairs)
    boundaries = np.linspace(0, number_of_pairs, worker_count + 1, dtype=int)
    return [(int(start), int(stop)) for start, stop in zip(boundaries[:-1], boundaries[1:]) if start < stop]


def _register_scan_chunk_independent(payload: tuple[list[str], int, _RegistrationConfig]) -> tuple[int, list[np.ndarray], list[float], list[float], list[dict[str, float]]]:
    '''Register one CPU chunk as independent current-to-previous scan pairs.'''
    scan_path_strings, first_pair_index, config = payload
    open3d = _import_open3d()
    open3d_device = open3d.core.Device('CPU:0')
    scan_paths = [Path(path) for path in scan_path_strings]
    relative_poses: list[np.ndarray] = []
    fitness: list[float] = []
    inlier_rmse: list[float] = []
    diagnostics: list[dict[str, float]] = []
    previous_cloud = _preprocess_scan_tensor(open3d, scan_paths[0], open3d_device, config)
    for scan_path in scan_paths[1:]:
        current_cloud = _preprocess_scan_tensor(open3d, scan_path, open3d_device, config)
        relative_pose, pair_fitness, pair_rmse, pair_diagnostics = _register_tensor_multiscale(open3d, source=current_cloud, target=previous_cloud, initial_transform=np.eye(4), config=config)
        relative_poses.append(relative_pose)
        fitness.append(pair_fitness)
        inlier_rmse.append(pair_rmse)
        diagnostics.append(pair_diagnostics)
        previous_cloud = current_cloud
    return first_pair_index, relative_poses, fitness, inlier_rmse, diagnostics


def _register_scan_pairs_parallel_independent(scans: list[Path], *, config: _RegistrationConfig, number_of_workers: int) -> tuple[list[np.ndarray], list[float], list[float], list[dict[str, float]]]:
    '''Register independent CPU current-to-previous pairs in process chunks.'''
    if config.device != 'cpu':
        raise ValueError('Independent multiprocessing is supported only with device="cpu"')
    chunks = _scan_pair_chunks(len(scans), number_of_workers)
    payloads = [([str(path) for path in scans[pair_start:pair_stop + 1]], pair_start, config) for pair_start, pair_stop in chunks]
    process_context = mp.get_context('spawn')
    chunk_results = []
    with ProcessPoolExecutor(max_workers=len(payloads), mp_context=process_context) as executor:
        futures = [executor.submit(_register_scan_chunk_independent, payload) for payload in payloads]
        for future in tqdm(as_completed(futures), total=len(futures), desc='LiDAR ICP chunks [cpu]'):
            chunk_results.append(future.result())
    chunk_results.sort(key=lambda result: result[0])
    relative_poses = [pose for _, chunk_poses, _, _, _ in chunk_results for pose in chunk_poses]
    fitness = [value for _, _, chunk_fitness, _, _ in chunk_results for value in chunk_fitness]
    inlier_rmse = [value for _, _, _, chunk_rmse, _ in chunk_results for value in chunk_rmse]
    diagnostics = [value for _, _, _, _, chunk_diagnostics in chunk_results for value in chunk_diagnostics]
    expected_pair_count = len(scans) - 1
    if len(relative_poses) != expected_pair_count:
        raise RuntimeError(f'Parallel ICP produced an unexpected number of scan pairs: {len(relative_poses)} instead of {expected_pair_count}')
    return relative_poses, fitness, inlier_rmse, diagnostics


def _summarize_registration_diagnostics(diagnostics: list[dict[str, float]]) -> dict[str, float | None]:
    '''Store compact diagnostic summaries without bloating the normalized CSV.'''
    if not diagnostics:
        return {}
    summary: dict[str, float | None] = {}
    diagnostic_names = sorted({name for pair_diagnostics in diagnostics for name in pair_diagnostics})
    for diagnostic_name in diagnostic_names:
        values = np.asarray([pair_diagnostics.get(diagnostic_name, np.nan) for pair_diagnostics in diagnostics], dtype=float)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            summary[f'{diagnostic_name}_median'] = None
            summary[f'{diagnostic_name}_p90'] = None
        else:
            summary[f'{diagnostic_name}_median'] = float(np.median(finite_values))
            summary[f'{diagnostic_name}_p90'] = float(np.percentile(finite_values, 90))
    return summary


def _json_compatible(value: Any) -> Any:
    '''Convert common NumPy and path objects into JSON-compatible values.'''
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return list(value)
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def save_lidar_data_csv(lidar_data: LidarData, filepath: str | Path) -> Path:
    '''Save a normalized ``LidarData`` entity into one CSV without raw scan paths.'''
    output_path = Path(filepath).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    interval_timestamps = np.asarray(lidar_data.timestamps_s, dtype=float).reshape(-1)
    relative_poses = np.asarray(lidar_data.relative_poses_se3, dtype=float)
    scan_timestamps = np.asarray(lidar_data.scan_timestamps_s, dtype=float).reshape(-1)
    fitness = np.asarray(lidar_data.fitness, dtype=float).reshape(-1)
    inlier_rmse = np.asarray(lidar_data.inlier_rmse, dtype=float).reshape(-1)
    number_of_intervals = interval_timestamps.size
    if number_of_intervals == 0:
        raise ValueError('lidar_data must contain at least one relative-pose interval')
    if relative_poses.shape != (number_of_intervals, 4, 4):
        raise ValueError('relative_poses_se3 must have shape (N, 4, 4), where N is the number of interval timestamps')
    if scan_timestamps.shape != (number_of_intervals + 1,):
        raise ValueError('scan_timestamps_s must have shape (N + 1,)')
    if fitness.shape != (number_of_intervals,):
        raise ValueError('fitness must have shape (N,)')
    if inlier_rmse.shape != (number_of_intervals,):
        raise ValueError('inlier_rmse must have shape (N,)')
    arrays_to_check = (interval_timestamps, relative_poses, scan_timestamps, fitness, inlier_rmse)
    if not all(np.all(np.isfinite(array)) for array in arrays_to_check):
        raise ValueError('lidar_data contains non-finite numerical values')
    metadata_json = json.dumps(dict(lidar_data.metadata), default=_json_compatible, ensure_ascii=False, separators=(',', ':'))
    pose_columns = [f'T_{row}{column}' for row in range(4) for column in range(4)]
    records: list[dict[str, object]] = []
    for interval_index in range(number_of_intervals):
        flattened_pose = relative_poses[interval_index].reshape(-1)
        record: dict[str, object] = {'interval_index': interval_index, 'timestamp_s': interval_timestamps[interval_index], 'scan_start_timestamp_s': scan_timestamps[interval_index], 'scan_end_timestamp_s': scan_timestamps[interval_index + 1], 'fitness': fitness[interval_index], 'inlier_rmse': inlier_rmse[interval_index]}
        for column_name, value in zip(pose_columns, flattened_pose):
            record[column_name] = float(value)
        record['metadata_json'] = metadata_json if interval_index == 0 else ''
        records.append(record)
    column_order = ['interval_index', 'timestamp_s', 'scan_start_timestamp_s', 'scan_end_timestamp_s', 'fitness', 'inlier_rmse', *pose_columns, 'metadata_json']
    pd.DataFrame(records, columns=column_order).to_csv(output_path, index=False, float_format='%.17g')
    return output_path


def load_lidar_data_csv(filepath: str | Path) -> LidarData:
    '''Load a normalized ``LidarData`` cache created by ``save_lidar_data_csv``.'''
    source_path = Path(filepath).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f'LiDAR CSV does not exist: {source_path}')
    table = pd.read_csv(source_path)
    pose_columns = [f'T_{row}{column}' for row in range(4) for column in range(4)]
    required_columns = ['interval_index', 'timestamp_s', 'scan_start_timestamp_s', 'scan_end_timestamp_s', 'fitness', 'inlier_rmse', *pose_columns, 'metadata_json']
    missing_columns = [column for column in required_columns if column not in table.columns]
    if missing_columns:
        raise ValueError(f'LiDAR CSV is missing columns: {missing_columns}')
    if table.empty:
        raise ValueError('LiDAR CSV contains no interval rows')
    table = table.sort_values('interval_index', kind='stable').reset_index(drop=True)
    expected_indices = np.arange(table.shape[0], dtype=int)
    stored_indices = table['interval_index'].to_numpy(dtype=int)
    if not np.array_equal(stored_indices, expected_indices):
        raise ValueError('interval_index must contain consecutive values starting at zero')
    interval_timestamps = table['timestamp_s'].to_numpy(dtype=float)
    scan_start_timestamps = table['scan_start_timestamp_s'].to_numpy(dtype=float)
    scan_end_timestamps = table['scan_end_timestamp_s'].to_numpy(dtype=float)
    if not np.allclose(scan_start_timestamps[1:], scan_end_timestamps[:-1], rtol=0.0, atol=1e-12):
        raise ValueError('Consecutive interval rows contain inconsistent scan timestamps')
    relative_poses = table.loc[:, pose_columns].to_numpy(dtype=float).reshape(-1, 4, 4)
    scan_timestamps = np.concatenate([scan_start_timestamps[:1], scan_end_timestamps])
    fitness = table['fitness'].to_numpy(dtype=float)
    inlier_rmse = table['inlier_rmse'].to_numpy(dtype=float)
    numerical_arrays = (interval_timestamps, scan_timestamps, relative_poses, fitness, inlier_rmse)
    if not all(np.all(np.isfinite(array)) for array in numerical_arrays):
        raise ValueError('LiDAR CSV contains non-finite numerical values')
    metadata_json = '{}'
    for value in table['metadata_json']:
        if isinstance(value, str) and value.strip():
            metadata_json = value
            break
    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        raise ValueError('metadata_json contains invalid JSON') from exc
    if not isinstance(metadata, dict):
        raise ValueError('metadata_json must encode a JSON object')
    return LidarData(timestamps_s=interval_timestamps, relative_poses_se3=relative_poses, scan_timestamps_s=scan_timestamps, source_scan_paths=[], fitness=fitness, inlier_rmse=inlier_rmse, metadata=metadata)
