"""LiDAR scan loading and pairwise ICP odometry for New College data."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from pathlib import Path
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from tqdm import tqdm
import os
import numpy as np

from .data import LidarData

SAFE_PARALLELIZM = True
if SAFE_PARALLELIZM:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

def discover_lidar_archives(dataset_root: str | Path) -> list[Path]:
    """Find Ouster zip archives that contain raw PCD scans.

    Args:
        dataset_root: Path to the New College experiment root.

    Returns:
        Sorted list of zip files below ``raw_format/ouster_zip_files`` when that
        directory exists, otherwise all zip files below the dataset root.
    """
    root = Path(dataset_root).expanduser()
    ouster_dir = root / "raw_format" / "ouster_zip_files"
    search_root = ouster_dir if ouster_dir.exists() else root
    return sorted(path for path in search_root.rglob("*.zip") if path.is_file())


def discover_lidar_scans(
    dataset_root: str | Path,
    *,
    extract_if_needed: bool = True,
) -> list[Path]:
    """Find raw PCD scans, extracting zip archives in place only when needed.

    Args:
        dataset_root: Path to the New College experiment root.
        extract_if_needed: If true, extract PCD files from archives beside their
            source zip when no already-extracted scans are found.

    Returns:
        Sorted list of PCD scan paths.
    """
    root = Path(dataset_root).expanduser()
    ouster_dir = root / "raw_format" / "ouster_zip_files"
    search_root = ouster_dir if ouster_dir.exists() else root
    scans = sorted(path for path in search_root.rglob("*.pcd") if path.is_file())
    if scans or not extract_if_needed:
        return scans

    # Open3D reads PCD from paths, so extract each archive into a sibling folder.
    for archive in discover_lidar_archives(root):
        target_dir = archive.with_suffix("")
        target_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            pcd_members = [name for name in zf.namelist() if name.lower().endswith(".pcd")]
            existing = [target_dir / Path(name).name for name in pcd_members]
            if existing and all(path.exists() for path in existing):
                continue
            for member in pcd_members:
                output_path = target_dir / Path(member).name
                if output_path.exists():
                    continue
                with zf.open(member) as src, output_path.open("wb") as dst:
                    dst.write(src.read())
    return sorted(path for path in search_root.rglob("*.pcd") if path.is_file())


def sort_by_filename(scans: list[Path]) -> list[Path]:
    """Sort New College Ouster scans by filename timestamp.

    The extracted dataset can be split across several directories, with each
    directory containing scans spread over approximately the same time span.
    Sorting complete paths therefore groups scans by directory and produces a
    saw-like timestamp sequence. This function ignores the parent directories
    and orders all scans globally by the integer timestamp encoded in
    ``cloud_<seconds>_<nanoseconds>.pcd``.

    Args:
        scans: LiDAR scan paths to sort.

    Returns:
        New list ordered chronologically by ``(seconds, nanoseconds)``. Paths
        with identical timestamps are ordered deterministically by full path.

    Raises:
        ValueError: If any filename does not follow the expected New College
            Ouster naming convention.
    """

    def timestamp_key(path: Path) -> tuple[int, int, str]:
        components = _timestamp_components_from_path(path)
        if components is None:
            raise ValueError(
                "Could not parse LiDAR timestamp from filename "
                f"{path.name!r}; expected cloud_<seconds>_<nanoseconds>.pcd"
            )
        seconds, nanoseconds = components
        return seconds, nanoseconds, str(path)

    return sorted(scans, key=timestamp_key)


def load_lidar_relative_poses(
    dataset_root: str | Path,
    *,
    voxel_size_m: float = 0.5,
    max_correspondence_distance_m: float = 1.5,
    max_iterations: int = 50,
    max_scans: int | None = None,
    scan_period_s: float | None = None,
    extract_if_needed: bool = True,
    number_of_workers: int = 1,
) -> LidarData:
    """Compute pairwise LiDAR relative poses from ordered PCD scans using ICP.

    Args:
        dataset_root: Path to the New College experiment root.
        voxel_size_m: Voxel size for scan downsampling in meters.
        max_correspondence_distance_m: ICP correspondence gate in meters.
        max_iterations: Maximum ICP iterations per scan pair.
        max_scans: Optional cap for quick notebook checks.
        scan_period_s: Fallback scan period used when timestamps cannot be
            parsed from filenames or metadata.
        extract_if_needed: Extract PCD files beside source zips if needed.

    Returns:
        ``LidarData`` containing interval midpoint timestamps and relative
        ``SE(3)`` transforms from scan ``i`` to ``i + 1``.
    """
    open3d = _import_open3d()
    scans = discover_lidar_scans(dataset_root, extract_if_needed=extract_if_needed)
    scans = sort_by_filename(scans)
    if max_scans is not None:
        scans = scans[:max_scans]
    if len(scans) < 2:
        raise ValueError("At least two PCD scans are required to compute relative poses.")

    scan_timestamps_s = _timestamps_from_paths(scans, scan_period_s=scan_period_s)

    if number_of_workers <= 1:
        relative_poses: list[np.ndarray] = []
        fitness: list[float] = []
        inlier_rmse: list[float] = []

        previous_cloud = _load_and_preprocess_cloud(
            open3d,
            scans[0],
            voxel_size_m,
        )

        for scan_path in tqdm(scans[1:], desc="LiDAR ICP"):
            current_cloud = _load_and_preprocess_cloud(
                open3d,
                scan_path,
                voxel_size_m,
            )

            result = _register_pair(
                open3d,
                source=current_cloud,
                target=previous_cloud,
                max_correspondence_distance_m=(
                    max_correspondence_distance_m
                ),
                max_iterations=max_iterations,
            )

            relative_poses.append(
                np.asarray(result.transformation, dtype=float)
            )
            fitness.append(float(result.fitness))
            inlier_rmse.append(float(result.inlier_rmse))

            previous_cloud = current_cloud
    else:
        relative_poses, fitness, inlier_rmse = (
            _register_scan_pairs_parallel(
                scans,
                voxel_size_m=voxel_size_m,
                max_correspondence_distance_m=(
                    max_correspondence_distance_m
                ),
                max_iterations=max_iterations,
                number_of_workers=number_of_workers,
            )
        )

    poses = np.stack(relative_poses)
    interval_timestamps_s = 0.5 * (scan_timestamps_s[:-1] + scan_timestamps_s[1:])
    return LidarData(
        timestamps_s=interval_timestamps_s,
        relative_poses_se3=poses,
        scan_timestamps_s=scan_timestamps_s,
        source_scan_paths=[str(path) for path in scans],
        fitness=np.asarray(fitness),
        inlier_rmse=np.asarray(inlier_rmse),
        metadata={
            "voxel_size_m": voxel_size_m,
            "max_correspondence_distance_m": max_correspondence_distance_m,
            "max_iterations": max_iterations,
        },
    )


def _import_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "open3d is required for New College LiDAR ICP loading. "
            "Install it in the notebook environment with `pip install open3d`."
        ) from exc
    return o3d


def _load_and_preprocess_cloud(open3d, path: Path, voxel_size_m: float):
    cloud = open3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f"Point cloud is empty: {path}")

    # Downsample and estimate normals for point-to-plane ICP.
    if voxel_size_m > 0.0:
        cloud = cloud.voxel_down_sample(voxel_size_m)
    radius = max(voxel_size_m * 2.0, 1.0)
    cloud.estimate_normals(
        open3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    return cloud


def _register_pair(
    open3d,
    *,
    source,
    target,
    max_correspondence_distance_m: float,
    max_iterations: int,
):
    estimation = open3d.pipelines.registration.TransformationEstimationPointToPlane()
    criteria = open3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=int(max_iterations)
    )
    return open3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance_m,
        np.eye(4),
        estimation,
        criteria,
    )


def _timestamps_from_paths(scans: list[Path], *, scan_period_s: float | None) -> np.ndarray:
    parsed = np.asarray([_timestamp_from_path(path) for path in scans], dtype=float)
    if np.all(np.isfinite(parsed)) and np.all(np.diff(parsed) > 0.0):
        return parsed
    if scan_period_s is None:
        raise ValueError(
            "Could not parse strictly increasing timestamps from PCD filenames. "
            "Set scan_period_s in the notebook configuration."
        )
    return np.arange(len(scans), dtype=float) * float(scan_period_s)


def _timestamp_components_from_path(path: Path) -> tuple[int, int] | None:
    """Return integer timestamp components encoded in a scan filename.

    Args:
        path: Path whose stem should follow
            ``cloud_<seconds>_<nanoseconds>``.

    Returns:
        ``(seconds, nanoseconds)`` when parsing succeeds, otherwise ``None``.
    """

    match = re.fullmatch(r"cloud_(\d+)_(\d{1,9})", path.stem)
    if match is None:
        return None

    seconds = int(match.group(1))
    nanoseconds = int(match.group(2))
    if nanoseconds >= 1_000_000_000:
        return None
    return seconds, nanoseconds


def _timestamp_from_path(path: Path) -> float:
    """Parse ``cloud_<seconds>_<nanoseconds>.pcd`` into Unix seconds.

    Args:
        path: Path whose stem follows the New College Ouster naming convention,
            for example ``cloud_1583836591_582553088``.

    Returns:
        Timestamp in seconds as ``seconds + nanoseconds * 1e-9``. Returns
        ``numpy.nan`` when the filename does not match the expected template or
        contains an invalid nanosecond field.
    """

    components = _timestamp_components_from_path(path)
    if components is None:
        return np.nan

    seconds, nanoseconds = components
    return float(seconds) + float(nanoseconds) * 1e-9

def _scan_pair_chunks(
    number_of_scans: int,
    number_of_workers: int,
) -> list[tuple[int, int]]:
    """Split scan-pair indices into contiguous chunks.

    A returned pair ``(start, stop)`` represents ICP pairs with indices

    ``start, start + 1, ..., stop - 1``.

    The worker processing that range receives scans
    ``scans[start:stop + 1]``, so adjacent chunks overlap by one scan.

    Args:
        number_of_scans: Total number of chronologically sorted scans.
        number_of_workers: Requested number of worker processes.

    Returns:
        Nonempty half-open pair-index ranges.
    """
    number_of_pairs = number_of_scans - 1
    if number_of_pairs <= 0:
        return []

    worker_count = min(
        max(int(number_of_workers), 1),
        number_of_pairs,
    )

    boundaries = np.linspace(
        0,
        number_of_pairs,
        worker_count + 1,
        dtype=int,
    )

    return [
        (int(start), int(stop))
        for start, stop in zip(boundaries[:-1], boundaries[1:])
        if start < stop
    ]


def _register_scan_chunk(
    payload: tuple[
        list[str],
        int,
        float,
        float,
        int,
    ],
) -> tuple[
    int,
    list[np.ndarray],
    list[float],
    list[float],
]:
    """Load and register one contiguous chunk of LiDAR scans.

    Args:
        payload: Tuple containing scan paths, global first-pair index, voxel
            size, correspondence distance, and maximum ICP iterations.

    Returns:
        Global first-pair index followed by relative poses, fitness values, and
        inlier RMSE values for this chunk.
    """
    (
        scan_path_strings,
        first_pair_index,
        voxel_size_m,
        max_correspondence_distance_m,
        max_iterations,
    ) = payload

    open3d = _import_open3d()
    scan_paths = [Path(path) for path in scan_path_strings]

    relative_poses: list[np.ndarray] = []
    fitness: list[float] = []
    inlier_rmse: list[float] = []

    ##################################################
    # Load the first scan shared with the preceding pair
    ##################################################

    previous_cloud = _load_and_preprocess_cloud(
        open3d,
        scan_paths[0],
        voxel_size_m,
    )

    ##################################################
    # Register consecutive scans inside this chunk
    ##################################################

    for scan_path in scan_paths[1:]:
        current_cloud = _load_and_preprocess_cloud(
            open3d,
            scan_path,
            voxel_size_m,
        )

        result = _register_pair(
            open3d,
            source=current_cloud,
            target=previous_cloud,
            max_correspondence_distance_m=(
                max_correspondence_distance_m
            ),
            max_iterations=max_iterations,
        )

        relative_poses.append(
            np.asarray(result.transformation, dtype=float)
        )
        fitness.append(float(result.fitness))
        inlier_rmse.append(float(result.inlier_rmse))

        previous_cloud = current_cloud

    return (
        first_pair_index,
        relative_poses,
        fitness,
        inlier_rmse,
    )


def _register_scan_pairs_parallel(
    scans: list[Path],
    *,
    voxel_size_m: float,
    max_correspondence_distance_m: float,
    max_iterations: int,
    number_of_workers: int,
) -> tuple[list[np.ndarray], list[float], list[float]]:
    """Register all consecutive scan pairs using overlapping process chunks.

    Args:
        scans: Chronologically sorted scan paths.
        voxel_size_m: Voxel size used during preprocessing.
        max_correspondence_distance_m: ICP correspondence threshold.
        max_iterations: Maximum ICP iterations per pair.
        number_of_workers: Number of worker processes.

    Returns:
        Ordered relative poses, fitness values, and inlier RMSE values.
    """
    chunks = _scan_pair_chunks(
        len(scans),
        number_of_workers,
    )

    payloads = [
        (
            [
                str(path)
                for path in scans[pair_start : pair_stop + 1]
            ],
            pair_start,
            voxel_size_m,
            max_correspondence_distance_m,
            max_iterations,
        )
        for pair_start, pair_stop in chunks
    ]

    ##################################################
    # Use fresh worker processes for Open3D instances
    ##################################################

    process_context = mp.get_context("spawn")
    chunk_results = []

    with ProcessPoolExecutor(
        max_workers=len(payloads),
        mp_context=process_context,
    ) as executor:
        futures = [
            executor.submit(_register_scan_chunk, payload)
            for payload in payloads
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="LiDAR ICP chunks",
        ):
            chunk_results.append(future.result())

    ##################################################
    # Restore global chronological pair order
    ##################################################

    chunk_results.sort(key=lambda result: result[0])

    relative_poses = [
        pose
        for _, chunk_poses, _, _ in chunk_results
        for pose in chunk_poses
    ]
    fitness = [
        value
        for _, _, chunk_fitness, _ in chunk_results
        for value in chunk_fitness
    ]
    inlier_rmse = [
        value
        for _, _, _, chunk_rmse in chunk_results
        for value in chunk_rmse
    ]

    expected_pair_count = len(scans) - 1
    if len(relative_poses) != expected_pair_count:
        raise RuntimeError(
            "Parallel ICP produced an unexpected number of scan pairs: "
            f"{len(relative_poses)} instead of {expected_pair_count}"
        )

    return relative_poses, fitness, inlier_rmse

def _json_compatible(value: Any) -> Any:
    """Convert common NumPy and path objects into JSON-compatible values.

    Args:
        value: Arbitrary metadata value.

    Returns:
        A value accepted by ``json.dumps``.

    Raises:
        TypeError: If the value cannot be represented in JSON.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, set):
        return list(value)

    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def save_lidar_data_csv(
    lidar_data: LidarData,
    filepath: str | Path,
) -> Path:
    """Save a ``LidarData`` entity into one CSV file without scan paths.

    Each row represents one relative-pose interval between consecutive scans.
    The original ``N + 1`` scan timestamps are stored as start/end timestamp
    pairs and can therefore be reconstructed from ``N`` interval rows.

    Shared metadata is serialized as JSON in the final column of the first row.
    The remaining rows leave that column empty.

    Args:
        lidar_data: LiDAR data containing interval timestamps, relative poses,
            scan timestamps, ICP fitness, RMSE, and metadata.
        filepath: Destination CSV path.

    Returns:
        Path to the saved CSV file.

    Raises:
        ValueError: If fields have inconsistent dimensions, contain non-finite
            values, or contain no relative-pose intervals.
    """
    output_path = Path(filepath).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    interval_timestamps = np.asarray(
        lidar_data.timestamps_s,
        dtype=float,
    ).reshape(-1)
    relative_poses = np.asarray(
        lidar_data.relative_poses_se3,
        dtype=float,
    )
    scan_timestamps = np.asarray(
        lidar_data.scan_timestamps_s,
        dtype=float,
    ).reshape(-1)
    fitness = np.asarray(
        lidar_data.fitness,
        dtype=float,
    ).reshape(-1)
    inlier_rmse = np.asarray(
        lidar_data.inlier_rmse,
        dtype=float,
    ).reshape(-1)

    number_of_intervals = interval_timestamps.size

    ##################################################
    # Validate the interval and scan sequence
    ##################################################

    if number_of_intervals == 0:
        raise ValueError(
            "lidar_data must contain at least one relative-pose interval"
        )

    if relative_poses.shape != (number_of_intervals, 4, 4):
        raise ValueError(
            "relative_poses_se3 must have shape (N, 4, 4), where "
            "N is the number of interval timestamps"
        )

    if scan_timestamps.shape != (number_of_intervals + 1,):
        raise ValueError(
            "scan_timestamps_s must have shape (N + 1,)"
        )

    if fitness.shape != (number_of_intervals,):
        raise ValueError("fitness must have shape (N,)")

    if inlier_rmse.shape != (number_of_intervals,):
        raise ValueError("inlier_rmse must have shape (N,)")

    arrays_to_check = (
        interval_timestamps,
        relative_poses,
        scan_timestamps,
        fitness,
        inlier_rmse,
    )
    if not all(np.all(np.isfinite(array)) for array in arrays_to_check):
        raise ValueError("lidar_data contains non-finite numerical values")

    ##################################################
    # Serialize the shared metadata
    ##################################################

    metadata_json = json.dumps(
        dict(lidar_data.metadata),
        default=_json_compatible,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    pose_columns = [
        f"T_{row}{column}"
        for row in range(4)
        for column in range(4)
    ]

    records: list[dict[str, object]] = []

    ##################################################
    # Write one row for every consecutive scan pair
    ##################################################

    for interval_index in range(number_of_intervals):
        flattened_pose = relative_poses[interval_index].reshape(-1)

        record: dict[str, object] = {
            "interval_index": interval_index,
            "timestamp_s": interval_timestamps[interval_index],
            "scan_start_timestamp_s": scan_timestamps[interval_index],
            "scan_end_timestamp_s": scan_timestamps[interval_index + 1],
            "fitness": fitness[interval_index],
            "inlier_rmse": inlier_rmse[interval_index],
        }

        for column_name, value in zip(pose_columns, flattened_pose):
            record[column_name] = float(value)

        record["metadata_json"] = (
            metadata_json if interval_index == 0 else ""
        )
        records.append(record)

    column_order = [
        "interval_index",
        "timestamp_s",
        "scan_start_timestamp_s",
        "scan_end_timestamp_s",
        "fitness",
        "inlier_rmse",
        *pose_columns,
        "metadata_json",
    ]

    table = pd.DataFrame(records, columns=column_order)
    table.to_csv(
        output_path,
        index=False,
        float_format="%.17g",
    )
    return output_path


def load_lidar_data_csv(filepath: str | Path) -> LidarData:
    """Load a ``LidarData`` entity from a CSV that does not store scan paths.

    Args:
        filepath: CSV path produced by :func:`save_lidar_data_csv`.

    Returns:
        Reconstructed ``LidarData`` entity. Its ``source_scan_paths`` field is
        an empty list because paths are intentionally not stored in the CSV.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing, rows are malformed, or the
            scan timestamp sequence is inconsistent.
    """
    source_path = Path(filepath).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f"LiDAR CSV does not exist: {source_path}")

    table = pd.read_csv(source_path)

    pose_columns = [
        f"T_{row}{column}"
        for row in range(4)
        for column in range(4)
    ]
    required_columns = [
        "interval_index",
        "timestamp_s",
        "scan_start_timestamp_s",
        "scan_end_timestamp_s",
        "fitness",
        "inlier_rmse",
        *pose_columns,
        "metadata_json",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in table.columns
    ]
    if missing_columns:
        raise ValueError(
            f"LiDAR CSV is missing columns: {missing_columns}"
        )

    if table.empty:
        raise ValueError("LiDAR CSV contains no interval rows")

    ##################################################
    # Restore and validate interval ordering
    ##################################################

    table = table.sort_values(
        "interval_index",
        kind="stable",
    ).reset_index(drop=True)

    expected_indices = np.arange(table.shape[0], dtype=int)
    stored_indices = table["interval_index"].to_numpy(dtype=int)
    if not np.array_equal(stored_indices, expected_indices):
        raise ValueError(
            "interval_index must contain consecutive values starting at zero"
        )

    interval_timestamps = table["timestamp_s"].to_numpy(dtype=float)
    scan_start_timestamps = table[
        "scan_start_timestamp_s"
    ].to_numpy(dtype=float)
    scan_end_timestamps = table[
        "scan_end_timestamp_s"
    ].to_numpy(dtype=float)

    if not np.allclose(
        scan_start_timestamps[1:],
        scan_end_timestamps[:-1],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Consecutive interval rows contain inconsistent scan timestamps"
        )

    ##################################################
    # Reconstruct matrices and the N + 1 scan sequence
    ##################################################

    relative_poses = table.loc[
        :,
        pose_columns,
    ].to_numpy(dtype=float).reshape(-1, 4, 4)

    scan_timestamps = np.concatenate(
        [
            scan_start_timestamps[:1],
            scan_end_timestamps,
        ]
    )

    fitness = table["fitness"].to_numpy(dtype=float)
    inlier_rmse = table["inlier_rmse"].to_numpy(dtype=float)

    numerical_arrays = (
        interval_timestamps,
        scan_timestamps,
        relative_poses,
        fitness,
        inlier_rmse,
    )
    if not all(np.all(np.isfinite(array)) for array in numerical_arrays):
        raise ValueError("LiDAR CSV contains non-finite numerical values")

    ##################################################
    # Restore shared metadata from the first populated cell
    ##################################################

    metadata_json = "{}"
    for value in table["metadata_json"]:
        if isinstance(value, str) and value.strip():
            metadata_json = value
            break

    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "metadata_json contains invalid JSON"
        ) from exc

    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must encode a JSON object")

    return LidarData(
        timestamps_s=interval_timestamps,
        relative_poses_se3=relative_poses,
        scan_timestamps_s=scan_timestamps,
        source_scan_paths=[],
        fitness=fitness,
        inlier_rmse=inlier_rmse,
        metadata=metadata,
    )