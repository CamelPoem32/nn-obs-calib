"""Batch independent scan odometry. Outputs go directly in each sequence folder.

Example from the repository root (module invocation also supported):
    python src/boreas_dataset/import_lidar_odometry.py /mnt/d/Downloads/MobRobLab/BoreasDataset
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from boreas_dataset.dataset import BoreasDataset
    from boreas_dataset.odometry import compute_lidar_odometry, save_lidar_data_csv
else:
    from .dataset import BoreasDataset
    from .odometry import compute_lidar_odometry, save_lidar_data_csv


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract Velodyne/Aeva scan odometry for every downloaded Boreas sequence.")
    parser.add_argument("dataset_root", type=Path, nargs="?", help="Parent or one sequence; otherwise BOREAS_DATA_ROOT.")
    parser.add_argument("--sequences", nargs="+", help="Optional sequence IDs; default all discovered sequences.")
    parser.add_argument("--sources", nargs="+", choices=("velodyne", "aeva"), default=("velodyne", "aeva"))
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output CSVs; default skip existing files.")
    parser.add_argument("--dry-run", action="store_true", help="List jobs, scan counts and exact output locations without ICP.")
    parser.add_argument("--max-scans", type=int, help="Optional bounded run; recorded in output metadata.")
    parser.add_argument("--scan-step", type=int, default=1)
    parser.add_argument("--time-start", help="Inclusive absolute UTC seconds.")
    parser.add_argument("--time-end", help="Inclusive absolute UTC seconds.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cuda-device-id", type=int, default=0)
    parser.add_argument("--voxel-size-m", type=float, default=0.5)
    parser.add_argument("--max-correspondence-distance-m", type=float, default=1.5)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--local-map-scans", type=int, default=5)
    parser.add_argument("--voxel-scale-factors", type=float, nargs="+", default=(2.0, 1.0, 0.5))
    parser.add_argument("--robust-kernel", choices=("tukey", "huber", "cauchy", "l2", "none"), default="tukey")
    parser.add_argument("--robust-kernel-scale-factor", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--previous-motion-initialization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bidirectional-check", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verbosity", "-v", type=int, choices=(0, 1, 2), default=1)
    return parser


def run(args: argparse.Namespace) -> dict:
    """Process jobs serially; skip absent sensors, continue failures, report status."""
    dataset = BoreasDataset(args.dataset_root)
    ids = dataset.sequences if args.sequences is None else args.sequences
    if not ids:
        raise ValueError(f"No Boreas sequences found under {dataset.root}")
    if args.scan_step < 1 or (args.max_scans is not None and args.max_scans < 2):
        raise ValueError("scan-step must be positive and max-scans must be at least two")
    report = {"saved": [], "skipped": [], "failed": [], "planned": []}
    for sequence_id in dict.fromkeys(ids):
        sequence = dataset.sequence(sequence_id)
        for source in dict.fromkeys(args.sources):
            destination = sequence.root / f"lidar_odometry_{source}.csv"
            if source not in sequence.sensors["lidar"]:
                if args.verbosity:
                    print(f"SKIP {sequence_id}/{source}: scans not downloaded", flush=True)
                report["skipped"].append(str(destination))
                continue
            if destination.exists() and not args.overwrite:
                if args.verbosity:
                    print(f"SKIP existing {destination}", flush=True)
                report["skipped"].append(str(destination))
                continue
            selected = sequence.lidar_scans(source=source, time_start=args.time_start, time_end=args.time_end)[::args.scan_step]
            selected = selected if args.max_scans is None else selected[:args.max_scans]
            if args.verbosity or args.dry_run:
                print(f"{'PLAN' if args.dry_run else 'RUN'} {sequence_id}/{source}: {len(selected)} scans -> {destination}", flush=True)
            if len(selected) < 2:
                report["failed"].append({"path": str(destination), "error": "fewer than two selected scans"})
                continue
            if args.dry_run:
                report["planned"].append(str(destination))
                continue
            temporary = destination.with_suffix(".csv.partial")
            try:
                result = compute_lidar_odometry(sequence.root, source=source, time_start=args.time_start,
                    time_end=args.time_end, scan_step=args.scan_step, max_scans=args.max_scans,
                    voxel_size_m=args.voxel_size_m, max_correspondence_distance_m=args.max_correspondence_distance_m,
                    max_iterations=args.max_iterations, local_map_scans=args.local_map_scans,
                    voxel_scale_factors=tuple(args.voxel_scale_factors), use_previous_motion_initialization=args.previous_motion_initialization,
                    robust_kernel=None if args.robust_kernel == "none" else args.robust_kernel,
                    robust_kernel_scale_factor=args.robust_kernel_scale_factor, bidirectional_check=args.bidirectional_check,
                    number_of_workers=args.workers, device=args.device, cuda_device_id=args.cuda_device_id)
                save_lidar_data_csv(result, temporary)
                temporary.replace(destination)
                report["saved"].append(str(destination))
                if args.verbosity:
                    print(f"SAVED {destination}: {len(result.timestamps_s)} intervals, "
                          f"median fitness={np.median(result.fitness):.4f}, median RMSE={np.median(result.inlier_rmse):.4f} m", flush=True)
            except Exception as error:
                if temporary.exists():
                    temporary.unlink()
                report["failed"].append({"path": str(destination), "error": str(error)})
                print(f"FAILED {sequence_id}/{source}: {error}", file=sys.stderr, flush=True)
    if args.verbosity:
        print("Batch:", {key: len(value) for key, value in report.items()}, flush=True)
    return report


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    try:
        report = run(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

# python src/boreas_dataset/import_lidar_odometry.py /mnt/d/Downloads/MobRobLab/BoreasDataset