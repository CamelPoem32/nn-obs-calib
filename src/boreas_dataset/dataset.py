"""Lightweight Boreas sequence discovery and lazy sensor access."""

from __future__ import annotations

import os
from pathlib import Path

from .calibration import load_calibrations
from .imu import IMU_SOURCES, load_imu
from .lidar import LIDAR_SOURCES, discover_lidar_scans, load_lidar_poses
from .timing import timestamp_statistics


DEFAULT_SEQUENCE_ID = "boreas-2024-12-03-12-54"
DEFAULT_DATA_ROOT = (Path("D:/Downloads/MobRobLab/BoreasDataset") if os.name == "nt"
                     else Path("/mnt/d/Downloads/MobRobLab/BoreasDataset"))


def resolve_data_root(root: str | Path | None = None) -> Path:
    """Explicit root overrides BOREAS_DATA_ROOT, which overrides the local example."""
    return Path(root if root is not None else os.environ.get("BOREAS_DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser()


def discover_sequences(root: str | Path | None = None) -> list[str]:
    """Accept either a dataset parent or one sequence directory, including partial downloads."""
    root = resolve_data_root(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Boreas dataset root is unavailable: {root}")
    if root.name.startswith("boreas-"):
        return [root.name]
    return sorted(p.name for p in root.glob("boreas-*") if p.is_dir())


class BoreasDataset:
    """Dataset parent; constructing it loads no sensor measurements or point clouds."""

    def __init__(self, root: str | Path | None = None):
        self.root = resolve_data_root(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Boreas dataset root is unavailable: {self.root}")

    @property
    def sequences(self) -> list[str]:
        return discover_sequences(self.root)

    def sequence(self, sequence_id: str = DEFAULT_SEQUENCE_ID) -> BoreasSequence:
        if sequence_id not in self.sequences:
            raise FileNotFoundError(f"Sequence {sequence_id!r} unavailable under {self.root}")
        path = self.root if self.root.name == sequence_id else self.root / sequence_id
        return BoreasSequence(path)


class BoreasSequence:
    """One sequence. Time bounds throughout this API are absolute UTC seconds."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Sequence directory is unavailable: {self.root}")
        self.sequence_id = self.root.name

    @property
    def sensors(self) -> dict:
        """Report available supported IMUs/LiDAR and detected out-of-scope sources."""
        return {"imu": [key for key, spec in IMU_SOURCES.items() if (self.root / spec[0]).is_file()],
                "lidar": [key for key, (folder, _, _) in LIDAR_SOURCES.items() if any((self.root / folder).glob("*.bin"))],
                "detected_unimplemented": [key for key in ("radar", "camera")
                                           if (self.root / key).is_dir()]}

    def inventory(self) -> dict:
        """Expected file availability and actual trajectory/calibration/map-like files."""
        expected = [spec[0] for spec in IMU_SOURCES.values()] + ["applanix/lidar_poses.csv",
                    "applanix/gps_post_process.csv", "calib/T_applanix_lidar.txt", "calib/T_applanix_dmu.txt"]
        # Search map-like paths without opening any raw sensor files.
        maps = [str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file()
                and (p.suffix.lower() in {".las", ".laz", ".pcd", ".ply"} or "map" in p.stem.lower())]
        return {"expected_files": {name: (self.root / name).is_file() for name in expected},
                "trajectories": sorted(p.name for p in (self.root / "applanix").glob("*poses.csv")),
                "calibrations": sorted(p.stem for p in (self.root / "calib").glob("T_*.txt")),
                "map_like_files": sorted(maps), "sensors": self.sensors}

    def imu(self, source: str = "dmu", *, time_start=None, time_end=None):
        return load_imu(self.root, source, time_start=time_start, time_end=time_end)

    def lidar_scans(self, *, source: str = "velodyne", time_start=None, time_end=None):
        return discover_lidar_scans(self.root, source=source, time_start=time_start, time_end=time_end)

    def lidar_poses(self, *, source="velodyne", time_start=None, time_end=None):
        if source not in LIDAR_SOURCES:
            raise ValueError(f"Unknown LiDAR source: {source}")
        folder = LIDAR_SOURCES[source][0]
        return load_lidar_poses(self.root / f"applanix/{folder}_poses.csv", source=source,
                                time_start=time_start, time_end=time_end)

    @property
    def calibrations(self):
        return load_calibrations(self.root)

    def summary(self) -> dict:
        """Explicitly scan CSV timestamps for counts, duration, coverage and acquisition gaps.

        No point clouds are loaded. IMU sources are processed one at a time.
        Missing optional files are listed in inventory and skipped here.
        """
        coverage = {}
        for source in self.sensors["imu"]:
            imu = self.imu(source)
            coverage[f"imu/{source}"] = timestamp_statistics(imu.timestamps_ns)
        for source, (folder, _, _) in LIDAR_SOURCES.items():
            if (self.root / folder).is_dir():
                coverage[f"lidar/{source}"] = timestamp_statistics([scan.timestamp_ns for scan in self.lidar_scans(source=source)])
            if (self.root / f"applanix/{folder}_poses.csv").is_file():
                coverage[f"gt/{folder}"] = timestamp_statistics(self.lidar_poses(source=source).timestamps_ns)
        starts = [v["start_s"] for v in coverage.values() if v["count"]]
        ends = [v["end_s"] for v in coverage.values() if v["count"]]
        return {"sequence_id": self.sequence_id, "root": str(self.root), "coverage": coverage,
                "duration_s": max(ends) - min(starts) if starts else 0.0,
                "common_interval_s": (max(starts), min(ends)) if starts and max(starts) <= min(ends) else None}
