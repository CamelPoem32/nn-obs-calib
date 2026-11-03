"""Time-aligned relative SE(3) motion pairs from two scan-pair odometry streams.

Uses only existing odometry, not raw point clouds. Each sensor's local trajectory
is composed from its own relative transforms. For asynchronous scan times, the
local poses are interpolated with SciPy quaternion SLERP + linear world-position
interpolation. This is a *trajectory approximation*, not a continuous-time
LiDAR registration or motion-distortion correction.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from baselines.li_init.structures import LidarIncrements
from .geometry import invert_T


@dataclass(frozen=True)
class LidarMotionPairs:
    left: np.ndarray  # [M,4,4] relative start/end motion
    right: np.ndarray # [M,4,4] relative start/end motion on same intervals
    t_start: np.ndarray
    t_end: np.ndarray


class _RelativeTrajectory:
    def __init__(self, data: LidarIncrements, max_gap_s: float, time_shift_s: float = 0.0):
        edges = data.delta_T if data.increment_convention == 'start_to_end' else invert_T(data.delta_T)
        starts = data.t_start - time_shift_s
        ends = data.t_end - time_shift_s
        if np.any(ends - starts > max_gap_s) or np.any(starts[1:] - ends[:-1] > max_gap_s):
            raise ValueError('LiDAR odometry contains a scan interval or gap larger than pair_max_scan_gap_s')
        self.timestamps = np.r_[starts[0], ends]
        if np.any(np.diff(self.timestamps) <= 0):
            raise ValueError('Non-increasing LiDAR pose times')
        self.transforms = np.empty((len(self.timestamps), 4, 4))
        self.transforms[0] = np.eye(4)
        for k, edge in enumerate(edges):
            self.transforms[k + 1] = self.transforms[k] @ edge
        self.slerp = Slerp(self.timestamps, Rotation.from_matrix(self.transforms[:, :3, :3]))

    def pose(self, t: np.ndarray) -> np.ndarray:
        query = np.asarray(t, dtype=np.float64).reshape(-1)
        if query.min() < self.timestamps[0] - 1e-8 or query.max() > self.timestamps[-1] + 1e-8:
            raise ValueError('Requested LiDAR pose outside measured time range')
        query = np.clip(query, self.timestamps[0], self.timestamps[-1])
        out = np.repeat(np.eye(4)[None], len(query), axis=0)
        out[:, :3, :3] = self.slerp(query).as_matrix()
        for d in range(3):
            out[:, d, 3] = np.interp(query, self.timestamps, self.transforms[:, d, 3])
        return out


def align_lidar_motion_pairs(left: LidarIncrements, right: LidarIncrements, *, max_gap_s: float = 0.75, min_intervals: int = 3, right_time_offset_s: float = 0.0) -> LidarMotionPairs:
    """Evaluate right-LiDAR relative motion on left-LiDAR scan intervals.

    ``right_time_offset_s`` follows t_right=t_left+offset, so right timestamps
    are shifted backward to the left clock before matching.
    """
    tr_left = _RelativeTrajectory(left, max_gap_s)
    tr_right = _RelativeTrajectory(right, max_gap_s, time_shift_s=right_time_offset_s)
    starts, ends = left.t_start, left.t_end
    valid = (starts >= tr_right.timestamps[0] - 1e-8) & (ends <= tr_right.timestamps[-1] + 1e-8)
    if np.count_nonzero(valid) < min_intervals:
        raise ValueError(f'Too few overlapping LiDAR motion pairs: {valid.sum()}')
    a, b = starts[valid], ends[valid]
    l_a, l_b = tr_left.pose(a), tr_left.pose(b)
    r_a, r_b = tr_right.pose(a), tr_right.pose(b)
    d_l = invert_T(l_a) @ l_b
    d_r = invert_T(r_a) @ r_b
    return LidarMotionPairs(left=d_l, right=d_r, t_start=a.copy(), t_end=b.copy())
