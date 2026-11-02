"""Convert relative LiDAR odometry into LI-Init-style body angular velocities.

The original push_Lidar_CalibState() accepts angular velocities from FAST-LO;
we replace that ROS/point-cloud frontend using scan-pair SE(3) increments.
"""

from __future__ import annotations

import numpy as np
import torch

from obscalib.geometry.lie import so3_log
from .structures import AngularVelocitySeries, LidarIncrements


def lidar_increments_to_motion(increments: LidarIncrements) -> AngularVelocitySeries:
    """Extract the scan-interval mean body angular velocity, expressed in LiDAR axes.

    Each interval is represented at its midpoint.  The rotation logarithm is
    computed with obscalib's canonical, near-pi-safe SO(3) implementation.
    This is an interval average, not an instantaneous FAST-LO velocity: very
    fast motion changes within a scan interval are therefore not recoverable.
    """

    R = increments.delta_T[:, :3, :3]
    if increments.increment_convention == "end_to_start":
        R = np.swapaxes(R, -1, -2)
    phi = so3_log(torch.as_tensor(np.ascontiguousarray(R), dtype=torch.float64)).detach().cpu().numpy()
    dt = increments.t_end - increments.t_start
    t = 0.5 * (increments.t_start + increments.t_end)
    return AngularVelocitySeries(t=t, ang_vel=phi / dt[:, None])


def central_diff(states: AngularVelocitySeries) -> AngularVelocitySeries:
    """Estimate angular acceleration via centered finite differences (LI-Init name).

    Unlike original LI-Init, derivative endpoints use one-sided differences
    rather than retaining empty endpoint states.  For two samples use secants.
    Irregular timestamps are supported by numpy.gradient.
    """

    edge_order = 2 if states.t.size >= 3 else 1
    ang_acc = np.gradient(states.ang_vel, states.t, axis=0, edge_order=edge_order)
    return AngularVelocitySeries(t=states.t, ang_vel=states.ang_vel, ang_acc=ang_acc)
