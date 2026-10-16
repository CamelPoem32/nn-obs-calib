"""Plotting helpers for inspecting New College sensor streams in notebooks."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np

from .data import IMUData, LidarData


def plot_imu_streams(imu_streams: Mapping[str, IMUData] | list[IMUData]) -> tuple[plt.Figure, np.ndarray]:
    """Plot accelerometer and gyroscope components for one or more IMU streams.

    Args:
        imu_streams: Mapping or list of ``IMUData`` streams.

    Returns:
        Matplotlib figure and axes array.
    """
    items = list(imu_streams.items()) if isinstance(imu_streams, Mapping) else [(imu.name, imu) for imu in imu_streams]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    component_names = ("x", "y", "z")

    # Draw all streams on shared axes so timing and scale differences stand out.
    for name, imu in items:
        for axis, component in enumerate(component_names):
            axes[0].plot(imu.timestamps_s, imu.accel_mps2[:, axis], label=f"{name} accel {component}", alpha=0.8)
            axes[1].plot(imu.timestamps_s, imu.gyro_radps[:, axis], label=f"{name} gyro {component}", alpha=0.8)

    axes[0].set_ylabel("accel, m/s^2")
    axes[1].set_ylabel("gyro, rad/s")
    axes[1].set_xlabel("time, s")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(ncol=3, fontsize="small")
    fig.tight_layout()
    return fig, axes


def plot_lidar_rates(
    lidar_data: LidarData,
    angular_velocity_radps: np.ndarray,
    linear_velocity_mps: np.ndarray,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot LiDAR-derived angular and linear velocities.

    Args:
        lidar_data: LiDAR relative-pose stream.
        angular_velocity_radps: ``(N, 3)`` angular velocities.
        linear_velocity_mps: ``(N, 3)`` linear velocities.

    Returns:
        Matplotlib figure and axes array.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    component_names = ("x", "y", "z")

    # Plot rates against interval timestamps produced by the LiDAR loader.
    for axis, component in enumerate(component_names):
        axes[0].plot(lidar_data.timestamps_s, angular_velocity_radps[:, axis], label=f"omega {component}")
        axes[1].plot(lidar_data.timestamps_s, linear_velocity_mps[:, axis], label=f"v {component}")

    axes[0].set_ylabel("angular velocity, rad/s")
    axes[1].set_ylabel("linear velocity, m/s")
    axes[1].set_xlabel("time, s")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(ncol=3, fontsize="small")
    fig.tight_layout()
    return fig, axes


def plot_sensor_timestamps(
    imu_streams,
    lidar_data,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot IMU and LiDAR sample times on one raster-style timeline.

    Args:
        imu_streams: Mapping or list of ``IMUData`` streams.
        lidar_data: LiDAR relative-pose stream.

    Returns:
        Matplotlib figure and axis.
    """
    items = list(imu_streams.items()) if isinstance(imu_streams, Mapping) else [(imu.name, imu) for imu in imu_streams]
    fig, axis = plt.subplots(figsize=(14, 3 + 0.4 * len(items)))

    # Use vertical ticks to make synchronization gaps visually obvious.
    y = 0
    for name, imu in items:
        x = np.min(imu.timestamps_s)
        axis.vlines(imu.timestamps_s, y - 0.35, y + 0.35, alpha=0.25)
        axis.text(x, y, name, va="center", ha="right")
        y += 1
    x = np.min(lidar_data.timestamps_s)
    axis.vlines(lidar_data.timestamps_s, y - 0.35, y + 0.35, color="tab:red", alpha=0.8)
    axis.text(x, y, "lidar", va="center", ha="right", color="tab:red")

    axis.set_yticks([])
    axis.set_xlabel("time, s")
    axis.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    return fig, axis
