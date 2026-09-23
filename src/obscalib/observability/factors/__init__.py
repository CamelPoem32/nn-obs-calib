"""Observability factor kernels.

Canonical Torch implementations remain the default reference backend. Compiled
NumPy/Numba kernels are exported with explicit ``_njitted`` suffixes and are
selected by the high-level linearization functions when ``njit=True``.
"""

from obscalib.observability.factors.gyroscope_njitted import gyroscope_interval_is_supported_njitted, integrate_gyroscope_signal_linear_njitted, interpolate_gyroscope_linear_njitted, linearize_gyroscope_factor_njitted
from obscalib.observability.factors.lidar_njitted import linearize_lidar_factor_njitted

__all__ = [
    "gyroscope_interval_is_supported_njitted",
    "integrate_gyroscope_signal_linear_njitted",
    "interpolate_gyroscope_linear_njitted",
    "linearize_gyroscope_factor_njitted",
    "linearize_lidar_factor_njitted",
]
