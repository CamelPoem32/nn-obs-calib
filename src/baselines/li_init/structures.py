"""Validated sensor inputs, configuration, and results for a ROS-free LI-Init baseline.

Names and equation conventions follow LI-Init, but vectorized records replace its
per-sample C++ CalibState and ROS message buffers.

All timestamps are seconds, angular velocities rad/s, accelerations m/s^2,
and rotations are active column-vector rotations.

Compatible with:
    preprocessing.py
    temporal.py
    lidar_motion.py
    rotation.py
    translation.py
    estimator.py
    adapters.py
    benchmark.py
    notebooks/xx_evaluate_li_init.ipynb

Verified against 21 synthetic tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _times(value: ArrayLike, name: str) -> FloatArray:
    """Validate a strictly increasing timestamp sequence."""

    t = np.asarray(value, dtype=np.float64)

    if t.ndim != 1 or t.size < 2 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError(f"{name} must contain at least two finite, strictly increasing timestamps")

    return t.copy()


def _vectors(value: ArrayLike, n: int, name: str) -> FloatArray:
    """Validate a sequence of three-dimensional vectors."""

    values = np.asarray(value, dtype=np.float64)

    if values.shape != (n, 3) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite array of shape ({n}, 3)")

    return values.copy()


@dataclass(frozen=True)
class IMUSeries:
    """Raw IMU measurements.

    Attributes:
        t: Measurement timestamps in seconds, shape [N].
        data_gyr: Angular velocity in rad/s, shape [N, 3].
        data_acc: Optional specific force in m/s^2, shape [N, 3].

    Gyroscope and accelerometer biases are still included in the measurements.
    """

    t: ArrayLike
    data_gyr: ArrayLike
    data_acc: ArrayLike | None = None

    def __post_init__(self) -> None:
        t = _times(self.t, "IMU timestamps")

        object.__setattr__(self, "t", t)
        object.__setattr__(self, "data_gyr", _vectors(self.data_gyr, t.size, "data_gyr"))

        if self.data_acc is not None:
            object.__setattr__(self, "data_acc", _vectors(self.data_acc, t.size, "data_acc"))


@dataclass(frozen=True)
class LidarIncrements:
    """Relative SE(3) odometry between consecutive LiDAR scans.

    The default convention is:

        delta_T[k] = inv(T_WL(t_start[k])) @ T_WL(t_end[k])

    This transform maps coordinates from the end LiDAR frame into the
    start LiDAR frame.

    If the producer supplies inverse increments, specify:

        increment_convention="end_to_start"

    Absolute LiDAR poses and raw point clouds are not required for
    rotational calibration.

    Attributes:
        t_start: Interval start timestamps, shape [N].
        t_end: Interval end timestamps, shape [N].
        delta_T: Relative SE(3) transformations, shape [N, 4, 4].
        increment_convention: Direction of the supplied transformations.
    """

    t_start: ArrayLike
    t_end: ArrayLike
    delta_T: ArrayLike
    increment_convention: Literal["start_to_end", "end_to_start"] = "start_to_end"

    def __post_init__(self) -> None:
        start = np.asarray(self.t_start, dtype=np.float64)
        end = np.asarray(self.t_end, dtype=np.float64)
        transforms = np.asarray(self.delta_T, dtype=np.float64)

        n = start.size

        if start.ndim != 1 or n < 2 or end.shape != start.shape or not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
            raise ValueError("t_start and t_end must be finite arrays of the same length, at least two")

        if np.any(end <= start) or np.any(np.diff(start) <= 0) or np.any(np.diff(end) <= 0):
            raise ValueError("LiDAR intervals must have positive duration and increasing timestamps")

        if np.any(start[1:] < end[:-1] - 1e-9):
            raise ValueError("LiDAR odometry intervals must not overlap")

        if transforms.shape != (n, 4, 4) or not np.all(np.isfinite(transforms)):
            raise ValueError(f"delta_T must have finite shape ({n}, 4, 4)")

        if not np.allclose(transforms[:, 3, :], [0.0, 0.0, 0.0, 1.0], rtol=0, atol=1e-6):
            raise ValueError("delta_T must contain homogeneous SE(3) transforms")

        R = transforms[:, :3, :3]

        if not np.allclose(R @ np.swapaxes(R, -1, -2), np.eye(3), atol=1e-4) or np.any(np.linalg.det(R) <= 0):
            raise ValueError("delta_T rotation blocks must belong to SO(3)")

        if self.increment_convention not in ("start_to_end", "end_to_start"):
            raise ValueError("Unsupported increment_convention")

        object.__setattr__(self, "t_start", start.copy())
        object.__setattr__(self, "t_end", end.copy())
        object.__setattr__(self, "delta_T", transforms.copy())


@dataclass(frozen=True)
class AngularVelocitySeries:
    """Angular-motion representation of LI-Init's CalibState sequence.

    Attributes:
        t: Measurement timestamps, shape [N].
        ang_vel: Body-frame angular velocities, shape [N, 3].
        ang_acc: Optional temporal derivatives of angular velocity,
            shape [N, 3].

    ang_acc is angular acceleration, not IMU linear acceleration.
    """

    t: ArrayLike
    ang_vel: ArrayLike
    ang_acc: ArrayLike | None = None

    def __post_init__(self) -> None:
        t = _times(self.t, "angular-motion timestamps")

        object.__setattr__(self, "t", t)
        object.__setattr__(self, "ang_vel", _vectors(self.ang_vel, t.size, "ang_vel"))

        if self.ang_acc is not None:
            object.__setattr__(self, "ang_acc", _vectors(self.ang_acc, t.size, "ang_acc"))


@dataclass(frozen=True)
class LIInitConfig:
    """Configuration of the ROS-free LI-Init-inspired calibration pipeline.

    The LI-Init-inspired defaults retained here include:
        - Identity initialization of extrinsic rotation.
        - Zero initial gyroscope bias and temporal offset.
        - Ordinary least-squares optimization.
        - Sixth-order Butterworth filter with normalized cutoff 0.15.
        - Accelerometer bias bounds of +/-0.01 m/s^2.

    Filtering and acceleration normalization are disabled by default,
    because the intended application uses short five-second windows.

    The SVD rotation initializer, timestamp-aware correlation settings,
    interpolation-based temporal residual and translation spline parameters
    are adaptations of the original method.

    All parameters are configurable.
    """

    # ------------------------------------------------------------------
    # Data accumulation and preprocessing
    # ------------------------------------------------------------------

    # Original LI-Init accumulation setting. The window-based adapter
    # explicitly controls its own window duration instead.
    data_accum_length: float = 300.0

    # Filtering is intentionally disabled by default.
    filtering_enabled: bool = False

    # Source-like Butterworth parameters.
    filter_order: int = 6
    filter_normalized_cutoff: float = 0.15
    mean_filter_half_window: int = 2

    # The original pipeline includes optional acceleration normalization.
    # It is unsuitable by default for arbitrary dynamic five-second windows,
    # since the first IMU measurements need not be stationary.
    normalize_acceleration: bool = False
    gravity_norm: float = 9.81

    # ------------------------------------------------------------------
    # Rotation calibration
    # ------------------------------------------------------------------

    # "identity": original LI-Init-style initialization.
    # "svd": global algebraic initialization, recommended for large angles.
    rotation_initialization: Literal["svd", "identity"] = "identity"

    estimate_gyro_bias: bool = True
    initial_gyro_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Temporal calibration
    # ------------------------------------------------------------------

    estimate_time_offset: bool = True

    # Convention:
    #     t_imu = t_lidar + time_offset
    initial_time_offset: float = 0.0
    time_offset_bounds: tuple[float, float] = (-0.5, 0.5)

    # "taylor": LI-Init-inspired first-order temporal residual.
    # "interpolate": directly interpolate the IMU at shifted timestamps.
    temporal_residual: Literal["taylor", "interpolate"] = "taylor"

    # Maximum local search distance from a coarse temporal candidate
    # when using the Taylor residual.
    temporal_local_half_width: float = 0.05

    # ------------------------------------------------------------------
    # Coarse temporal synchronization
    # ------------------------------------------------------------------

    # Correlation search step in seconds.
    # None uses the median LiDAR motion-sample interval.
    xcorr_step: float | None = 0.01

    # Minimum fraction of LiDAR motion samples that must overlap
    # with the time-shifted IMU signal.
    xcorr_min_overlap_fraction: float = 0.5

    # Minimum angular-speed standard deviation for valid correlation.
    # Low-excitation windows must not generate arbitrary synchronization.
    xcorr_min_std: float = 1e-4

    # Minimum acceptable Pearson correlation.
    xcorr_min_score: float = 0.1

    # Keep several coarse candidates to reduce the effect of local
    # correlation ambiguities.
    xcorr_num_peaks: int = 3

    # Minimum separation between selected temporal candidates.
    # None selects an automatic separation based on xcorr_step.
    xcorr_peak_separation: float | None = None

    # ------------------------------------------------------------------
    # Numerical optimization
    # ------------------------------------------------------------------

    # "linear" corresponds to ordinary least squares.
    loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = "linear"

    f_scale: float = 1.0
    max_nfev: int = 200

    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float = 1e-8

    # Minimum number of measurements required for rotation estimation.
    min_samples: int = 6

    # Relative singular-value threshold for detecting degeneracy.
    observability_rtol: float = 1e-3

    # ------------------------------------------------------------------
    # Auxiliary translation calibration
    # ------------------------------------------------------------------

    # Translation is not part of the primary success criterion.
    estimate_translation: bool = False

    # Minimum number of overlapping LiDAR and accelerometer samples.
    translation_min_samples: int = 10

    # Smoothing factor for reconstructed LiDAR position splines.
    # Zero means interpolation without additional smoothing.
    translation_spline_smoothing: float = 0.0

    # Relative singular-value threshold after marginalizing
    # accelerometer bias and gravity nuisance parameters.
    translation_observability_rtol: float = 1e-3

    # Initial IMU position expressed in LiDAR coordinates.
    initial_p_IL: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Initial accelerometer bias in IMU coordinates, m/s^2.
    initial_accelerometer_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Source-like bounded accelerometer bias optimization.
    accelerometer_bias_bounds: tuple[float, float] = (-0.01, 0.01)

    def __post_init__(self) -> None:
        """Validate configuration before entering numerical optimization."""

        if not np.isfinite(self.data_accum_length) or self.data_accum_length <= 0:
            raise ValueError("data_accum_length must be positive and finite")

        if not isinstance(self.filter_order, int) or isinstance(self.filter_order, bool) or self.filter_order < 1:
            raise ValueError("filter_order must be a positive integer")

        if not np.isfinite(self.filter_normalized_cutoff) or not 0 < self.filter_normalized_cutoff < 1:
            raise ValueError("filter_normalized_cutoff must lie in (0, 1)")

        if not isinstance(self.mean_filter_half_window, int) or self.mean_filter_half_window < 0:
            raise ValueError("mean_filter_half_window must be a nonnegative integer")

        if not np.isfinite(self.gravity_norm) or self.gravity_norm <= 0:
            raise ValueError("gravity_norm must be positive and finite")

        if self.rotation_initialization not in ("identity", "svd"):
            raise ValueError("rotation_initialization must be 'identity' or 'svd'")

        if self.temporal_residual not in ("taylor", "interpolate"):
            raise ValueError("temporal_residual must be 'taylor' or 'interpolate'")

        if self.loss not in ("linear", "soft_l1", "huber", "cauchy", "arctan"):
            raise ValueError("Unsupported least-squares loss")

        if not np.isfinite(self.initial_time_offset):
            raise ValueError("initial_time_offset must be finite")

        if len(self.time_offset_bounds) != 2:
            raise ValueError("time_offset_bounds must contain (lower, upper)")

        lo, hi = self.time_offset_bounds

        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi and lo <= self.initial_time_offset <= hi):
            raise ValueError("time_offset_bounds must bracket initial_time_offset")

        if not np.isfinite(self.temporal_local_half_width) or self.temporal_local_half_width <= 0:
            raise ValueError("temporal_local_half_width must be positive and finite")

        # Each vector initializer must contain three finite coordinates.
        for name, vector in (
            ("initial_gyro_bias", self.initial_gyro_bias),
            ("initial_p_IL", self.initial_p_IL),
            ("initial_accelerometer_bias", self.initial_accelerometer_bias),
        ):
            array = np.asarray(vector, dtype=np.float64)

            if array.shape != (3,) or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain three finite values")

        # Coarse temporal synchronization configuration.
        if self.xcorr_step is not None and (not np.isfinite(self.xcorr_step) or self.xcorr_step <= 0):
            raise ValueError("xcorr_step must be None or a positive finite number")

        if not np.isfinite(self.xcorr_min_overlap_fraction) or not 0 < self.xcorr_min_overlap_fraction <= 1:
            raise ValueError("xcorr_min_overlap_fraction must lie in (0, 1]")

        if not np.isfinite(self.xcorr_min_std) or self.xcorr_min_std < 0:
            raise ValueError("xcorr_min_std must be nonnegative and finite")

        if not np.isfinite(self.xcorr_min_score) or not -1 <= self.xcorr_min_score <= 1:
            raise ValueError("xcorr_min_score must lie in [-1, 1]")

        if not isinstance(self.xcorr_num_peaks, int) or isinstance(self.xcorr_num_peaks, bool) or self.xcorr_num_peaks < 1:
            raise ValueError("xcorr_num_peaks must be a positive integer")

        if self.xcorr_peak_separation is not None and (
            not np.isfinite(self.xcorr_peak_separation) or self.xcorr_peak_separation <= 0
        ):
            raise ValueError("xcorr_peak_separation must be None or positive and finite")

        # Least-squares settings.
        if not np.isfinite(self.f_scale) or self.f_scale <= 0:
            raise ValueError("f_scale must be positive and finite")

        if not isinstance(self.max_nfev, int) or isinstance(self.max_nfev, bool) or self.max_nfev <= 0:
            raise ValueError("max_nfev must be a positive integer")

        for name, value in (
            ("ftol", self.ftol),
            ("xtol", self.xtol),
            ("gtol", self.gtol),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")

        if not isinstance(self.min_samples, int) or self.min_samples < 3:
            raise ValueError("min_samples must be an integer >= 3")

        for name, value in (
            ("observability_rtol", self.observability_rtol),
            ("translation_observability_rtol", self.translation_observability_rtol),
        ):
            if not np.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must lie in (0, 1)")

        # Translation configuration.
        if not isinstance(self.translation_min_samples, int) or self.translation_min_samples < 3:
            raise ValueError("translation_min_samples must be an integer >= 3")

        if not np.isfinite(self.translation_spline_smoothing) or self.translation_spline_smoothing < 0:
            raise ValueError("translation_spline_smoothing must be nonnegative and finite")

        if len(self.accelerometer_bias_bounds) != 2:
            raise ValueError("accelerometer_bias_bounds must contain (lower, upper)")

        bias_lo, bias_hi = self.accelerometer_bias_bounds

        if not (np.isfinite(bias_lo) and np.isfinite(bias_hi) and bias_lo < bias_hi):
            raise ValueError("accelerometer_bias_bounds must be finite and ordered")

        if np.any(np.asarray(self.initial_accelerometer_bias) < bias_lo) or np.any(
            np.asarray(self.initial_accelerometer_bias) > bias_hi
        ):
            raise ValueError("initial_accelerometer_bias must lie within accelerometer_bias_bounds")


@dataclass(frozen=True)
class RotationCalibrationResult:
    """Result of LI-Init's rotational calibration stages.

    The estimated rotation R_LI maps LiDAR vectors into IMU coordinates:

        omega_I(t_L + tau) = R_LI @ omega_L(t_L) + gyro_bias

    Attributes:
        R_LI: Estimated rotation matrix, shape [3, 3].
        gyro_bias: Gyroscope bias in IMU coordinates, shape [3].
        time_offset: Estimated clock offset in seconds.
        success: Optimization and rotation observability succeeded.
        observable: Whether rotation is sufficiently constrained.
        message: Optimization status or failure reason.
        rms_residual: RMS angular-velocity residual in rad/s.
        num_samples: Number of aligned angular-velocity samples.
        rotation_singular_values: Singular values describing motion excitation.
        rotation_rank: Rank used to assess rotational observability.
        joint_singular_values: Optional joint optimizer Jacobian singular values.
        joint_rank: Optional joint optimizer rank.
        nfev: Number of function evaluations.
    """

    R_LI: FloatArray
    gyro_bias: FloatArray
    time_offset: float
    success: bool
    observable: bool
    message: str
    rms_residual: float
    num_samples: int
    rotation_singular_values: FloatArray
    rotation_rank: int
    joint_singular_values: FloatArray | None = None
    joint_rank: int | None = None
    nfev: int = 0


@dataclass(frozen=True)
class TemporalCalibrationResult:
    """Result of coarse LiDAR-IMU temporal synchronization.

    Time offset convention:

        t_imu = t_lidar + time_offset

    Attributes:
        time_offset: Estimated coarse offset in seconds.
        success: Whether correlation passed the acceptance threshold.
        message: Synchronization status or failure description.
        candidate_offsets: Selected candidate offsets ordered by score.
        tested_offsets: All evaluated offsets in seconds.
        scores: Correlation scores for all tested offsets.
        overlap_counts: Number of overlapping samples for each offset.
    """

    time_offset: float
    success: bool
    message: str
    candidate_offsets: FloatArray
    tested_offsets: FloatArray
    scores: FloatArray
    overlap_counts: NDArray[np.int64]


@dataclass(frozen=True)
class TranslationCalibrationResult:
    """Result of auxiliary LiDAR-IMU translational calibration.

    Translation is estimated together with accelerometer bias and gravity.

    Attributes:
        p_IL: IMU origin expressed in LiDAR coordinates, shape [3].
        p_LI: LiDAR origin expressed in IMU coordinates, shape [3].
        accelerometer_bias: Accelerometer bias in IMU coordinates, shape [3].
        gravity_W: Gravity in the locally reconstructed world frame, shape [3].
        success: Optimization succeeded and translation is observable.
        observable: Whether all three translation DoFs are identifiable.
        message: Optimization status or failure description.
        rms_residual: RMS accelerometer residual.
        num_samples: Number of aligned accelerometer samples.
        translation_singular_values: Singular values after eliminating
            nuisance parameters.
        translation_rank: Observable translation rank (0 to 3).
        nfev: Number of numerical function evaluations.

    A converged optimizer does not imply that translation is observable.
    """

    p_IL: FloatArray
    p_LI: FloatArray
    accelerometer_bias: FloatArray
    gravity_W: FloatArray
    success: bool
    observable: bool
    message: str
    rms_residual: float
    num_samples: int
    translation_singular_values: FloatArray
    translation_rank: int
    nfev: int


@dataclass(frozen=True)
class LIInitializationResult:
    """Complete result of the ROS-free LI-Init calibration pipeline.

    The primary calibration consists of extrinsic rotation and temporal offset.
    Translation is optional and does not affect the primary success criterion.

    Attributes:
        rotation: Final joint rotation, gyroscope-bias and timing solution.
        temporal: Coarse temporal synchronization result, if enabled.
        translation: Auxiliary lever-arm calibration result, if enabled.
        rotation_only: Intermediate rotation-only initialization result.
    """

    rotation: RotationCalibrationResult
    temporal: TemporalCalibrationResult | None
    translation: TranslationCalibrationResult | None
    rotation_only: RotationCalibrationResult | None

    @property
    def R_LI(self) -> FloatArray:
        """Return the final LiDAR-to-IMU rotation matrix."""
        return self.rotation.R_LI

    @property
    def time_offset(self) -> float:
        """Return the refined temporal offset in seconds."""
        return self.rotation.time_offset

    @property
    def gyro_bias(self) -> FloatArray:
        """Return the final gyroscope bias in IMU coordinates."""
        return self.rotation.gyro_bias

    @property
    def success(self) -> bool:
        """Return whether the primary rotation and temporal stages succeeded.

        Translation is auxiliary and has an independent success flag.
        """
        return bool(
            self.rotation.success
            and (self.temporal is None or self.temporal.success)
        )