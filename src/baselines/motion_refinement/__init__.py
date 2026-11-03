"""ROS-free local motion-based rotational calibration refinement."""
from .structures import IMUSeries, LidarIncrements, MotionRefinementConfig, MotionRefinementInput, MotionRefinementResult, RotationObservability, IterationTrace
from .estimator import MotionRefinementCalibrator
from .preintegration import integrate_gyro_intervals
from .lidar_pair import align_lidar_motion_pairs, LidarMotionPairs
from .observability import analyze_rotation_observability

__all__ = ['IMUSeries', 'LidarIncrements', 'MotionRefinementConfig', 'MotionRefinementInput', 'MotionRefinementResult', 'RotationObservability', 'IterationTrace', 'MotionRefinementCalibrator', 'integrate_gyro_intervals', 'align_lidar_motion_pairs', 'LidarMotionPairs', 'analyze_rotation_observability']
from .benchmark import MotionRefinementWindow, run_motion_refinement_rollout, rotation_error_degrees
__all__ += ['MotionRefinementWindow', 'run_motion_refinement_rollout', 'rotation_error_degrees']
