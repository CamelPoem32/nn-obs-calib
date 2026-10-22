"""Flow matching on calibration corrections in SE(3) x R.

Import order and geometry conventions follow ``obscalib.geometry.lie``. No
model, checkpoint, configuration, or training-loop imports are required here.
"""

from .calibration_path import CalibrationFlowBatch, make_calibration_flow_batch, sample_flow_times
from .lie_integrator import CalibrationFlowResult, CalibrationFlowState, CalibrationVelocityField, integrate_calibration_flow
from .loss import CalibrationFlowLoss, CalibrationFlowLossWeights, calibration_velocity_loss

__all__ = [
    "CalibrationFlowBatch", "make_calibration_flow_batch", "sample_flow_times",
    "CalibrationFlowResult", "CalibrationFlowState", "CalibrationVelocityField", "integrate_calibration_flow",
    "CalibrationFlowLoss", "CalibrationFlowLossWeights", "calibration_velocity_loss",
]
