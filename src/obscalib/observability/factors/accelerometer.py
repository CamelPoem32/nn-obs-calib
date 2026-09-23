"""Simple accelerometer observability factor.

The initial observability pipeline deliberately uses the conservative one-pose
gravity-alignment model. Translational acceleration, angular acceleration,
lever-arm effects, accelerometer bias, scale, and axis misalignment are not
modeled here.

The temporal calibration convention is

    t_true = t_sensor + tau_I.

All SE(3) perturbations are left perturbations with rotation-first tangent
ordering [phi, rho].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from obscalib.geometry.lie import so3_hat


@dataclass(frozen=True)
class AccelerometerFactorTerms:
    """Reusable quantities for one simple accelerometer factor.

    The factor predicts gravity-induced specific force in the IMU frame at one
    trajectory pose. These terms are retained for diagnostics and future
    analytic-versus-finite-difference tests.
    """

    residual: torch.Tensor
    predicted_specific_force: torch.Tensor
    measured_specific_force: torch.Tensor
    gravity_world: torch.Tensor
    T_W_B: torch.Tensor
    T_B_I: torch.Tensor
    sensor_time: float
    true_time: float
    tau_I: float


@dataclass(frozen=True)
class AccelerometerFactorLinearization:
    """Complete local linearization of one simple accelerometer factor.

    The residual has shape [3]. H_pose and H_T_B_I have shape [3, 6], while
    H_tau_I has shape [3, 1]. Translation columns of both SE(3) Jacobian blocks
    are exactly zero for this gravity-only model.
    """

    residual: torch.Tensor
    H_pose: torch.Tensor
    H_T_B_I: torch.Tensor
    H_tau_I: torch.Tensor
    terms: AccelerometerFactorTerms


def _validate_scientific_tensor(tensor: torch.Tensor, *, name: str) -> None:
    """Validate one tensor at the scientific observability boundary.

    Factor kernels operate on detached CPU float64 tensors so numerical
    diagnostics are not affected by model dtype or device choices.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be stored on CPU.")
    if tensor.requires_grad:
        raise ValueError(f"{name} must be detached from autograd.")
    if tensor.dtype != torch.float64:
        raise ValueError(f"{name} must use torch.float64.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values.")


def _validate_transform(transform: torch.Tensor, *, name: str) -> None:
    """Validate one unbatched SE(3) transform.

    Group consistency is handled by the shared geometry layer; this helper
    enforces only the scientific tensor and matrix-shape contract.
    """

    _validate_scientific_tensor(transform, name=name)

    if transform.shape != (4, 4):
        raise ValueError(f"{name} must have shape [4, 4].")


def _validate_vector3(vector: torch.Tensor, *, name: str) -> None:
    """Validate one unbatched three-dimensional scientific vector.

    Accelerometer measurements and the world gravity vector both use this
    representation.
    """

    _validate_scientific_tensor(vector, name=name)

    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape [3].")


def simple_accelerometer_residual(T_W_B: torch.Tensor, T_B_I: torch.Tensor, measured_specific_force_I: torch.Tensor, gravity_world: torch.Tensor) -> torch.Tensor:
    """Evaluate the gravity-alignment accelerometer residual.

    The prediction is

        f_hat_I = C^T (-R^T g_W),

    where R is the body orientation in the world frame and C is the IMU
    orientation in the body frame. The returned residual is prediction minus
    measurement.
    """

    _validate_transform(T_W_B, name="T_W_B")
    _validate_transform(T_B_I, name="T_B_I")
    _validate_vector3(measured_specific_force_I, name="measured_specific_force_I")
    _validate_vector3(gravity_world, name="gravity_world")

    R = T_W_B[:3, :3]
    C = T_B_I[:3, :3]
    predicted_specific_force = C.transpose(-1, -2) @ (-R.transpose(-1, -2) @ gravity_world)

    return predicted_specific_force - measured_specific_force_I


def simple_accelerometer_analytic_blocks(T_W_B: torch.Tensor, T_B_I: torch.Tensor, measured_specific_force_I: torch.Tensor, gravity_world: torch.Tensor, spatial_twist: torch.Tensor, *, sensor_time: float = 0.0, tau_I: float = 0.0) -> AccelerometerFactorLinearization:
    """Compute the analytic Jacobian blocks of one simple accelerometer factor.

    The body pose and IMU extrinsic use left SE(3) perturbations. Temporal
    sensitivity follows the queried trajectory pose through its spatial twist,

        H_tau_I = H_pose xi_spatial.

    This is the same one-pose gravity-only model used by the validated source
    implementation.
    """

    _validate_transform(T_W_B, name="T_W_B")
    _validate_transform(T_B_I, name="T_B_I")
    _validate_vector3(measured_specific_force_I, name="measured_specific_force_I")
    _validate_vector3(gravity_world, name="gravity_world")
    _validate_scientific_tensor(spatial_twist, name="spatial_twist")

    if spatial_twist.shape != (6,):
        raise ValueError("spatial_twist must have shape [6].")

    if not torch.isfinite(torch.tensor(sensor_time, dtype=torch.float64)):
        raise ValueError("sensor_time must be finite.")
    if not torch.isfinite(torch.tensor(tau_I, dtype=torch.float64)):
        raise ValueError("tau_I must be finite.")

    R = T_W_B[:3, :3]
    C = T_B_I[:3, :3]
    y_g = -R.transpose(-1, -2) @ gravity_world
    predicted_specific_force = C.transpose(-1, -2) @ y_g
    residual = predicted_specific_force - measured_specific_force_I

    H_pose = torch.zeros((3, 6), dtype=torch.float64)
    H_pose[:, :3] = -C.transpose(-1, -2) @ R.transpose(-1, -2) @ so3_hat(gravity_world)

    H_T_B_I = torch.zeros((3, 6), dtype=torch.float64)
    H_T_B_I[:, :3] = C.transpose(-1, -2) @ so3_hat(y_g)

    H_tau_I = (H_pose @ spatial_twist)[:, None]

    terms = AccelerometerFactorTerms(
        residual=residual,
        predicted_specific_force=predicted_specific_force,
        measured_specific_force=measured_specific_force_I,
        gravity_world=gravity_world,
        T_W_B=T_W_B,
        T_B_I=T_B_I,
        sensor_time=float(sensor_time),
        true_time=float(sensor_time + tau_I),
        tau_I=float(tau_I),
    )

    return AccelerometerFactorLinearization(residual=residual, H_pose=H_pose, H_T_B_I=H_T_B_I, H_tau_I=H_tau_I, terms=terms)


def linearize_simple_accelerometer_factor(T_W_B: torch.Tensor, T_B_I: torch.Tensor, measured_specific_force_I: torch.Tensor, gravity_world: torch.Tensor, spatial_twist: torch.Tensor, *, sensor_time: float = 0.0, tau_I: float = 0.0) -> AccelerometerFactorLinearization:
    """Linearize one conservative gravity-only accelerometer factor.

    This public entry point intentionally exposes only the simple factor during
    the first migration. The former three-pose dynamic lever-arm factor remains
    outside the initial observability baseline.
    """

    return simple_accelerometer_analytic_blocks(T_W_B, T_B_I, measured_specific_force_I, gravity_world, spatial_twist, sensor_time=sensor_time, tau_I=tau_I)


__all__ = [
    "AccelerometerFactorLinearization",
    "AccelerometerFactorTerms",
    "linearize_simple_accelerometer_factor",
    "simple_accelerometer_analytic_blocks",
    "simple_accelerometer_residual",
]