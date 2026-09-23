"""LiDAR relative-pose observability factor.

The factor compares a measured LiDAR-frame relative pose against the motion
predicted from two body poses and the LiDAR extrinsic calibration.

The temporal calibration convention is

    t_true = t_sensor + tau_L.

All SE(3) perturbations are left perturbations with rotation-first tangent
ordering [phi, rho].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from obscalib.geometry.lie import se3_adjoint, se3_inverse, se3_left_jacobian_inverse, se3_log


@dataclass(frozen=True)
class LidarFactorTerms:
    """Reusable quantities for one LiDAR relative-pose factor.

    The prediction uses A = inv(T_0) T_1 and Z_hat = inv(T_B_L) A T_B_L.
    The residual is prediction-first, r = Log(Z_hat inv(Z_measurement)).
    """

    residual: torch.Tensor
    E_L: torch.Tensor
    T_0: torch.Tensor
    T_1: torch.Tensor
    T_B_L: torch.Tensor
    A: torch.Tensor
    Z_hat: torch.Tensor


@dataclass(frozen=True)
class LidarFactorLinearization:
    """Complete local linearization of one LiDAR relative-pose factor.

    The residual has shape [6]. Start-pose, end-pose, and extrinsic blocks have
    shape [6, 6], while the temporal-offset block has shape [6, 1].
    """

    residual: torch.Tensor
    H_start_pose: torch.Tensor
    H_end_pose: torch.Tensor
    H_T_B_L: torch.Tensor
    H_tau_L: torch.Tensor
    intermediate_terms: LidarFactorTerms


def _validate_scientific_tensor(tensor: torch.Tensor, *, name: str) -> None:
    """Validate one tensor at the scientific observability boundary.

    LiDAR factor kernels operate on detached CPU float64 tensors so their
    numerical behavior is independent of learned-model dtype and device.
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

    Group consistency is delegated to the shared geometry implementation; this
    helper enforces the scientific tensor and matrix-shape contract.
    """

    _validate_scientific_tensor(transform, name=name)

    if transform.shape != (4, 4):
        raise ValueError(f"{name} must have shape [4, 4].")


def _validate_twist(twist: torch.Tensor, *, name: str) -> None:
    """Validate one rotation-first six-dimensional spatial twist.

    Temporal LiDAR sensitivity uses spatial twists because time shifts move the
    two support poses by left perturbations in the world frame.
    """

    _validate_scientific_tensor(twist, name=name)

    if twist.shape != (6,):
        raise ValueError(f"{name} must have shape [6].")


def lidar_factor_residual_and_terms(start_body_pose: torch.Tensor, end_body_pose: torch.Tensor, body_from_lidar: torch.Tensor, lidar_measurement: torch.Tensor) -> LidarFactorTerms:
    """Evaluate one prediction-first LiDAR relative-pose residual.

    The predicted sensor motion is

        A = inv(T_0) T_1,
        Z_hat = inv(T_B_L) A T_B_L,

    and the residual is

        r_L = Log(Z_hat inv(Z_measurement)).
    """

    _validate_transform(start_body_pose, name="start_body_pose")
    _validate_transform(end_body_pose, name="end_body_pose")
    _validate_transform(body_from_lidar, name="body_from_lidar")
    _validate_transform(lidar_measurement, name="lidar_measurement")

    A = se3_inverse(start_body_pose) @ end_body_pose
    Z_hat = se3_inverse(body_from_lidar) @ A @ body_from_lidar
    E_L = Z_hat @ se3_inverse(lidar_measurement)
    residual = se3_log(E_L)

    return LidarFactorTerms(residual=residual, E_L=E_L, T_0=start_body_pose, T_1=end_body_pose, T_B_L=body_from_lidar, A=A, Z_hat=Z_hat)


def _lidar_pose_common_jacobian(terms: LidarFactorTerms) -> torch.Tensor:
    """Compute the common transported pose Jacobian block.

    Start- and end-pose derivatives differ only by sign under the selected
    prediction-first residual and left-perturbation convention.
    """

    J_l_inv_r = se3_left_jacobian_inverse(terms.residual)
    transport = se3_inverse(terms.T_B_L) @ se3_inverse(terms.T_0)

    return J_l_inv_r @ se3_adjoint(transport)


def lidar_factor_start_pose_jacobian_left(terms: LidarFactorTerms) -> torch.Tensor:
    """Compute the Jacobian with respect to the start body pose.

    The returned block has shape [6, 6] and follows the project rotation-first
    left-perturbation convention.
    """

    return -_lidar_pose_common_jacobian(terms)


def lidar_factor_end_pose_jacobian_left(terms: LidarFactorTerms) -> torch.Tensor:
    """Compute the Jacobian with respect to the end body pose.

    The returned block has shape [6, 6] and is the positive common transported
    pose block.
    """

    return _lidar_pose_common_jacobian(terms)


def lidar_factor_extrinsic_jacobian_left(terms: LidarFactorTerms) -> torch.Tensor:
    """Compute the Jacobian with respect to T_B_L.

    Both rotational and translational extrinsic components may be observable
    when the relative body motion excites their corresponding directions.
    """

    J_l_inv_r = se3_left_jacobian_inverse(terms.residual)
    identity = torch.eye(6, dtype=terms.residual.dtype, device=terms.residual.device)

    return J_l_inv_r @ se3_adjoint(se3_inverse(terms.T_B_L)) @ (se3_adjoint(terms.A) - identity)


def lidar_factor_temporal_offset_jacobian_spatial(terms: LidarFactorTerms, start_spatial_twist: torch.Tensor, end_spatial_twist: torch.Tensor) -> torch.Tensor:
    """Compute the temporal-offset Jacobian from the endpoint spatial twists.

    Because t_true = t_sensor + tau_L, increasing tau_L moves both queried body
    poses forward along the trajectory. The resulting block has shape [6, 1].
    """

    _validate_twist(start_spatial_twist, name="start_spatial_twist")
    _validate_twist(end_spatial_twist, name="end_spatial_twist")

    H_tau_L = _lidar_pose_common_jacobian(terms) @ (end_spatial_twist - start_spatial_twist)

    return H_tau_L[:, None]


def linearize_lidar_factor(start_body_pose: torch.Tensor, end_body_pose: torch.Tensor, body_from_lidar: torch.Tensor, lidar_measurement: torch.Tensor, start_spatial_twist: torch.Tensor, end_spatial_twist: torch.Tensor) -> LidarFactorLinearization:
    """Compute all analytic blocks of one LiDAR relative-pose factor.

    Residual, trajectory-pose, LiDAR-extrinsic, and temporal-offset terms are
    evaluated from the same reusable factor state.
    """

    terms = lidar_factor_residual_and_terms(start_body_pose, end_body_pose, body_from_lidar, lidar_measurement)

    return LidarFactorLinearization(
        residual=terms.residual,
        H_start_pose=lidar_factor_start_pose_jacobian_left(terms),
        H_end_pose=lidar_factor_end_pose_jacobian_left(terms),
        H_T_B_L=lidar_factor_extrinsic_jacobian_left(terms),
        H_tau_L=lidar_factor_temporal_offset_jacobian_spatial(terms, start_spatial_twist, end_spatial_twist),
        intermediate_terms=terms,
    )


__all__ = [
    "LidarFactorLinearization",
    "LidarFactorTerms",
    "lidar_factor_end_pose_jacobian_left",
    "lidar_factor_extrinsic_jacobian_left",
    "lidar_factor_residual_and_terms",
    "lidar_factor_start_pose_jacobian_left",
    "lidar_factor_temporal_offset_jacobian_spatial",
    "linearize_lidar_factor",
]