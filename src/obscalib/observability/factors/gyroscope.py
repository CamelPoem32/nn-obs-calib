"""Gyroscope factor residuals and analytic Jacobians for observability extraction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from obscalib.geometry.lie import so3_exp, so3_left_jacobian, so3_left_jacobian_inverse, so3_log


GYROSCOPE_BIAS_PARAMETER_NAMES = ("b_g_x", "b_g_y", "b_g_z")


def gyroscope_bias_nuisance_key(calibration_key: str) -> str:
    """Return the deterministic nuisance-layout key for one IMU gyroscope bias."""

    if not calibration_key:
        raise ValueError("calibration_key must be non-empty.")

    return f"{calibration_key}.gyro_bias"


@dataclass(frozen=True)
class GyroscopeFactorTerms:
    """Reusable terms for one gyroscope propagation factor."""

    residual: torch.Tensor
    E_I: torch.Tensor
    R_k: torch.Tensor
    R_k1: torch.Tensor
    C: torch.Tensor
    phi: torch.Tensor
    Delta_R: torch.Tensor
    Q: torch.Tensor
    delta_t: float
    omega_start_shifted: torch.Tensor
    omega_end_shifted: torch.Tensor
    lower_sensor_time: float
    upper_sensor_time: float


@dataclass(frozen=True)
class GyroscopeFactorLinearization:
    """Complete analytic linearization of one gyroscope propagation factor."""

    residual: torch.Tensor
    H_start_pose: torch.Tensor
    H_end_pose: torch.Tensor
    H_T_B_I: torch.Tensor
    H_b_g: torch.Tensor
    H_tau_I: torch.Tensor
    intermediate_terms: GyroscopeFactorTerms


def _validate_scientific_tensor(tensor: torch.Tensor, *, name: str) -> None:
    """Validate the CPU float64 convention used by scientific observability kernels."""

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


def _validate_gyroscope_signal(sample_times: torch.Tensor, omega_samples: torch.Tensor) -> None:
    """Validate one unbatched three-axis gyroscope signal."""

    _validate_scientific_tensor(sample_times, name="sample_times")
    _validate_scientific_tensor(omega_samples, name="omega_samples")

    if sample_times.ndim != 1:
        raise ValueError("sample_times must have shape [N].")
    if omega_samples.ndim != 2 or omega_samples.shape[1] != 3:
        raise ValueError("omega_samples must have shape [N, 3].")
    if sample_times.shape[0] != omega_samples.shape[0]:
        raise ValueError("sample_times and omega_samples must contain the same number of samples.")
    if sample_times.numel() < 2:
        raise ValueError("At least two gyroscope samples are required.")
    if torch.any(sample_times[1:] <= sample_times[:-1]):
        raise ValueError("sample_times must be strictly increasing.")


def _validate_transform(transform: torch.Tensor, *, name: str) -> None:
    """Validate one scientific SE(3) transform."""

    _validate_scientific_tensor(transform, name=name)

    if transform.shape != (4, 4):
        raise ValueError(f"{name} must have shape [4, 4].")


def _validate_vector3(vector: torch.Tensor, *, name: str) -> None:
    """Validate one scientific three-vector."""

    _validate_scientific_tensor(vector, name=name)

    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape [3].")


def _as_scalar(value: float | torch.Tensor, *, reference: torch.Tensor, name: str) -> torch.Tensor:
    """Convert one scalar to the dtype and device of a scientific reference tensor."""

    scalar = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)

    if scalar.numel() != 1:
        raise ValueError(f"{name} must be scalar.")
    if not torch.isfinite(scalar).all():
        raise ValueError(f"{name} must be finite.")

    return scalar.reshape(())


def interpolate_gyroscope_linear(sample_times: torch.Tensor, omega_samples: torch.Tensor, query_times: float | torch.Tensor) -> torch.Tensor:
    """Evaluate a gyroscope signal using piecewise-linear interpolation without extrapolation."""

    _validate_gyroscope_signal(sample_times, omega_samples)

    query = torch.as_tensor(query_times, dtype=sample_times.dtype, device=sample_times.device)

    if query.ndim > 1:
        raise ValueError("query_times must be scalar or have shape [M].")
    if not torch.isfinite(query).all():
        raise ValueError("query_times must contain only finite values.")
    if torch.any(query < sample_times[0]) or torch.any(query > sample_times[-1]):
        raise ValueError("query_times must lie inside the gyroscope timestamp range.")

    upper_indices = torch.searchsorted(sample_times, query, right=True).clamp(min=1, max=sample_times.numel() - 1)
    lower_indices = upper_indices - 1

    lower_times = sample_times[lower_indices]
    upper_times = sample_times[upper_indices]
    alpha = (query - lower_times) / (upper_times - lower_times)

    return omega_samples[lower_indices] + alpha[..., None] * (omega_samples[upper_indices] - omega_samples[lower_indices])


def integrate_gyroscope_signal_linear(sample_times: torch.Tensor, omega_samples: torch.Tensor, lower_time: float | torch.Tensor, upper_time: float | torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Integrate a bias-corrected piecewise-linear gyroscope signal exactly by trapezoids."""

    _validate_gyroscope_signal(sample_times, omega_samples)
    _validate_vector3(bias, name="bias")

    lower = _as_scalar(lower_time, reference=sample_times, name="lower_time")
    upper = _as_scalar(upper_time, reference=sample_times, name="upper_time")

    if upper <= lower:
        raise ValueError("upper_time must be greater than lower_time.")
    if lower < sample_times[0] or upper > sample_times[-1]:
        raise ValueError("Gyroscope integration bounds must lie inside the sampled timestamp range.")

    interior_times = sample_times[(sample_times > lower) & (sample_times < upper)]
    integration_times = torch.cat((lower.reshape(1), interior_times, upper.reshape(1)))
    interpolated_omega = interpolate_gyroscope_linear(sample_times, omega_samples, integration_times)
    corrected_omega = interpolated_omega - bias[None, :]
    time_steps = integration_times[1:] - integration_times[:-1]

    return torch.sum(0.5 * time_steps[:, None] * (corrected_omega[:-1] + corrected_omega[1:]), dim=0)


def gyroscope_interval_is_supported(sample_times: torch.Tensor, true_start_time: float | torch.Tensor, true_end_time: float | torch.Tensor, imu_time_offset: float | torch.Tensor) -> bool:
    """Return whether one trajectory interval can be integrated without gyroscope extrapolation."""

    if sample_times.ndim != 1 or sample_times.numel() < 2:
        return False

    t_start = float(torch.as_tensor(true_start_time).item())
    t_end = float(torch.as_tensor(true_end_time).item())
    tau = float(torch.as_tensor(imu_time_offset).item())

    if not t_end > t_start:
        return False

    lower_sensor_time = t_start - tau
    upper_sensor_time = t_end - tau

    return lower_sensor_time >= float(sample_times[0].item()) and upper_sensor_time <= float(sample_times[-1].item())


def gyroscope_factor_residual_and_terms(start_body_pose: torch.Tensor, end_body_pose: torch.Tensor, body_from_imu: torch.Tensor, gyro_bias: torch.Tensor, imu_time_offset: float | torch.Tensor, true_start_time: float | torch.Tensor, true_end_time: float | torch.Tensor, imu_sensor_timestamps: torch.Tensor, gyroscope_samples: torch.Tensor) -> GyroscopeFactorTerms:
    """Evaluate one prediction-first gyroscope propagation residual and its reusable terms.

    The project time-offset convention is

        t_true = t_sensor + tau_I,

    therefore the gyroscope signal corresponding to a trajectory time is queried at

        t_sensor = t_true - tau_I.

    The residual is

        E_I = R_k C Exp(phi) C^T R_{k+1}^T,
        r_I = Log(E_I),

    where C is the rotation of T_B_I and phi is the integrated bias-corrected gyroscope signal.
    """

    _validate_transform(start_body_pose, name="start_body_pose")
    _validate_transform(end_body_pose, name="end_body_pose")
    _validate_transform(body_from_imu, name="body_from_imu")
    _validate_vector3(gyro_bias, name="gyro_bias")
    _validate_gyroscope_signal(imu_sensor_timestamps, gyroscope_samples)

    t_start = _as_scalar(true_start_time, reference=imu_sensor_timestamps, name="true_start_time")
    t_end = _as_scalar(true_end_time, reference=imu_sensor_timestamps, name="true_end_time")
    tau = _as_scalar(imu_time_offset, reference=imu_sensor_timestamps, name="imu_time_offset")

    if t_end <= t_start:
        raise ValueError("true_end_time must be greater than true_start_time.")

    lower_sensor_time = t_start - tau
    upper_sensor_time = t_end - tau

    phi = integrate_gyroscope_signal_linear(imu_sensor_timestamps, gyroscope_samples, lower_sensor_time, upper_sensor_time, gyro_bias)

    R_k = start_body_pose[:3, :3]
    R_k1 = end_body_pose[:3, :3]
    C = body_from_imu[:3, :3]
    Delta_R = so3_exp(phi)
    Q = C @ Delta_R @ C.transpose(-1, -2)
    E_I = R_k @ Q @ R_k1.transpose(-1, -2)
    residual = so3_log(E_I)

    omega_start_shifted = interpolate_gyroscope_linear(imu_sensor_timestamps, gyroscope_samples, lower_sensor_time) - gyro_bias
    omega_end_shifted = interpolate_gyroscope_linear(imu_sensor_timestamps, gyroscope_samples, upper_sensor_time) - gyro_bias

    return GyroscopeFactorTerms(residual=residual, E_I=E_I, R_k=R_k, R_k1=R_k1, C=C, phi=phi, Delta_R=Delta_R, Q=Q, delta_t=float((t_end - t_start).item()), omega_start_shifted=omega_start_shifted, omega_end_shifted=omega_end_shifted, lower_sensor_time=float(lower_sensor_time.item()), upper_sensor_time=float(upper_sensor_time.item()))


def gyroscope_factor_start_pose_jacobian_left(terms: GyroscopeFactorTerms) -> torch.Tensor:
    """Return the start-pose block H_T_k = [J_l^-1(r_I), 0], shape [3, 6]."""

    H_start_pose = torch.zeros((3, 6), dtype=terms.residual.dtype, device=terms.residual.device)
    H_start_pose[:, :3] = so3_left_jacobian_inverse(terms.residual)

    return H_start_pose


def gyroscope_factor_end_pose_jacobian_left(terms: GyroscopeFactorTerms) -> torch.Tensor:
    """Return the end-pose block H_T_k1 = [-J_l^-1(r_I) E_I, 0], shape [3, 6]."""

    J_l_inv_r = so3_left_jacobian_inverse(terms.residual)
    H_end_pose = torch.zeros((3, 6), dtype=terms.residual.dtype, device=terms.residual.device)
    H_end_pose[:, :3] = -J_l_inv_r @ terms.E_I

    return H_end_pose


def gyroscope_factor_extrinsic_jacobian_left(terms: GyroscopeFactorTerms) -> torch.Tensor:
    """Return the T_B_I block; gyro-only translation columns are exactly zero."""

    J_l_inv_r = so3_left_jacobian_inverse(terms.residual)
    H_T_B_I = torch.zeros((3, 6), dtype=terms.residual.dtype, device=terms.residual.device)
    H_T_B_I[:, :3] = J_l_inv_r @ terms.R_k @ (torch.eye(3, dtype=terms.residual.dtype, device=terms.residual.device) - terms.Q)

    return H_T_B_I


def gyroscope_factor_bias_jacobian(terms: GyroscopeFactorTerms) -> torch.Tensor:
    """Return the constant gyroscope-bias block H_b_g with shape [3, 3]."""

    J_l_inv_r = so3_left_jacobian_inverse(terms.residual)
    J_l_phi = so3_left_jacobian(terms.phi)

    return -J_l_inv_r @ terms.R_k @ terms.C @ J_l_phi * terms.delta_t


def gyroscope_factor_temporal_offset_jacobian(terms: GyroscopeFactorTerms) -> torch.Tensor:
    """Return H_tau_I with shape [3, 1] for the convention t_sensor = t_true - tau_I."""

    J_l_inv_r = so3_left_jacobian_inverse(terms.residual)
    J_l_phi = so3_left_jacobian(terms.phi)

    # Leibniz rule for integration limits [t_k - tau_I, t_k1 - tau_I].
    dphi_d_tau = terms.omega_start_shifted - terms.omega_end_shifted
    H_tau_I = J_l_inv_r @ terms.R_k @ terms.C @ J_l_phi @ dphi_d_tau

    return H_tau_I[:, None]


def linearize_gyroscope_factor(start_body_pose: torch.Tensor, end_body_pose: torch.Tensor, body_from_imu: torch.Tensor, gyro_bias: torch.Tensor, imu_time_offset: float | torch.Tensor, true_start_time: float | torch.Tensor, true_end_time: float | torch.Tensor, imu_sensor_timestamps: torch.Tensor, gyroscope_samples: torch.Tensor) -> GyroscopeFactorLinearization:
    """Compute all analytic blocks of one gyroscope propagation factor."""

    terms = gyroscope_factor_residual_and_terms(start_body_pose, end_body_pose, body_from_imu, gyro_bias, imu_time_offset, true_start_time, true_end_time, imu_sensor_timestamps, gyroscope_samples)

    return GyroscopeFactorLinearization(residual=terms.residual, H_start_pose=gyroscope_factor_start_pose_jacobian_left(terms), H_end_pose=gyroscope_factor_end_pose_jacobian_left(terms), H_T_B_I=gyroscope_factor_extrinsic_jacobian_left(terms), H_b_g=gyroscope_factor_bias_jacobian(terms), H_tau_I=gyroscope_factor_temporal_offset_jacobian(terms), intermediate_terms=terms)


__all__ = [
    "GYROSCOPE_BIAS_PARAMETER_NAMES",
    "GyroscopeFactorLinearization",
    "GyroscopeFactorTerms",
    "gyroscope_bias_nuisance_key",
    "gyroscope_factor_bias_jacobian",
    "gyroscope_factor_end_pose_jacobian_left",
    "gyroscope_factor_extrinsic_jacobian_left",
    "gyroscope_factor_residual_and_terms",
    "gyroscope_factor_start_pose_jacobian_left",
    "gyroscope_factor_temporal_offset_jacobian",
    "gyroscope_interval_is_supported",
    "integrate_gyroscope_signal_linear",
    "interpolate_gyroscope_linear",
    "linearize_gyroscope_factor",
]