"""
Differentiable batched Lie-group operations used by obs-calib.

SE(3) tangent convention
------------------------
The package uses

    xi = [phi, rho]

where

    phi : rotation-vector tangent coordinates in radians
    rho : translational tangent coordinates

For calibration corrections, the established update convention is the
left-multiplicative update

    T_next = Exp_SE3(delta_xi) @ T_current.

The functions in this module operate on arbitrary leading batch dimensions.
For example:

    phi: [..., 3]
    R:   [..., 3, 3]
    xi:  [..., 6]
    T:   [..., 4, 4]
"""

from __future__ import annotations

import math

import torch


# For theta < 1e-2 rad, use Taylor expansions instead of expressions such as
# (1 - cos(theta)) / theta^2, which suffer from cancellation in float32.
_SMALL_ANGLE_SQ = 1e-4

# The standard SO(3) logarithm divides by sin(theta), which becomes poorly
# conditioned near pi. Use a diagonal-based axis extraction in this region.
_NEAR_PI_THRESHOLD = 1e-3


def _validate_vector(tensor: torch.Tensor, size: int, name: str) -> None:
    """Validate a floating-point tensor whose last dimension has fixed size."""

    if tensor.ndim < 1 or tensor.shape[-1] != size:
        raise ValueError(
            f"{name} must have shape [..., {size}], got {tuple(tensor.shape)}."
        )

    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must be a floating-point tensor.")


def _validate_matrix(
    tensor: torch.Tensor,
    rows: int,
    cols: int,
    name: str,
) -> None:
    """Validate a floating-point tensor with fixed final matrix dimensions."""

    if tensor.ndim < 2 or tensor.shape[-2:] != (rows, cols):
        raise ValueError(
            f"{name} must have shape [..., {rows}, {cols}], "
            f"got {tuple(tensor.shape)}."
        )

    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must be a floating-point tensor.")


def _identity_3(reference: torch.Tensor) -> torch.Tensor:
    """Create a 3x3 identity matrix expanded over reference batch dimensions."""

    return torch.eye(
        3,
        dtype=reference.dtype,
        device=reference.device,
    ).expand(reference.shape[:-1] + (3, 3))


def so3_hat(phi: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation vectors to skew-symmetric so(3) matrices.

    Parameters
    ----------
    phi:
        Rotation vectors with shape [..., 3].

    Returns
    -------
    torch.Tensor
        Skew-symmetric matrices with shape [..., 3, 3].
    """

    _validate_vector(phi, 3, "phi")

    phi_x, phi_y, phi_z = phi.unbind(dim=-1)
    zero = torch.zeros_like(phi_x)

    # Construct
    #
    #     [     0  -phi_z   phi_y ]
    #     [ phi_z       0  -phi_x ]
    #     [-phi_y   phi_x       0 ]
    #
    # without modifying the input tensor.
    return torch.stack(
        (
            zero,
            -phi_z,
            phi_y,
            phi_z,
            zero,
            -phi_x,
            -phi_y,
            phi_x,
            zero,
        ),
        dim=-1,
    ).reshape(phi.shape[:-1] + (3, 3))


def so3_vee(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert skew-symmetric so(3) matrices to rotation vectors.

    Parameters
    ----------
    matrix:
        Skew-symmetric matrices with shape [..., 3, 3].

    Returns
    -------
    torch.Tensor
        Vectors with shape [..., 3].
    """

    _validate_matrix(matrix, 3, 3, "matrix")

    return torch.stack(
        (
            matrix[..., 2, 1],
            matrix[..., 0, 2],
            matrix[..., 1, 0],
        ),
        dim=-1,
    )


def _so3_coefficients(
    phi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute stable scalar coefficients used by SO(3) and SE(3) exponentials.

    For theta = ||phi||:

        A = sin(theta) / theta
        B = (1 - cos(theta)) / theta^2
        C = (theta - sin(theta)) / theta^3

    Taylor expansions are used near zero to avoid division and cancellation.
    """

    theta_sq = torch.sum(phi * phi, dim=-1)
    small_angle = theta_sq < _SMALL_ANGLE_SQ

    # The regular formulas are evaluated with safe nonzero denominators.
    # torch.where() later selects the Taylor branch for genuinely small angles.
    safe_theta_sq = torch.where(
        small_angle,
        torch.ones_like(theta_sq),
        theta_sq,
    )
    safe_theta = torch.sqrt(safe_theta_sq)

    theta_fourth = theta_sq * theta_sq
    theta_sixth = theta_fourth * theta_sq

    # Taylor expansions around theta = 0.
    A_small = (
        1.0
        - theta_sq / 6.0
        + theta_fourth / 120.0
        - theta_sixth / 5040.0
    )
    B_small = (
        0.5
        - theta_sq / 24.0
        + theta_fourth / 720.0
        - theta_sixth / 40320.0
    )
    C_small = (
        1.0 / 6.0
        - theta_sq / 120.0
        + theta_fourth / 5040.0
        - theta_sixth / 362880.0
    )

    # Standard closed-form coefficients away from zero.
    A_regular = torch.sin(safe_theta) / safe_theta
    B_regular = (1.0 - torch.cos(safe_theta)) / safe_theta_sq
    C_regular = (
        safe_theta - torch.sin(safe_theta)
    ) / (safe_theta * safe_theta_sq)

    A = torch.where(small_angle, A_small, A_regular)
    B = torch.where(small_angle, B_small, B_regular)
    C = torch.where(small_angle, C_small, C_regular)

    return A, B, C


def so3_exp(phi: torch.Tensor) -> torch.Tensor:
    """
    Apply the exponential map from so(3) vectors to SO(3).

    Uses Rodrigues' formula

        R = I + A K + B K^2,

    where K = hat(phi).

    Parameters
    ----------
    phi:
        Rotation vectors with shape [..., 3].

    Returns
    -------
    torch.Tensor
        Rotation matrices with shape [..., 3, 3].
    """

    _validate_vector(phi, 3, "phi")

    A, B, _ = _so3_coefficients(phi)

    K = so3_hat(phi)
    K_squared = K @ K
    identity = _identity_3(phi)

    return (
        identity
        + A[..., None, None] * K
        + B[..., None, None] * K_squared
    )


def so3_left_jacobian(phi: torch.Tensor) -> torch.Tensor:
    """
    Compute the SO(3) left Jacobian.

    For K = hat(phi),

        J(phi) = I + B K + C K^2.

    This Jacobian maps the translational tangent component rho to the
    translation contained in Exp_SE3([phi, rho]).
    """

    _validate_vector(phi, 3, "phi")

    _, B, C = _so3_coefficients(phi)

    K = so3_hat(phi)
    K_squared = K @ K
    identity = _identity_3(phi)

    return (
        identity
        + B[..., None, None] * K
        + C[..., None, None] * K_squared
    )


def so3_left_jacobian_inverse(phi: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse SO(3) left Jacobian.

    The inverse is

        J(phi)^-1 = I - 1/2 K + D K^2,

    with a Taylor expansion for D near zero.

    The expression is intended for the principal SO(3) logarithm region and
    therefore does not attempt to resolve singularities at multiples of 2*pi.
    """

    _validate_vector(phi, 3, "phi")

    theta_sq = torch.sum(phi * phi, dim=-1)
    small_angle = theta_sq < _SMALL_ANGLE_SQ

    # Avoid evaluating regular-form denominators at theta = 0.
    safe_theta_sq = torch.where(
        small_angle,
        torch.ones_like(theta_sq),
        theta_sq,
    )
    safe_theta = torch.sqrt(safe_theta_sq)

    theta_fourth = theta_sq * theta_sq
    theta_sixth = theta_fourth * theta_sq

    # Series for
    #
    #     D = 1/theta^2 - (1 + cos(theta)) / (2 theta sin(theta)).
    D_small = (
        1.0 / 12.0
        + theta_sq / 720.0
        + theta_fourth / 30240.0
        + theta_sixth / 1209600.0
    )

    # The half-angle form is better behaved around theta = pi than the
    # equivalent expression containing sin(theta) directly.
    half_theta = 0.5 * safe_theta
    D_regular = (
        1.0
        - half_theta * torch.cos(half_theta) / torch.sin(half_theta)
    ) / safe_theta_sq

    D = torch.where(small_angle, D_small, D_regular)

    K = so3_hat(phi)
    K_squared = K @ K
    identity = _identity_3(phi)

    return (
        identity
        - 0.5 * K
        + D[..., None, None] * K_squared
    )


def _so3_axis_near_pi(rotation: torch.Tensor) -> torch.Tensor:
    """
    Recover a rotation axis from an SO(3) matrix whose angle is near pi.

    The ordinary antisymmetric-part formula becomes ill-conditioned because
    sin(theta) approaches zero. Instead, recover the dominant axis component
    from the diagonal and infer the remaining components from symmetric terms.

    At exactly pi, the sign of the axis is mathematically non-unique.
    """

    diagonal_x = torch.clamp(
        0.5 * (rotation[..., 0, 0] + 1.0),
        min=0.0,
    )
    diagonal_y = torch.clamp(
        0.5 * (rotation[..., 1, 1] + 1.0),
        min=0.0,
    )
    diagonal_z = torch.clamp(
        0.5 * (rotation[..., 2, 2] + 1.0),
        min=0.0,
    )

    # Candidate construction divides by the dominant axis component.
    # Clamping only protects the non-selected candidate branches from zero
    # division; the selected branch always uses the largest diagonal term.
    epsilon = torch.finfo(rotation.dtype).eps

    axis_x = torch.sqrt(diagonal_x.clamp_min(epsilon))
    axis_y = torch.sqrt(diagonal_y.clamp_min(epsilon))
    axis_z = torch.sqrt(diagonal_z.clamp_min(epsilon))

    candidate_x = torch.stack(
        (
            axis_x,
            (rotation[..., 0, 1] + rotation[..., 1, 0]) / (4.0 * axis_x),
            (rotation[..., 0, 2] + rotation[..., 2, 0]) / (4.0 * axis_x),
        ),
        dim=-1,
    )

    candidate_y = torch.stack(
        (
            (rotation[..., 0, 1] + rotation[..., 1, 0]) / (4.0 * axis_y),
            axis_y,
            (rotation[..., 1, 2] + rotation[..., 2, 1]) / (4.0 * axis_y),
        ),
        dim=-1,
    )

    candidate_z = torch.stack(
        (
            (rotation[..., 0, 2] + rotation[..., 2, 0]) / (4.0 * axis_z),
            (rotation[..., 1, 2] + rotation[..., 2, 1]) / (4.0 * axis_z),
            axis_z,
        ),
        dim=-1,
    )

    # Select the numerically safest candidate using the largest diagonal term.
    dominant_index = torch.stack(
        (
            diagonal_x,
            diagonal_y,
            diagonal_z,
        ),
        dim=-1,
    ).argmax(dim=-1)

    axis = torch.where(
        (dominant_index == 0)[..., None],
        candidate_x,
        torch.where(
            (dominant_index == 1)[..., None],
            candidate_y,
            candidate_z,
        ),
    )

    # Numerical noise can make the recovered vector slightly non-unit.
    axis = axis / torch.linalg.vector_norm(
        axis,
        dim=-1,
        keepdim=True,
    ).clamp_min(epsilon)

    # Away from exactly pi, the antisymmetric part still provides the sign.
    # At exactly pi either sign represents the same rotation.
    antisymmetric_vector = so3_vee(
        rotation - rotation.transpose(-1, -2)
    )
    orientation = torch.where(
        torch.sum(axis * antisymmetric_vector, dim=-1, keepdim=True) < 0.0,
        -torch.ones_like(axis[..., :1]),
        torch.ones_like(axis[..., :1]),
    )

    return axis * orientation


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
    """
    Apply the principal logarithmic map from SO(3) to rotation vectors.

    Parameters
    ----------
    rotation:
        Rotation matrices with shape [..., 3, 3].

    Returns
    -------
    torch.Tensor
        Principal rotation vectors with shape [..., 3] and angle in [0, pi].

    Notes
    -----
    The logarithm is inherently non-unique at rotations of exactly pi.
    """

    _validate_matrix(rotation, 3, 3, "rotation")

    # Recover cos(theta) from the trace.
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_theta = torch.clamp(
        0.5 * (trace - 1.0),
        min=-1.0,
        max=1.0,
    )

    # vee(R - R^T) = 2 sin(theta) * axis.
    antisymmetric_vector = so3_vee(
        rotation - rotation.transpose(-1, -2)
    )
    sin_theta_abs = 0.5 * torch.linalg.vector_norm(
        antisymmetric_vector,
        dim=-1,
    )

    # atan2 provides better angle behavior than acos near theta = 0.
    theta = torch.atan2(sin_theta_abs, cos_theta)
    theta_sq = theta * theta

    small_angle = theta_sq < _SMALL_ANGLE_SQ
    near_pi = torch.abs(math.pi - theta) < _NEAR_PI_THRESHOLD

    # Standard branch:
    #
    #     phi = theta / (2 sin(theta)) * vee(R - R^T).
    #
    # Replace the denominator while evaluating the unused small-angle branch
    # so no division by zero enters autograd.
    safe_sin_theta = torch.where(
        small_angle,
        torch.ones_like(sin_theta_abs),
        sin_theta_abs,
    )
    coefficient_regular = theta / (2.0 * safe_sin_theta)

    theta_fourth = theta_sq * theta_sq

    # theta / (2 sin(theta))
    coefficient_small = (
        0.5
        + theta_sq / 12.0
        + 7.0 * theta_fourth / 720.0
    )

    phi_regular = torch.where(
        small_angle[..., None],
        coefficient_small[..., None] * antisymmetric_vector,
        coefficient_regular[..., None] * antisymmetric_vector,
    )

    # Near pi, recover the axis from the symmetric/diagonal matrix terms
    # because the antisymmetric part approaches zero.
    axis_near_pi = _so3_axis_near_pi(rotation)
    phi_near_pi = theta[..., None] * axis_near_pi

    return torch.where(
        near_pi[..., None],
        phi_near_pi,
        phi_regular,
    )


def se3_hat(xi: torch.Tensor) -> torch.Tensor:
    """
    Convert SE(3) tangent vectors [phi, rho] to se(3) matrices.

    Parameters
    ----------
    xi:
        Tangent vectors with shape [..., 6], ordered as

            xi[..., :3] = phi
            xi[..., 3:] = rho

    Returns
    -------
    torch.Tensor
        Lie-algebra matrices with shape [..., 4, 4].
    """

    _validate_vector(xi, 6, "xi")

    phi = xi[..., :3]
    rho = xi[..., 3:]

    xi_hat = torch.zeros(
        xi.shape[:-1] + (4, 4),
        dtype=xi.dtype,
        device=xi.device,
    )

    xi_hat[..., :3, :3] = so3_hat(phi)
    xi_hat[..., :3, 3] = rho

    return xi_hat


def se3_vee(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert se(3) matrices to tangent vectors ordered as [phi, rho].

    Parameters
    ----------
    matrix:
        Lie-algebra matrices with shape [..., 4, 4].

    Returns
    -------
    torch.Tensor
        Tangent vectors with shape [..., 6].
    """

    _validate_matrix(matrix, 4, 4, "matrix")

    phi = so3_vee(matrix[..., :3, :3])
    rho = matrix[..., :3, 3]

    return torch.cat((phi, rho), dim=-1)


def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """
    Apply the closed-form exponential map from se(3) to SE(3).

    The tangent ordering is

        xi = [phi, rho].

    The resulting transform is

        Exp(xi) = [ Exp(phi)   J(phi) rho ]
                  [    0             1    ]

    where J(phi) is the SO(3) left Jacobian.

    Parameters
    ----------
    xi:
        Tangent vectors with shape [..., 6].

    Returns
    -------
    torch.Tensor
        Homogeneous transforms with shape [..., 4, 4].
    """

    _validate_vector(xi, 6, "xi")

    phi = xi[..., :3]
    rho = xi[..., 3:]

    # Compute the SO(3) coefficients only once because both the rotation and
    # SE(3) translation use the same phi-dependent quantities.
    A, B, C = _so3_coefficients(phi)

    K = so3_hat(phi)
    K_squared = K @ K
    identity = _identity_3(phi)

    # Rodrigues formula for the rotational component.
    rotation = (
        identity
        + A[..., None, None] * K
        + B[..., None, None] * K_squared
    )

    # SO(3) left Jacobian maps rho to the translation represented by the
    # SE(3) exponential.
    left_jacobian = (
        identity
        + B[..., None, None] * K
        + C[..., None, None] * K_squared
    )
    translation = (
        left_jacobian @ rho.unsqueeze(-1)
    ).squeeze(-1)

    # Assemble the homogeneous transform while preserving arbitrary batch dims.
    transform = torch.zeros(
        xi.shape[:-1] + (4, 4),
        dtype=xi.dtype,
        device=xi.device,
    )

    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = translation
    transform[..., 3, 3] = 1.0

    return transform


def se3_log(transform: torch.Tensor) -> torch.Tensor:
    """
    Apply the principal logarithmic map from SE(3) to [phi, rho].

    Parameters
    ----------
    transform:
        Homogeneous transforms with shape [..., 4, 4].

    Returns
    -------
    torch.Tensor
        Tangent vectors with shape [..., 6], ordered as [phi, rho].

    Notes
    -----
    The translation column t of an SE(3) transform is generally not rho.
    For

        T = [R, t],

    the tangent coordinate is recovered using

        phi = Log_SO3(R)
        rho = J(phi)^-1 t.
    """

    _validate_matrix(transform, 4, 4, "transform")

    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]

    # Recover the principal rotational tangent coordinate.
    phi = so3_log(rotation)

    # Undo the SO(3) left-Jacobian coupling between phi and translation.
    left_jacobian_inverse = so3_left_jacobian_inverse(phi)
    rho = (
        left_jacobian_inverse @ translation.unsqueeze(-1)
    ).squeeze(-1)

    return torch.cat((phi, rho), dim=-1)


__all__ = [
    "se3_exp",
    "se3_hat",
    "se3_log",
    "se3_vee",
    "so3_exp",
    "so3_hat",
    "so3_left_jacobian",
    "so3_left_jacobian_inverse",
    "so3_log",
    "so3_vee",
]