"""Continuous six-dimensional SO(3) coordinates for neural rotation regression.

The representation concatenates the first and second *columns* of a rotation
matrix. It is not a six-dimensional Lie algebra and must not be passed to an
SE(3) exponential or a flow-matching velocity integrator.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def matrix_to_rotation_6d(R: torch.Tensor) -> torch.Tensor:
    """Return the two first columns of ``R`` as ``[..., 6]``.

    The coordinate order is [R00, R10, R20, R01, R11, R21]. This is intentionally
    column-major for the selected columns, not a flattened 3x2 matrix in row order.
    """
    if R.ndim < 2 or R.shape[-2:] != (3, 3):
        raise ValueError(f"R must have shape [..., 3, 3], got {tuple(R.shape)}")
    if not torch.is_floating_point(R):
        raise TypeError("R must be floating point")
    return torch.cat((R[..., :, 0], R[..., :, 1]), dim=-1)


def rotation_6d_to_matrix(rotation_6d: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Recover an SO(3) matrix with differentiable Gram-Schmidt orthogonalization.

    A deterministic least-aligned coordinate axis is substituted if an input is
    degenerate. The fallback keeps matrices valid and outputs finite; the map is
    necessarily non-smooth at degenerate 6D inputs, which have no unique rotation.
    """
    if rotation_6d.ndim < 1 or rotation_6d.shape[-1] != 6:
        raise ValueError(f"rotation_6d must have shape [..., 6], got {tuple(rotation_6d.shape)}")
    if not torch.is_floating_point(rotation_6d):
        raise TypeError("rotation_6d must be floating point")
    if eps <= 0:
        raise ValueError("eps must be positive")

    a, b = rotation_6d[..., :3], rotation_6d[..., 3:]
    identity_axis = torch.zeros_like(a)
    identity_axis[..., 0] = 1.0
    a = torch.where(torch.linalg.vector_norm(a, dim=-1, keepdim=True) > eps, a, identity_axis)
    e1 = F.normalize(a, dim=-1, eps=eps)

    b_orthogonal = b - torch.sum(e1 * b, dim=-1, keepdim=True) * e1
    fallback_axis = F.one_hot(e1.abs().argmin(dim=-1), num_classes=3).to(dtype=e1.dtype)
    fallback = fallback_axis - torch.sum(e1 * fallback_axis, dim=-1, keepdim=True) * e1
    b_orthogonal = torch.where(torch.linalg.vector_norm(b_orthogonal, dim=-1, keepdim=True) > eps, b_orthogonal, fallback)
    e2 = F.normalize(b_orthogonal, dim=-1, eps=eps)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack((e1, e2, e3), dim=-1)


__all__ = ["matrix_to_rotation_6d", "rotation_6d_to_matrix"]
