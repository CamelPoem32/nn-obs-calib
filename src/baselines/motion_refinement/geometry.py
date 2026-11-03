"""NumPy boundary around the project's canonical Torch Lie-group implementation.

No SO(3)/SE(3) exponential or logarithm is reimplemented here. SciPy's Slerp
is used separately by lidar_pair.py for trajectory interpolation.
"""
from __future__ import annotations

import numpy as np
import torch
from obscalib.geometry.lie import so3_exp, so3_log, se3_log


def exp_so3(phi: np.ndarray) -> np.ndarray:
    """Apply the existing canonical SO(3) exponential to float64 NumPy vectors."""
    t = torch.as_tensor(np.ascontiguousarray(np.asarray(phi, dtype=np.float64)), dtype=torch.float64)
    return so3_exp(t).detach().cpu().numpy()


def log_so3(R: np.ndarray) -> np.ndarray:
    """Apply the existing near-pi-safe SO(3) logarithm."""
    t = torch.as_tensor(np.ascontiguousarray(np.asarray(R, dtype=np.float64)), dtype=torch.float64)
    return so3_log(t).detach().cpu().numpy()


def log_se3(T: np.ndarray) -> np.ndarray:
    """SE(3) logarithm in the project's [phi, rho] tangent convention."""
    t = torch.as_tensor(np.ascontiguousarray(np.asarray(T, dtype=np.float64)), dtype=torch.float64)
    return se3_log(t).detach().cpu().numpy()


def invert_T(T: np.ndarray) -> np.ndarray:
    """Rigid-transform inverse, handling leading batch axes."""
    a = np.asarray(T, dtype=np.float64)
    out = np.zeros_like(a)
    out[..., :3, :3] = np.swapaxes(a[..., :3, :3], -1, -2)
    out[..., :3, 3] = -np.einsum('...ij,...j->...i', out[..., :3, :3], a[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def validate_rotation(R: np.ndarray, name: str = 'rotation') -> np.ndarray:
    a = np.asarray(R, dtype=np.float64)
    if a.shape != (3, 3) or not np.all(np.isfinite(a)) or not np.allclose(a.T @ a, np.eye(3), atol=1e-5, rtol=0) or abs(np.linalg.det(a) - 1) > 1e-5:
        raise ValueError(f'{name} must be a finite SO(3) matrix, shape (3,3)')
    return a.copy()


def validate_pose(T: np.ndarray, name: str = 'pose') -> np.ndarray:
    a = np.asarray(T, dtype=np.float64)
    if a.shape != (4, 4) or not np.all(np.isfinite(a)) or not np.allclose(a[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError(f'{name} must be a finite homogeneous 4x4 transform')
    validate_rotation(a[:3, :3], name)
    return a.copy()
