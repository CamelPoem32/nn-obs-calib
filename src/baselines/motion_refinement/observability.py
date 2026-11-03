"""Local measurement observability, with gyro-bias/time columns marginalized.

The information matrix deliberately excludes calibration priors; otherwise a
strong prior could incorrectly make completely unexcited motion look observable.
These diagnostics are local linearizations, not calibrated uncertainty bounds.
"""
from __future__ import annotations

import numpy as np
from .structures import MotionRefinementConfig, RotationObservability


def analyze_rotation_observability(J_measurement: np.ndarray, config: MotionRefinementConfig) -> RotationObservability:
    """TSVD on the Schur/marginalized rotation Jacobian (first 3 columns)."""
    J = np.asarray(J_measurement, dtype=np.float64)
    if J.ndim != 2 or J.shape[1] < 3 or not np.isfinite(J).all():
        raise ValueError('Expected finite measurement Jacobian [num_residuals,>=3]')
    Jrot = J[:, :3]
    nuisance = J[:, 3:]
    if nuisance.shape[1]:
        U, s_n, _ = np.linalg.svd(nuisance, full_matrices=True)
        t = max(config.observability_atol, (s_n[0] if s_n.size else 0) * config.observability_rtol)
        rank_n = int((s_n > t).sum())
        Jrot = U[:, rank_n:].T @ Jrot
    _, s, Vt = np.linalg.svd(Jrot, full_matrices=False)
    s = np.pad(s, (0, max(0, 3 - len(s))))[:3]
    threshold = max(config.observability_atol, s[0] * config.observability_rtol)
    mask = s > threshold
    rank = int(mask.sum())
    V = Vt.T
    projector = V[:, mask] @ V[:, mask].T if rank else np.zeros((3, 3))
    condition = float(s[0] / s[rank - 1]) if rank > 0 else float('inf')
    return RotationObservability(singular_values=s, rank=rank, projector=projector, condition_number=condition, num_nuisance_parameters=nuisance.shape[1])
