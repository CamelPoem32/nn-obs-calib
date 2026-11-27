"""Tensor contracts shared by the numerical and learned branches."""
from dataclasses import dataclass, field
import torch


@dataclass
class AngularCorrespondences:
    """Angular increments in radians, [B,T,3], with a boolean [B,T] mask."""
    reference_vectors: torch.Tensor
    source_vectors: torch.Tensor
    sample_mask: torch.Tensor
    weights: torch.Tensor | None = None


@dataclass
class WahbaResult:
    """Rotation mapping source vectors to reference vectors; all fields batched.

    Residuals are weighted RMS angular-vector errors in radians. Information
    is the unnormalized Gauss-Newton matrix for left rotation perturbations.
    Eigenvalues are ascending. `valid` does not imply full observability.
    """
    rotation: torch.Tensor
    information_matrix: torch.Tensor
    eigenvalues: torch.Tensor
    residual_before: torch.Tensor
    residual_after: torch.Tensor
    motion_energy: torch.Tensor
    num_correspondences: torch.Tensor
    valid: torch.Tensor


@dataclass
class NumericalFeatureBatch:
    """Features [B,C,F] in calibration_keys order; valid [B,C].

    The reference and unsupported sensors have no solution and are invalid.
    """
    values: torch.Tensor
    valid: torch.Tensor
    calibration_keys: tuple[str, ...]
    reference_key: str
    results: dict[str, WahbaResult] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.values.ndim != 3 or self.valid.shape != self.values.shape[:2] or self.valid.dtype != torch.bool:
            raise ValueError('Expected values [B,C,F] and boolean valid [B,C].')
        if self.valid.device != self.values.device or not self.values.is_floating_point():
            raise ValueError('Feature tensors must be floating point, with colocated mask.')
        if len(self.calibration_keys) != self.values.shape[1] or len(set(self.calibration_keys)) != len(self.calibration_keys) or self.reference_key not in self.calibration_keys:
            raise ValueError('Invalid calibration keys/reference.')

    def aligned(self, calibration_keys: tuple[str, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        """Route by key, not by stream insertion order; always mask reference."""
        if len(calibration_keys) != len(self.calibration_keys) or set(calibration_keys) != set(self.calibration_keys):
            raise ValueError('Numerical features and token calibration keys differ.')
        order = [self.calibration_keys.index(key) for key in calibration_keys]
        valid = self.valid[:, order].clone()
        valid[:, calibration_keys.index(self.reference_key)] = False
        return self.values[:, order], valid
