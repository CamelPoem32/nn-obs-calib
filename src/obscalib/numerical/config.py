"""Configuration for independent angular-motion Wahba estimates."""
from dataclasses import dataclass
import math
from typing import Literal


@dataclass(frozen=True)
class WahbaConfig:
    """Use only the supplied, prior-corrected timestamps; never estimate a lag.

    Motion energy is mean squared angular increment (rad²), before optional
    vector normalization. Rank-deficient but nonstatic estimates are retained.
    """

    reference_key: str | None = None
    reference_stream: str | None = None
    source_streams: tuple[tuple[str, str], ...] = ()
    min_correspondences: int = 3
    min_motion_energy: float = 1e-8
    normalize_vectors: bool = False
    use_weights: bool = False
    max_gyro_gap_s: float | None = None
    coverage_tolerance_s: float = 1e-6

    def __post_init__(self) -> None:
        if self.min_correspondences < 1:
            raise ValueError('min_correspondences must be positive.')
        for name in ('min_motion_energy', 'coverage_tolerance_s'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive.')
        if self.max_gyro_gap_s is not None and (not math.isfinite(self.max_gyro_gap_s) or self.max_gyro_gap_s <= 0):
            raise ValueError('max_gyro_gap_s must be finite and positive.')
        if len(dict(self.source_streams)) != len(self.source_streams):
            raise ValueError('Duplicate calibration keys in source_streams.')


@dataclass(frozen=True)
class NumericalFeatureConfig:
    """SO(3) output encoding and optional reliability features.

    6D uses concatenated first two columns by default; rows are available for
    compatibility with alternate Gram-Schmidt conventions.
    """

    rotation_mode: Literal['3d', '6d'] = '3d'
    rotation_6d_layout: Literal['columns', 'rows'] = 'columns'
    include_information_matrix: bool = True
    include_eigenvalues: bool = True
    include_residuals: bool = True
    include_motion_energy: bool = True

    def __post_init__(self) -> None:
        if self.rotation_mode not in ('3d', '6d') or self.rotation_6d_layout not in ('columns', 'rows'):
            raise ValueError('Unsupported rotation representation.')

    @property
    def feature_dim(self) -> int:
        return (3 if self.rotation_mode == '3d' else 6) + 6 * self.include_information_matrix + 3 * self.include_eigenvalues + 2 * self.include_residuals + self.include_motion_energy
