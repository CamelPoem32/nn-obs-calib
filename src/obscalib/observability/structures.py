'''Scientific observability outputs and optional window-level network features.'''

from __future__ import annotations

from dataclasses import dataclass

import torch

from obscalib.observability.layout import (
    CalibrationParameterLayout,
    NuisanceParameterLayout,
)
from obscalib.observability.timebase import ReferenceTimebase


@dataclass
class ObservabilityResult:
    '''Scientific observability output and optional window-level NN features.

    raw:
        Strategy-specific scientific result. The intended default is a
        WindowObservabilityMatrix or BatchedObservabilityMatrix containing
        the complete calibration Fisher information matrix.

    features:
        Optional network-ready observability representation with shape

            [B, d_observability]

        One vector is computed per temporal window and is later repeated
        across measurements belonging to that window.

        None means that the current experiment does not use observability
        features.

    A separate ObservabilityMapper converts the raw scientific structure
    into neural features. The estimator therefore does not need to discard
    matrix structure or numerical metadata required by scientific
    diagnostics.
    '''

    raw: object | None = None
    features: torch.Tensor | None = None


@dataclass(frozen=True)
class WindowObservabilityMatrix:
    '''Canonical scientific observability output for one temporal window.

    fisher_information_matrix:
        Calibration Fisher/Gauss-Newton information matrix in the physical
        parameterization defined by ``layout``.

        Its rows and columns contain calibration variables only.

    layout:
        Deterministic calibration-variable ordering of the Fisher matrix.

    reference_timebase:
        Reference trajectory/time grid used while constructing this window.

    projected_calibration_jacobian:
        Optional whitened calibration Jacobian after trajectory and nuisance
        directions have been marginalized.

        This is retained only for diagnostics. The Fisher information matrix
        is the canonical observability matrix.

    marginalized_nuisance_layout:
        Optional description of non-trajectory nuisance variables that were
        included in the full factor linearization and marginalized before
        constructing the calibration Fisher matrix.

        An explicit empty NuisanceParameterLayout means that there were no
        such nuisance variables. None means that this metadata was not
        supplied.

    Trajectory-state ordering is window-dependent and therefore belongs to
    linearization metadata rather than this final Fisher-matrix structure.

    Scientific tensors are deliberately stored as detached CPU float64
    tensors. Future NumPy/Numba kernels may perform the heavy computation
    internally before the estimator converts the small final result into
    this public representation.
    '''

    fisher_information_matrix: torch.Tensor
    layout: CalibrationParameterLayout
    reference_timebase: ReferenceTimebase
    projected_calibration_jacobian: torch.Tensor | None = None
    marginalized_nuisance_layout: NuisanceParameterLayout | None = None

    def __post_init__(self) -> None:
        dimension = self.layout.total_dimension

        self.reference_timebase.validate()

        _validate_scientific_tensor(
            self.fisher_information_matrix,
            name='fisher_information_matrix',
            expected_shape=(
                dimension,
                dimension,
            ),
        )

        if self.projected_calibration_jacobian is not None:
            _validate_scientific_tensor(
                self.projected_calibration_jacobian,
                name='projected_calibration_jacobian',
            )

            if (
                self.projected_calibration_jacobian.ndim != 2
                or self.projected_calibration_jacobian.shape[-1]
                != dimension
            ):
                raise ValueError(
                    'projected_calibration_jacobian must have shape '
                    '[M, D], where D matches the calibration parameter '
                    'layout.'
                )

            if (
                self.projected_calibration_jacobian.dtype
                != self.fisher_information_matrix.dtype
            ):
                raise ValueError(
                    'projected_calibration_jacobian and '
                    'fisher_information_matrix must have the same dtype.'
                )


@dataclass(frozen=True)
class BatchedObservabilityMatrix:
    '''Batched calibration Fisher matrices with one shared calibration layout.

    The batch stores one [D, D] Fisher matrix per temporal window.

    All batch elements must use the same calibration layout so the matrices
    can be stacked into

        [B, D, D].

    Reference timebases and marginalized nuisance layouts may differ between
    windows because the original raw streams remain ragged before Fisher
    matrix construction.
    '''

    fisher_information_matrix: torch.Tensor
    layout: CalibrationParameterLayout
    reference_timebases: tuple[ReferenceTimebase, ...]
    marginalized_nuisance_layouts: (
        tuple[NuisanceParameterLayout, ...]
        | None
    ) = None

    def __post_init__(self) -> None:
        dimension = self.layout.total_dimension
        tensor = self.fisher_information_matrix

        for reference_timebase in self.reference_timebases:
            reference_timebase.validate()

        _validate_scientific_tensor(
            tensor,
            name='fisher_information_matrix',
        )

        if (
            tensor.ndim != 3
            or tensor.shape[-2:] != (
                dimension,
                dimension,
            )
        ):
            raise ValueError(
                'Batched fisher_information_matrix must have '
                'shape [B, D, D].'
            )

        if tensor.shape[0] != len(self.reference_timebases):
            raise ValueError(
                'One reference timebase is required for each '
                'batch element.'
            )

        if (
            self.marginalized_nuisance_layouts is not None
            and tensor.shape[0]
            != len(self.marginalized_nuisance_layouts)
        ):
            raise ValueError(
                'One marginalized nuisance layout is required '
                'for each batch element.'
            )


def _validate_scientific_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> None:
    '''Enforce the numerical boundary for scientific observability tensors.'''

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f'{name} must be a torch.Tensor.'
        )

    if (
        expected_shape is not None
        and tuple(tensor.shape) != expected_shape
    ):
        raise ValueError(
            f'{name} must have shape {expected_shape}; '
            f'got {tuple(tensor.shape)}.'
        )

    if tensor.device.type != 'cpu':
        raise ValueError(
            f'{name} must be stored on CPU.'
        )

    if tensor.requires_grad:
        raise ValueError(
            f'{name} must be detached from autograd.'
        )

    if tensor.dtype != torch.float64:
        raise ValueError(
            f'{name} must use torch.float64 for stable '
            'scientific linear algebra.'
        )

    if not torch.isfinite(tensor).all():
        raise ValueError(
            f'{name} must contain only finite values.'
        )