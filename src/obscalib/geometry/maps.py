"""Interfaces for mapping heterogeneous observations to canonical vectors."""

from abc import ABC, abstractmethod

import torch
from torch import nn

from obscalib.geometry.lie import se3_log, so3_log


class GeometryMap(nn.Module, ABC):
    """Map one geometry-specific observation tensor to canonical vector coordinates."""

    @abstractmethod
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return the canonical vector representation of observations."""


class SE3LogMap(GeometryMap):
    """Map SE(3) observations to [phi, rho] se(3) tangent coordinates."""

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Apply the principal SE(3) logarithm."""

        return se3_log(observations)


class SO3LogMap(GeometryMap):
    """Map SO(3) observations to rotation-vector tangent coordinates."""

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Apply the principal SO(3) logarithm."""

        return so3_log(observations)


class VectorObservationMap(GeometryMap):
    """Pass through observations that are already represented as vectors."""

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return already-vector-valued measurements unchanged."""

        if observations.ndim < 1:
            raise ValueError("Vector observations must have at least one dimension.")

        return observations