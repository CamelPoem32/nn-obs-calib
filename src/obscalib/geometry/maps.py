"""Interfaces for mapping heterogeneous observations to canonical vectors."""

from abc import ABC, abstractmethod

import torch
from torch import nn

from obscalib.geometry.lie import se3_log, so3_log


class GeometryMap(nn.Module, ABC):
    """Map one geometry-specific observation tensor to canonical coordinates."""

    @abstractmethod
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return a canonical vector representation of observations."""


class SE3LogMap(GeometryMap):
    """Future logarithmic map from SE(3) observations to se(3) coordinates."""

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Map homogeneous transforms after perturbation conventions are fixed."""

        return se3_log(observations)


class SO3LogMap(GeometryMap):
    """Future logarithmic map from SO(3) observations to so(3) coordinates."""

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Map rotations after logarithm and coordinate conventions are fixed."""

        return so3_log(observations)


class VectorObservationMap(GeometryMap):
    """Return observations already represented as [..., d] vectors."""

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim < 1:
            raise ValueError(...)
        return observations
