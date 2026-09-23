"""Feature mappings from scientific observability results to NN-ready vectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import nn

from obscalib.observability.metrics import compute_condition_number, compute_crlb, compute_crlb_matrix, compute_singular_values
from obscalib.observability.numerics import ObservabilityNumericsConfig
from obscalib.observability.structures import ObservabilityResult, WindowObservabilityMatrix


def _window_matrix_from_result(result: ObservabilityResult) -> WindowObservabilityMatrix:
    """
    Extract the scientific one-window matrix result expected by current mappers.

    Batched mapping can be added later without weakening this explicit boundary.
    """

    if not isinstance(result, ObservabilityResult):
        raise TypeError("result must be an ObservabilityResult.")
    if not isinstance(result.raw, WindowObservabilityMatrix):
        raise TypeError("ObservabilityMapper currently requires result.raw to be a WindowObservabilityMatrix.")

    return result.raw


def _feature_result(result: ObservabilityResult, features: torch.Tensor) -> ObservabilityResult:
    """
    Return an observability result with one window-level feature row.

    Scientific ``raw`` data are preserved unchanged.
    """

    if features.ndim != 2 or features.shape[0] != 1:
        raise ValueError("Mapped observability features must have shape [1, d_observability].")
    if not torch.isfinite(features).all():
        raise ValueError("Mapped observability features must be finite before tokenization.")

    return ObservabilityResult(raw=result.raw, features=features)


class ObservabilityMapper(nn.Module, ABC):
    """
    Convert one scientific observability result into one NN feature vector.

    Output features have shape ``[1, d_observability]`` for the current
    single-window scientific boundary. The tokenizer may later repeat this vector
    across every measurement token in the same window.
    """

    @abstractmethod
    def forward(self, result: ObservabilityResult) -> ObservabilityResult:
        """
        Map one scientific observability result while preserving its raw payload.
        """


class FlattenObservabilityMapper(ObservabilityMapper):
    """
    Flatten the full physical-coordinate CRLB matrix.

    The feature dimension is ``D_C * D_C``. The underlying matrix is
    ``F_C^+`` and therefore contains parameter coupling information that is lost
    in the compact principal-direction CRLB vector.
    """

    def __init__(self, numerics: ObservabilityNumericsConfig | None = None) -> None:
        super().__init__()
        self.numerics = numerics if numerics is not None else ObservabilityNumericsConfig()

    def forward(self, result: ObservabilityResult) -> ObservabilityResult:
        """
        Return the flattened full CRLB matrix as one window-level feature row.
        """

        matrix_result = _window_matrix_from_result(result)
        crlb_matrix = compute_crlb_matrix(matrix_result, self.numerics)
        features = crlb_matrix.reshape(1, -1)

        return _feature_result(result, features)


class SoftRankObservabilityMapper(ObservabilityMapper):
    """
    Map the singular spectrum to one differentiable rank-like scalar.

    The feature is

        sum(sigmoid((sigma_i - threshold) / temperature)).

    Small temperature approaches a hard count around ``threshold``.
    """

    def __init__(self, threshold: float, temperature: float, numerics: ObservabilityNumericsConfig | None = None) -> None:
        super().__init__()

        if threshold < 0.0:
            raise ValueError("threshold must be nonnegative.")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")

        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.numerics = numerics if numerics is not None else ObservabilityNumericsConfig()

    def forward(self, result: ObservabilityResult) -> ObservabilityResult:
        """
        Return one smooth rank-like feature.
        """

        matrix_result = _window_matrix_from_result(result)
        singular_values = compute_singular_values(matrix_result, self.numerics)
        soft_rank = torch.sigmoid((singular_values - self.threshold) / self.temperature).sum()
        features = soft_rank.reshape(1, 1)

        return _feature_result(result, features)


class CRLBTanhObservabilityMapper(ObservabilityMapper):
    """
    Map principal-axis CRLB uncertainty to a bounded observability vector.

    ``compute_crlb`` returns principal variances ``1 / sigma_i**2``. Their square
    roots are principal one-sigma bounds ``1 / sigma_i``. The feature mapping is

        feature_i = 1 - tanh(std_i / alpha).

    Strongly observable directions approach one. Unobservable directions have
    infinite CRLB and map exactly to zero.
    """

    def __init__(self, alpha: float, numerics: ObservabilityNumericsConfig | None = None) -> None:
        super().__init__()

        if alpha <= 0.0:
            raise ValueError("alpha must be positive.")

        self.alpha = float(alpha)
        self.numerics = numerics if numerics is not None else ObservabilityNumericsConfig()

    def forward(self, result: ObservabilityResult) -> ObservabilityResult:
        """
        Return one bounded feature for each principal calibration direction.
        """

        matrix_result = _window_matrix_from_result(result)
        principal_crlb = compute_crlb(matrix_result, self.numerics)
        principal_standard_deviation = torch.sqrt(principal_crlb)
        features = 1.0 - torch.tanh(principal_standard_deviation / self.alpha)
        features = features.reshape(1, -1)

        return _feature_result(result, features)


class LogConditionObservabilityMapper(ObservabilityMapper):
    """
    Map the condition number to one bounded log-compressed feature.

    The internal compression is

        z = log(kappa)
        feature = z / (1 + z).

    Perfect conditioning ``kappa = 1`` maps to zero and rank deficiency
    ``kappa = +inf`` maps to one.
    """

    def __init__(self, numerics: ObservabilityNumericsConfig | None = None) -> None:
        super().__init__()
        self.numerics = numerics if numerics is not None else ObservabilityNumericsConfig()

    def forward(self, result: ObservabilityResult) -> ObservabilityResult:
        """
        Return one finite bounded condition-number feature.
        """

        matrix_result = _window_matrix_from_result(result)
        condition_number = compute_condition_number(matrix_result, self.numerics)
        condition_number = torch.clamp(condition_number, min=1.0)
        log_condition = torch.log(condition_number)

        if bool(torch.isinf(log_condition)):
            feature = torch.ones((), dtype=log_condition.dtype, device=log_condition.device)
        else:
            feature = log_condition / (1.0 + log_condition)

        return _feature_result(result, feature.reshape(1, 1))


class CombinedObservabilityMapper(ObservabilityMapper):
    """
    Concatenate several observability mappings in an explicit order.

    Each child mapper sees the same raw scientific result. Their feature blocks
    are concatenated along the last dimension.
    """

    def __init__(self, mappers: Sequence[ObservabilityMapper]) -> None:
        super().__init__()

        mappers = tuple(mappers)

        if not mappers:
            raise ValueError("CombinedObservabilityMapper requires at least one mapper.")
        if any(not isinstance(mapper, ObservabilityMapper) for mapper in mappers):
            raise TypeError("Every combined component must be an ObservabilityMapper.")

        self.mappers = nn.ModuleList(mappers)

    def forward(self, result: ObservabilityResult) -> ObservabilityResult:
        """
        Return the concatenation of all configured child-mapper features.
        """

        feature_blocks: list[torch.Tensor] = []

        for mapper in self.mappers:
            mapped = mapper(result)

            if mapped.features is None:
                raise ValueError("Every child mapper must return populated features.")

            feature_blocks.append(mapped.features)

        return _feature_result(result, torch.cat(feature_blocks, dim=-1))


__all__ = [
    "CRLBTanhObservabilityMapper",
    "CombinedObservabilityMapper",
    "FlattenObservabilityMapper",
    "LogConditionObservabilityMapper",
    "ObservabilityMapper",
    "SoftRankObservabilityMapper",
]