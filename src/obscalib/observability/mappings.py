"""Feature mappings derived from a window-level observability matrix."""

from abc import ABC, abstractmethod

from torch import nn

from obscalib.observability.structures import ObservabilityResult


class ObservabilityMapper(nn.Module, ABC):
    """
    Convert a scientific observability result into one window-level feature vector.

    Output features follow

        [B, d_observability].

    The final tokenizer repeats each feature vector for every measurement in the
    corresponding window.
    """

    @abstractmethod
    def forward(self, result: ObservabilityResult) -> ObservabilityResult:
        """Return the result with network-ready features while preserving raw data."""


class FlattenObservabilityMapper(ObservabilityMapper):
    """Future mapping that flattens the complete observability matrix."""

    def forward(self, result: ObservabilityResult) -> ObservabilityResult:
        # TODO: Flatten the matrix as [B, matrix_rows * matrix_cols].
        # TODO: Decide whether raw values require normalization before training.
        raise NotImplementedError("Flattened observability features are not implemented.")


class SoftRankObservabilityMapper(ObservabilityMapper):
    """
    Future smooth rank-like representation based on observability singular values.

    Intended form:

        soft_rank = sum(sigmoid((sigma_i - threshold) / temperature))

    where small temperature approaches a hard threshold count.
    """

    def __init__(self, threshold: float, temperature: float) -> None:
        super().__init__()

        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")

        self.threshold = threshold
        self.temperature = temperature

    def forward(self, result: ObservabilityResult) -> ObservabilityResult:
        # TODO: Define singular-value normalization before applying the threshold.
        # TODO: Return one smooth rank feature per window.
        raise NotImplementedError("Soft-rank observability features are not implemented.")


class CRLBTanhObservabilityMapper(ObservabilityMapper):
    """
    Future bounded CRLB-like uncertainty representation.

    Intended pipeline:

        observability matrix
            -> inverse / pseudoinverse
            -> directional standard deviations
            -> bounded feature such as 1 - tanh(std / alpha).
    """

    def __init__(self, alpha: float) -> None:
        super().__init__()

        if alpha <= 0.0:
            raise ValueError("alpha must be positive.")

        self.alpha = alpha

    def forward(self, result: ObservabilityResult) -> ObservabilityResult:
        # TODO: Define stable inversion / pseudoinversion and the exact directions
        # represented by the resulting standard-deviation vector.
        raise NotImplementedError("CRLB-like observability features are not implemented.")


class LogConditionObservabilityMapper(ObservabilityMapper):
    """Future compressed condition-number representation."""

    def forward(self, result: ObservabilityResult) -> ObservabilityResult:
        # TODO: Compute condition number from singular values.
        # TODO: Use a bounded or compressed representation such as log1p(kappa)
        # and decide whether additional clipping is required near singularity.
        raise NotImplementedError("Condition-number observability features are not implemented.")


class CombinedObservabilityMapper(ObservabilityMapper):
    """
    Future concatenation of several observability representations.

    Candidate features include flattened matrix values, soft rank, CRLB-like
    directional uncertainties, and compressed condition-number information.
    """

    def forward(self, result: ObservabilityResult) -> ObservabilityResult:
        # TODO: Define the selected component mappings and their concatenation order.
        raise NotImplementedError("Combined observability features are not implemented.")