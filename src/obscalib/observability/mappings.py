"""Feature mappings from scientific observability results to NN-ready vectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from numbers import Real

import torch
from torch import nn

from obscalib.observability.metrics import compute_condition_number, compute_crlb_matrix, compute_singular_values
from obscalib.observability.numerics import ObservabilityNumericsConfig
from obscalib.observability.structures import BatchedObservabilityMatrix, ObservabilityResult, WindowObservabilityMatrix


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


def _feature_result(result: ObservabilityResult, features: torch.Tensor, *, calibration_features: dict[str, torch.Tensor] | None = None) -> ObservabilityResult:
    """Return an observability result with one feature row per temporal window while preserving scientific raw data."""

    if features.ndim != 2:
        raise ValueError("Mapped observability features must have shape [B, d_observability].")

    if not torch.isfinite(features).all():
        raise ValueError("Mapped observability features must be finite before tokenization.")

    if calibration_features is not None:
        if not calibration_features:
            raise ValueError("calibration_features must not be empty when provided.")

        for calibration_key, calibration_feature in calibration_features.items():
            if not isinstance(calibration_key, str) or not calibration_key:
                raise ValueError("Every calibration feature key must be a non-empty string.")

            if calibration_feature.ndim != 2 or calibration_feature.shape[0] != features.shape[0]:
                raise ValueError(f"Calibration features for {calibration_key!r} must have shape [B, d_calibration_observability].")

            if not torch.isfinite(calibration_feature).all():
                raise ValueError(f"Calibration features for {calibration_key!r} must be finite.")

    return ObservabilityResult(raw=result.raw, features=features, calibration_features=calibration_features)

def _calibration_features_from_layout(features: torch.Tensor, layout) -> dict[str, torch.Tensor]:
    """Split one complete physical-coordinate feature vector into calibration-local blocks."""

    if features.ndim != 2:
        raise ValueError("features must have shape [B, D].")

    if features.shape[-1] != layout.total_dimension:
        raise ValueError(f"Feature dimension {features.shape[-1]} does not match calibration layout dimension {layout.total_dimension}.")

    return {block.calibration_key: features[:, block.parameter_slice] for block in layout.blocks}

def _coordinate_crlb_standard_deviation(fisher_information_matrix: torch.Tensor, numerics: ObservabilityNumericsConfig, *, relative_tolerance: float, nullspace_tolerance: float) -> torch.Tensor:
    """
    Calculate coordinate-wise observable-subspace CRLB-like one-sigma bounds.

    Accepts either one Fisher matrix [D, D] or a batch [B, D, D].
    The returned tensor has shape [D] or [B, D], respectively.

    Numerical observability is defined in projected-Jacobian singular-value
    units even though the implementation operates on Fisher eigenvalues:

        sigma_i > max(atol, relative_tolerance * sigma_max)

    with

        lambda_i(F) = sigma_i**2.
    """

    if fisher_information_matrix.ndim == 2:
        F = fisher_information_matrix.unsqueeze(0)
        squeeze_batch = True
    elif fisher_information_matrix.ndim == 3:
        F = fisher_information_matrix
        squeeze_batch = False
    else:
        raise ValueError(
            "fisher_information_matrix must have shape [D, D] or [B, D, D]."
        )

    F = 0.5 * (
        F
        + F.transpose(-1, -2)
    )

    ##################################################
    # Fisher eigendecomposition
    ##################################################

    eigenvalues, eigenvectors = torch.linalg.eigh(
        F
    )

    ##################################################
    # PSD validation
    ##################################################

    maximum_absolute_eigenvalue = torch.amax(
        torch.abs(
            eigenvalues
        ),
        dim=-1,
    )

    default_psd_tolerance = (
        torch.finfo(F.dtype).eps
        * F.shape[-1]
        * torch.clamp(
            maximum_absolute_eigenvalue,
            min=1.0,
        )
    )

    if numerics.psd_tolerance is None:
        psd_tolerance = default_psd_tolerance
    else:
        psd_tolerance = torch.full_like(
            maximum_absolute_eigenvalue,
            float(
                numerics.psd_tolerance
            ),
        )

    if torch.any(
        eigenvalues
        < -psd_tolerance[:, None]
    ):
        raise ValueError(
            "Fisher information matrix has materially negative eigenvalues."
        )

    eigenvalues = torch.clamp(
        eigenvalues,
        min=0.0,
    )

    ##################################################
    # CRLB observability threshold
    ##################################################

    maximum_eigenvalue = torch.amax(
        eigenvalues,
        dim=-1,
    )

    sigma_max = torch.sqrt(
        maximum_eigenvalue
    )

    singular_value_tolerance = (
        float(
            relative_tolerance
        )
        * sigma_max
    )

    if numerics.absolute_tolerance is not None:
        singular_value_tolerance = torch.maximum(
            singular_value_tolerance,
            torch.full_like(
                singular_value_tolerance,
                float(
                    numerics.absolute_tolerance
                ),
            ),
        )

    fisher_eigenvalue_tolerance = (
        singular_value_tolerance.square()
    )

    observable_mask = (
        eigenvalues
        > fisher_eigenvalue_tolerance[:, None]
    )

    null_mask = (
        ~observable_mask
    )

    ##################################################
    # Coordinate variance in observable subspace
    ##################################################

    inverse_eigenvalues = torch.where(
        observable_mask,
        torch.reciprocal(
            torch.clamp(
                eigenvalues,
                min=torch.finfo(
                    eigenvalues.dtype
                ).tiny,
            )
        ),
        torch.zeros_like(
            eigenvalues
        ),
    )

    coordinate_variance = torch.sum(
        eigenvectors.square()
        * inverse_eigenvalues[:, None, :],
        dim=-1,
    )

    coordinate_standard_deviation = torch.sqrt(
        torch.clamp(
            coordinate_variance,
            min=0.0,
        )
    )

    ##################################################
    # Completely unobservable physical coordinates
    ##################################################

    nullspace_coordinate_weight = torch.sum(
        eigenvectors.square()
        * null_mask[:, None, :],
        dim=-1,
    )

    fully_unobservable_coordinates = (
        nullspace_coordinate_weight
        >= (
            1.0
            - float(
                nullspace_tolerance
            )
        )
    )

    coordinate_standard_deviation = torch.where(
        fully_unobservable_coordinates,
        torch.full_like(
            coordinate_standard_deviation,
            float("inf"),
        ),
        coordinate_standard_deviation,
    )

    if squeeze_batch:
        return coordinate_standard_deviation[0]

    return coordinate_standard_deviation


def _coordinate_alpha_vector(alpha: float | Sequence[float], layout, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Expand CRLBTanh scales to the complete calibration layout."""

    dimension = (
        layout.total_dimension
    )

    if isinstance(
        alpha,
        Real,
    ):
        return torch.full(
            (dimension,),
            float(alpha),
            dtype=dtype,
            device=device,
        )

    values = torch.as_tensor(
        tuple(
            float(value)
            for value in alpha
        ),
        dtype=dtype,
        device=device,
    ).reshape(-1)

    if values.numel() == 7:
        values = values.repeat(
            len(
                layout.blocks
            )
        )

    elif values.numel() != dimension:
        raise ValueError(
            "CRLBTanh alpha must be a scalar, "
            "a 7-element [phi, rho, tau] block, "
            f"or a full {dimension}-element calibration vector."
        )

    return values


class ObservabilityMapper(nn.Module, ABC):
    """
    Convert one scientific observability result into one NN feature vector.

    Output features have shape

        [1, d_observability]

    for the current single-window scientific boundary. The tokenizer may later
    repeat this vector across every measurement token in the same window.
    """

    @abstractmethod
    def forward(self, result: ObservabilityResult) -> ObservabilityResult:
        """
        Map one scientific observability result while preserving its raw payload.
        """


class FlattenObservabilityMapper(ObservabilityMapper):
    """
    Flatten the full physical-coordinate CRLB matrix.

    The feature dimension is

        D_C * D_C.

    The underlying matrix is F_C^+ and therefore contains parameter coupling
    information that is lost in the compact coordinate-wise CRLBTanh vector.
    """

    def __init__(self, numerics: ObservabilityNumericsConfig | None = None) -> None:
        super().__init__()

        self.numerics = (
            numerics
            if numerics is not None
            else ObservabilityNumericsConfig()
        )

    def forward(self, result: ObservabilityResult) -> ObservabilityResult:
        """
        Return the flattened full CRLB matrix as one window-level feature row.
        """

        matrix_result = _window_matrix_from_result(
            result
        )

        crlb_matrix = compute_crlb_matrix(
            matrix_result,
            self.numerics,
        )

        features = crlb_matrix.reshape(
            1,
            -1,
        )

        return _feature_result(
            result,
            features,
        )


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
            raise ValueError(
                "threshold must be nonnegative."
            )

        if temperature <= 0.0:
            raise ValueError(
                "temperature must be positive."
            )

        self.threshold = float(
            threshold
        )

        self.temperature = float(
            temperature
        )

        self.numerics = (
            numerics
            if numerics is not None
            else ObservabilityNumericsConfig()
        )

    def forward(self, result: ObservabilityResult) -> ObservabilityResult:
        """
        Return one smooth rank-like feature.
        """

        matrix_result = _window_matrix_from_result(
            result
        )

        singular_values = compute_singular_values(
            matrix_result,
            self.numerics,
        )

        soft_rank = torch.sigmoid(
            (
                singular_values
                - self.threshold
            )
            / self.temperature
        ).sum()

        features = soft_rank.reshape(
            1,
            1,
        )

        return _feature_result(
            result,
            features,
        )


class CRLBTanhObservabilityMapper(ObservabilityMapper):
    """
    Map physical coordinate-wise CRLB uncertainty to bounded observability.

    Physical coordinates follow ``CalibrationParameterLayout`` exactly:

        [phi_x, phi_y, phi_z, rho_x, rho_y, rho_z, tau]

    for every calibration key in layout order.

    The mapper first calculates coordinate-wise observable-subspace one-sigma
    uncertainty and then applies

        feature_i = 1 - tanh(std_i / alpha_i).

    Strongly observable coordinates approach one.

    Fully unobservable coordinates have infinite standard deviation and
    therefore map exactly to zero.

    ``relative_tolerance`` is deliberately independent from the stricter
    numerical tolerance used while constructing the trajectory/nuisance
    projection. Its default 1e-5 is the practical CRLB observability threshold
    established from the calibration singular spectrum.
    """

    def __init__(self, alpha: float | Sequence[float], numerics: ObservabilityNumericsConfig | None = None, *, relative_tolerance: float = 1e-5, nullspace_tolerance: float = 1e-8) -> None:
        super().__init__()

        ##################################################
        # CRLBTanh scale
        ##################################################

        if isinstance(
            alpha,
            Real,
        ):
            if float(alpha) <= 0.0:
                raise ValueError(
                    "alpha must be positive."
                )

            self.alpha: float | tuple[float, ...] = float(
                alpha
            )

        else:
            alpha_values = tuple(
                float(value)
                for value in alpha
            )

            if not alpha_values:
                raise ValueError(
                    "alpha sequence must not be empty."
                )

            if any(
                value <= 0.0
                for value in alpha_values
            ):
                raise ValueError(
                    "Every alpha value must be positive."
                )

            self.alpha = alpha_values

        ##################################################
        # Coordinate-CRLB numerical policy
        ##################################################

        if relative_tolerance < 0.0:
            raise ValueError(
                "relative_tolerance must be nonnegative."
            )

        if not 0.0 <= nullspace_tolerance < 1.0:
            raise ValueError(
                "nullspace_tolerance must lie in [0, 1)."
            )

        self.numerics = (
            numerics
            if numerics is not None
            else ObservabilityNumericsConfig()
        )

        self.relative_tolerance = float(
            relative_tolerance
        )

        self.nullspace_tolerance = float(
            nullspace_tolerance
        )

    def forward(self, result: ObservabilityResult) -> ObservabilityResult:
        """
        Return one bounded feature for every physical calibration coordinate.
        """

        if isinstance(
            result.raw,
            WindowObservabilityMatrix,
        ):
            fisher_information_matrix = (
                result.raw.fisher_information_matrix
            )

            layout = (
                result.raw.layout
            )

        elif isinstance(
            result.raw,
            BatchedObservabilityMatrix,
        ):
            fisher_information_matrix = (
                result.raw.fisher_information_matrix
            )

            layout = (
                result.raw.layout
            )

        else:
            raise TypeError(
                "CRLBTanhObservabilityMapper requires "
                "WindowObservabilityMatrix or BatchedObservabilityMatrix."
            )

        coordinate_standard_deviation = (
            _coordinate_crlb_standard_deviation(
                fisher_information_matrix,
                self.numerics,
                relative_tolerance=self.relative_tolerance,
                nullspace_tolerance=self.nullspace_tolerance,
            )
        )

        if coordinate_standard_deviation.ndim == 1:
            coordinate_standard_deviation = (
                coordinate_standard_deviation.unsqueeze(0)
            )

        alpha = _coordinate_alpha_vector(
            self.alpha,
            layout,
            dtype=coordinate_standard_deviation.dtype,
            device=coordinate_standard_deviation.device,
        )

        features = (
            1.0
            - torch.tanh(
                coordinate_standard_deviation
                / alpha[None, :]
            )
        )

        calibration_features = _calibration_features_from_layout(features, layout)

        return _feature_result(result, features, calibration_features=calibration_features)


class LogConditionObservabilityMapper(ObservabilityMapper):
    """
    Map the condition number to one bounded log-compressed feature.

    The internal compression is

        z = log(kappa)
        feature = z / (1 + z).

    Perfect conditioning kappa = 1 maps to zero and rank deficiency
    kappa = +inf maps to one.
    """

    def __init__(self, numerics: ObservabilityNumericsConfig | None = None) -> None:
        super().__init__()

        self.numerics = (
            numerics
            if numerics is not None
            else ObservabilityNumericsConfig()
        )

    def forward(self, result: ObservabilityResult) -> ObservabilityResult:
        """
        Return one finite bounded condition-number feature.
        """

        matrix_result = _window_matrix_from_result(
            result
        )

        condition_number = compute_condition_number(
            matrix_result,
            self.numerics,
        )

        condition_number = torch.clamp(
            condition_number,
            min=1.0,
        )

        log_condition = torch.log(
            condition_number
        )

        if bool(
            torch.isinf(
                log_condition
            )
        ):
            feature = torch.ones(
                (),
                dtype=log_condition.dtype,
                device=log_condition.device,
            )

        else:
            feature = (
                log_condition
                / (
                    1.0
                    + log_condition
                )
            )

        return _feature_result(
            result,
            feature.reshape(
                1,
                1,
            ),
        )


class CombinedObservabilityMapper(ObservabilityMapper):
    """
    Concatenate several observability mappings in an explicit order.

    Each child mapper sees the same raw scientific result. Their feature blocks
    are concatenated along the last dimension.
    """

    def __init__(self, mappers: Sequence[ObservabilityMapper]) -> None:
        super().__init__()

        mappers = tuple(
            mappers
        )

        if not mappers:
            raise ValueError(
                "CombinedObservabilityMapper requires at least one mapper."
            )

        if any(
            not isinstance(
                mapper,
                ObservabilityMapper,
            )
            for mapper in mappers
        ):
            raise TypeError(
                "Every combined component must be an ObservabilityMapper."
            )

        self.mappers = nn.ModuleList(
            mappers
        )

    def forward(self, result: ObservabilityResult) -> ObservabilityResult:
        """
        Return the concatenation of all configured child-mapper features.
        """

        feature_blocks: list[torch.Tensor] = []

        for mapper in self.mappers:
            mapped = mapper(
                result
            )

            if mapped.features is None:
                raise ValueError(
                    "Every child mapper must return populated features."
                )

            feature_blocks.append(
                mapped.features
            )

        return _feature_result(
            result,
            torch.cat(
                feature_blocks,
                dim=-1,
            ),
        )


__all__ = [
    "CRLBTanhObservabilityMapper",
    "CombinedObservabilityMapper",
    "FlattenObservabilityMapper",
    "LogConditionObservabilityMapper",
    "ObservabilityMapper",
    "SoftRankObservabilityMapper",
]