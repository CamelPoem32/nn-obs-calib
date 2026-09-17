"""Named scalar loss components and configurable aggregation."""

from dataclasses import dataclass, replace

import torch

from obscalib.config import LossWeights


@dataclass
class LossComponents:
    """Precomputed named scalar terms and their optional weighted total."""

    rotation: torch.Tensor
    translation: torch.Tensor
    time_offset: torch.Tensor
    change_event: torch.Tensor
    change_time: torch.Tensor
    prior: torch.Tensor
    consistency: torch.Tensor
    total: torch.Tensor | None = None


def combine_loss_components(
    components: LossComponents,
    weights: LossWeights,
) -> LossComponents:
    """Return the components with a differentiable weighted total attached."""

    total = (
        weights.lambda_rotation * components.rotation
        + weights.lambda_translation * components.translation
        + weights.lambda_time_offset * components.time_offset
        + weights.lambda_change * components.change_event
        + weights.lambda_change_time * components.change_time
        + weights.lambda_prior * components.prior
        + weights.lambda_consistency * components.consistency
    )
    return replace(components, total=total)
