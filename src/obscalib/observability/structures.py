"""Scientific observability outputs and optional window-level network features."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ObservabilityResult:
    """
    Scientific observability output and optional window-level NN features.

    raw:
        Strategy-specific scientific result. The intended future default is the
        observability/information matrix computed from the complete raw window.

    features:
        Optional network-ready observability representation with shape

            [B, d_observability].

        One vector is computed per temporal window. During final token
        construction this vector is repeated across all N measurements.

        None means that the current experiment does not use observability.
    """

    raw: object | None = None
    features: torch.Tensor | None = None