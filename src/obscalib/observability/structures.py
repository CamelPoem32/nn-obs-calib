"""Explicit tensor contracts exchanged by calibration pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ObservabilityResult:
    """Scientific observability output and optional NN-ready features.

    raw deliberately has no tensor-shape contract: matrix-, rank-, and
    CRLB-based estimators may preserve different scientific structures.
    features, when populated by an observability mapper, follows
    [B, N, d_obs] for tokenization.
    """

    raw: object | None = None
    features: torch.Tensor | None = None