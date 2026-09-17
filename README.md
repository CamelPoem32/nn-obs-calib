# obs-calib

obs-calib is a modular PyTorch scaffold for a future neural sensor-calibration
pipeline. This milestone separates implemented tensor and network mechanics
from research-dependent geometry, observability, update, and loss definitions.

Implemented components include typed data contracts, mask-aware timestamp
sorting, token encoders, a Transformer with configurable learned summary
tokens, a configurable shared MLP, reusable calibration heads, and weighted
loss aggregation.

Geometry maps, observability estimators, state updates, dataset-specific
windowing/collation, geometric losses, and sequential rollout remain explicit
placeholders. They raise NotImplementedError instead of returning scientifically
meaningless values.

## Development

The package uses a src layout and requires Python 3.10 or newer.

    python -m pip install -e ".[test]"
    pytest

Tensor layouts are batch-first: measurements use [B, N, D], validity masks use
[B, N] with True meaning valid, and the Transformer prepends K learned summary
tokens to produce [B, K + N, D].
