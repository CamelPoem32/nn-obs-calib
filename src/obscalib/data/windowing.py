"""Dataset-specific temporal window construction boundary."""

from collections.abc import Sequence

from obscalib.data.structures import WindowBatch


def build_windows(*args: object, **kwargs: object) -> Sequence[WindowBatch]:
    """Build temporal windows once dataset-specific semantics are specified."""

    # TODO: Define overlap, boundary, label-alignment, and padding semantics.
    raise NotImplementedError("Dataset-specific window construction is not implemented.")
