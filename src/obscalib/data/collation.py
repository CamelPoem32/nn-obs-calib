"""Dataset-specific minibatch collation boundary."""

from collections.abc import Sequence

from obscalib.data.structures import WindowBatch


def collate_windows(windows: Sequence[WindowBatch]) -> WindowBatch:
    """Collate windows after padding and missing-sensor rules are fixed."""

    # TODO: Specify per-sensor padding and missing-sensor behavior.
    raise NotImplementedError("Window collation semantics are not implemented.")
