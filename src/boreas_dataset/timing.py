"""Epoch-safe timestamp conversion and explicit interval/matching utilities."""

from decimal import Decimal

import numpy as np


def timestamps_to_ns(values, unit: str) -> np.ndarray:
    """Parse strings/integers without first rounding large epochs to float."""
    scale = {"s": 1_000_000_000, "us": 1_000, "ns": 1}[unit]
    try:
        result = []
        for value in values:
            scaled = Decimal(str(value)) * scale
            if not scaled.is_finite() or scaled != scaled.to_integral_value():
                raise ValueError("Timestamp has non-finite or sub-nanosecond precision")
            result.append(int(scaled))
        return np.asarray(result, dtype=np.int64)
    except (ValueError, OverflowError, ArithmeticError) as error:
        raise ValueError("Invalid timestamps") from error


def timestamps_ns_to_s(timestamps_ns) -> np.ndarray:
    """Float64 UTC seconds; retain integer nanoseconds for precise differences."""
    ns = np.asarray(timestamps_ns, dtype=np.int64)
    return (ns // 1_000_000_000).astype(float) + (ns % 1_000_000_000) * 1e-9


def relative_seconds(timestamps_ns, origin_ns=None) -> np.ndarray:
    """Subtract integer epochs before converting to seconds (empty input allowed)."""
    ns = np.asarray(timestamps_ns, dtype=np.int64)
    origin = (ns[0] if ns.size else 0) if origin_ns is None else int(origin_ns)
    return (ns - origin).astype(float) * 1e-9


def validate_timestamps(timestamps_ns) -> None:
    ns = np.asarray(timestamps_ns)
    if ns.ndim != 1 or (ns.size > 1 and np.any(np.diff(ns) <= 0)):
        raise ValueError("Timestamps must be one-dimensional and strictly increasing; no samples are silently reordered or removed")


def interval_mask(timestamps_ns, time_start=None, time_end=None) -> np.ndarray:
    """Closed interval in absolute UTC seconds; None leaves that bound open."""
    start = None if time_start is None else timestamps_to_ns([time_start], "s")[0]
    end = None if time_end is None else timestamps_to_ns([time_end], "s")[0]
    if start is not None and end is not None and start > end:
        raise ValueError("time_start must be <= time_end")
    ns = np.asarray(timestamps_ns)
    mask = np.ones(ns.size, dtype=bool)
    if start is not None:
        mask &= ns >= start
    if end is not None:
        mask &= ns <= end
    return mask


def timestamp_statistics(timestamps_ns, *, gap_factor: float = 1.5, gap_threshold_s=None) -> dict:
    """Report native acquisition gaps; gap_indices identify samples before gaps."""
    validate_timestamps(timestamps_ns)
    if not np.isfinite(gap_factor) or gap_factor <= 1:
        raise ValueError("gap_factor must exceed one")
    if gap_threshold_s is not None and (not np.isfinite(gap_threshold_s) or gap_threshold_s <= 0):
        raise ValueError("gap_threshold_s must be positive")
    ns = np.asarray(timestamps_ns, dtype=np.int64)
    dt = np.diff(ns).astype(float) * 1e-9
    median = float(np.median(dt)) if dt.size else None
    threshold = gap_threshold_s if gap_threshold_s is not None else (gap_factor * median if median else np.inf)
    return {"count": ns.size, "start_s": float(timestamps_ns_to_s(ns[:1])[0]) if ns.size else None,
            "end_s": float(timestamps_ns_to_s(ns[-1:])[0]) if ns.size else None,
            "duration_s": float((ns[-1] - ns[0]) * 1e-9) if ns.size else 0.0,
            "median_dt_s": median, "frequency_hz": 1 / median if median else None,
            "min_dt_s": float(dt.min()) if dt.size else None, "max_dt_s": float(dt.max()) if dt.size else None,
            "gap_indices": np.flatnonzero(dt > threshold), "gap_durations_s": dt[dt > threshold]}


def nearest_timestamps(reference_ns, query_ns, *, tolerance_s: float,
                       allow_unmatched: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Nearest indices and signed reference-minus-query offsets; ties choose earlier.

    A finite nonnegative tolerance is mandatory; unmatched queries raise ValueError.
    With allow_unmatched=True their indices are -1, with nearest offsets retained
    for diagnostics. Never use those -1 entries to index measurement arrays.
    """
    reference = np.asarray(reference_ns, dtype=np.int64)
    query = np.asarray(query_ns, dtype=np.int64)
    validate_timestamps(reference)
    if reference.size == 0 or query.ndim != 1:
        raise ValueError("Need nonempty reference timestamps and one-dimensional queries")
    if not np.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError("tolerance_s must be finite and nonnegative")
    right = np.clip(np.searchsorted(reference, query), 0, reference.size - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(np.abs(reference[left] - query) <= np.abs(reference[right] - query), left, right)
    delta = (reference[indices] - query).astype(float) * 1e-9
    unmatched = np.abs(delta) > tolerance_s
    if np.any(unmatched) and not allow_unmatched:
        raise ValueError("No timestamp match within tolerance_s")
    indices = np.where(unmatched, -1, indices)
    return indices, delta
