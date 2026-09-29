"""Calibration metrics for model-judge confidence scores.

Pure functions over (confidence, correct) pairs. Checks call these; the math
never touches the integrity model, so it stays independently testable.

A judge is *calibrated* when its stated confidence matches its empirical
accuracy: of all verdicts it reports at 80% confidence, about 80% are right.
Expected calibration error (ECE) measures the gap with equal-width binning.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReliabilityBin:
    """One equal-width confidence bin of a reliability diagram."""

    lo: float  # bin lower edge (inclusive)
    hi: float  # bin upper edge (exclusive, except the top bin which is closed)
    count: int  # pairs landing in this bin
    mean_confidence: float  # mean stated confidence in the bin
    accuracy: float  # fraction of pairs in the bin that were correct


def bin_pairs(
    pairs: list[tuple[float, bool]], n_bins: int = 10
) -> list[ReliabilityBin]:
    """Group (confidence, correct) pairs into equal-width confidence bins.

    Only non-empty bins are returned, in increasing confidence order.
    Confidence 1.0 lands in the top bin. Input confidences must already be
    validated into [0, 1]; this function does not re-check.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for confidence, correct in pairs:
        index = min(int(confidence * n_bins), n_bins - 1)
        buckets[index].append((confidence, correct))
    bins: list[ReliabilityBin] = []
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue
        mean_conf = sum(c for c, _ in bucket) / len(bucket)
        accuracy = sum(1 for _, ok in bucket if ok) / len(bucket)
        bins.append(
            ReliabilityBin(
                lo=i / n_bins,
                hi=(i + 1) / n_bins,
                count=len(bucket),
                mean_confidence=mean_conf,
                accuracy=accuracy,
            )
        )
    return bins


def expected_calibration_error(
    pairs: list[tuple[float, bool]], n_bins: int = 10
) -> float:
    """Mean |accuracy - mean confidence| over bins, weighted by bin size.

    Zero for a perfectly calibrated judge; 1 in the worst case. This is an
    estimate on finite samples, so even a calibrated judge shows a small
    positive value -- the noise floor shrinks as the sample grows.
    """
    n = len(pairs)
    if n == 0:
        return 0.0
    return sum(
        (b.count / n) * abs(b.accuracy - b.mean_confidence)
        for b in bin_pairs(pairs, n_bins)
    )


def mean_signed_gap(pairs: list[tuple[float, bool]]) -> float:
    """Mean (confidence - correctness): positive means overconfident.

    This is the *direction* of miscalibration; ECE is its magnitude. A judge
    can be miscalibrated in both directions at once (high ECE, near-zero gap).
    """
    if not pairs:
        return 0.0
    return sum(c - (1.0 if ok else 0.0) for c, ok in pairs) / len(pairs)
