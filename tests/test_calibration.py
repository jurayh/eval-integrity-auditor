"""Unit tests for the calibration helper, plus the JUDGE-007 kill criterion.

The kill criterion: a synthetic calibrated judge (stated confidence matches
empirical accuracy) and a synthetic overconfident judge (claims ~0.9, right
~60%) must separate clearly on expected calibration error. If they don't,
the metric is not worth building on.
"""
from __future__ import annotations

import random

import pytest

from evalwarden.calibration import (
    bin_pairs,
    expected_calibration_error,
    mean_signed_gap,
)


def test_bin_pairs_edges():
    pairs = [(0.0, True), (0.999, True), (1.0, False)]
    bins = bin_pairs(pairs, n_bins=10)
    assert len(bins) == 2
    assert bins[0].lo == pytest.approx(0.0)
    assert bins[0].count == 1
    # 0.999 and 1.0 both land in the closed top bin.
    assert bins[-1].hi == pytest.approx(1.0)
    assert bins[-1].count == 2
    assert bins[-1].accuracy == pytest.approx(0.5)


def test_bin_pairs_rejects_bad_bins():
    with pytest.raises(ValueError):
        bin_pairs([(0.5, True)], n_bins=0)


def test_ece_perfect_is_zero():
    pairs = [(0.7, True)] * 7 + [(0.7, False)] * 3
    assert expected_calibration_error(pairs) == 0.0


def test_ece_known_value():
    # One bin: confidence 1.0, accuracy 0.5 -> ECE 0.5.
    pairs = [(1.0, True)] * 10 + [(1.0, False)] * 10
    assert expected_calibration_error(pairs) == pytest.approx(0.5)


def test_ece_weights_bins_by_size():
    # Bin A (90% of mass): conf 0.9, acc 0.9 -> no error.
    # Bin B (10% of mass): conf 1.0, acc 0.0 -> error 1.0.
    pairs = [(0.9, True)] * 81 + [(0.9, False)] * 9 + [(1.0, False)] * 10
    assert expected_calibration_error(pairs) == pytest.approx(0.1)


def test_ece_empty_is_zero():
    assert expected_calibration_error([]) == 0.0


def test_mean_signed_gap_direction():
    over = [(0.9, i < 6) for i in range(10)]  # 60% right at 90% confidence
    assert mean_signed_gap(over) == pytest.approx(0.3)
    under = [(0.4, True)] * 10  # always right at 40% confidence
    assert mean_signed_gap(under) == pytest.approx(-0.6)
    assert mean_signed_gap([]) == 0.0


def _synthetic_pairs(n, confidences, accuracy_fn, seed):
    rng = random.Random(seed)
    pairs = []
    for i in range(n):
        c = confidences[i % len(confidences)]
        pairs.append((c, rng.random() < accuracy_fn(c)))
    return pairs


def test_kill_criterion_separates_calibrated_from_overconfident():
    calibrated = _synthetic_pairs(
        400, [0.6, 0.7, 0.8, 0.9], lambda c: c, seed=7
    )
    overconfident = _synthetic_pairs(400, [0.9], lambda c: 0.6, seed=7)
    ece_cal = expected_calibration_error(calibrated)
    ece_over = expected_calibration_error(overconfident)
    assert ece_cal < 0.08, ece_cal
    assert ece_over > 0.20, ece_over
    assert ece_over > 3 * ece_cal
