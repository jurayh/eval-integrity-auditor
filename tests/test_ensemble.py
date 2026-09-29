"""Tests for src/evalwarden/ensemble.py: pure metric helpers plus the
JUDGE-008 spike kill criterion (seeded synthetic panels)."""
from __future__ import annotations

import random

from evalwarden.ensemble import (
    ablation_value,
    majority_of_others,
    max_agreement_partner,
    panel_agreement,
    panel_verdict,
    unique_contribution,
)


def _flip(verdict: str) -> str:
    return "sol-b" if verdict == "sol-a" else "sol-a"


def _synthetic_panel(seed: int, specs: dict, n: int = 200):
    """specs: judge_id -> accuracy (float) or ("copy", other_id).

    Binary verdicts against random truth. Returns (votes_by_judge, truth).
    """
    rng = random.Random(seed)
    truth = {f"t{i:03d}": rng.choice(["sol-a", "sol-b"]) for i in range(n)}
    verdicts: dict[str, dict[str, str]] = {}
    for jid, spec in specs.items():
        per_task = {}
        for task_id, label in truth.items():
            if isinstance(spec, tuple):
                per_task[task_id] = verdicts[spec[1]][task_id]
            else:
                per_task[task_id] = label if rng.random() < spec else _flip(label)
        verdicts[jid] = per_task
    return verdicts, truth


def test_majority_of_others_decisive():
    votes = {"A": "sol-a", "B": "sol-a", "C": "sol-b"}
    assert majority_of_others(votes, "C") == "sol-a"
    # Excluding A leaves B=sol-a, C=sol-b: tie -> None
    assert majority_of_others(votes, "A") is None
    # A lone other judge is always decisive.
    assert majority_of_others({"A": "sol-a", "B": "sol-b"}, "A") == "sol-b"


def test_majority_of_others_tie_is_none():
    assert majority_of_others({"A": "sol-a", "B": "sol-b"}, "C") is None
    assert majority_of_others({"A": "sol-a"}, "A") is None


def test_unique_contribution_perfect_copier_is_zero():
    votes, _ = _synthetic_panel(7, {"A": 0.7, "B": 0.7, "C": ("copy", "A")})
    rate, differs, decisive = unique_contribution(votes, "C")
    assert rate == 0.0
    assert differs == 0
    assert decisive > 50  # enough decisive items for the 0 to mean something


def test_unique_contribution_contrarian_is_one():
    votes = {
        "A": {f"t{i}": "sol-a" for i in range(10)},
        "B": {f"t{i}": "sol-a" for i in range(10)},
        "C": {f"t{i}": "sol-b" for i in range(10)},
    }
    rate, differs, decisive = unique_contribution(votes, "C")
    assert rate == 1.0
    assert decisive == 10


def test_unique_contribution_two_judge_panel():
    # With two judges the single other judge is always decisive, so unique
    # contribution is just the disagreement rate.
    votes = {
        "A": {"t1": "sol-a"},
        "B": {"t1": "sol-b"},
    }
    rate, _, decisive = unique_contribution(votes, "A")
    assert decisive == 1
    assert rate == 1.0  # A always differs from B-the-only-other


def test_panel_verdict_chair_breaks_ties():
    assert panel_verdict({"X": "sol-a", "Y": "sol-b"}) == "sol-a"  # X < Y
    assert panel_verdict({"Y": "sol-a", "Z": "sol-b"}) == "sol-a"  # Y < Z
    assert panel_verdict({"A": "sol-a", "B": "sol-a", "C": "sol-b"}) == "sol-a"
    assert panel_verdict({}) is None


def test_max_agreement_partner_names_the_mirror():
    votes, _ = _synthetic_panel(7, {"A": 0.7, "B": 0.7, "C": ("copy", "A")})
    partner, rate = max_agreement_partner(votes, "C")
    assert partner == "A"
    assert rate == 1.0


def test_ablation_copier_adds_nothing():
    votes, truth = _synthetic_panel(7, {"A": 0.7, "B": 0.7, "C": ("copy", "A")})
    assert ablation_value(votes, truth, "C") == 0.0


def test_ablation_independent_judge_adds_value():
    votes, truth = _synthetic_panel(7, {"A": 0.7, "B": 0.7, "D": 0.7})
    assert ablation_value(votes, truth, "A") > 0.0


def test_panel_agreement_empty_truth_is_zero():
    votes, _ = _synthetic_panel(7, {"A": 0.7, "B": 0.7})
    assert panel_agreement(votes, {}) == 0.0


def test_kill_criterion_copier_separated_from_independent():
    """Spike kill criterion: Panel A (C copies A) must flag C; Panel B
    (three independent noisy judges) must flag no one. Clean separation
    or the spike stops here."""
    votes_a, _ = _synthetic_panel(7, {"A": 0.7, "B": 0.7, "C": ("copy", "A")})
    uc_c, _, _ = unique_contribution(votes_a, "C")
    assert uc_c < 0.05, f"copier C must read redundant, got {uc_c}"

    votes_b, _ = _synthetic_panel(11, {"A": 0.7, "B": 0.7, "D": 0.7})
    for jid in ("A", "B", "D"):
        rate, _, _ = unique_contribution(votes_b, jid)
        assert rate > 0.20, f"independent judge {jid} must not read redundant, got {rate}"
