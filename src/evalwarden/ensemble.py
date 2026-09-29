"""Ensemble redundancy metrics for judge panels.

Pure functions over per-judge verdicts. Checks call these; the math never
touches the integrity model, so it stays independently testable.

A judge panel is redundant when some members never change the panel's mind:
they agree with the rest so consistently that the panel pays for N votes and
gets fewer effective ones. Unique contribution measures, per judge, the
fraction of items where its verdict differs from the majority of the *other*
judges -- the items where it could have decided differently than the bloc.

Exact definitions:
- Panel: the set of judge ids with attributed verdicts. Common items are the
  task ids on which every panel judge rendered a verdict.
- For judge j on item i, the other judges have a *decisive* majority when one
  verdict holds strictly more than half of their votes. On a tie the item is
  skipped for j: a split panel does not determine j's verdict, so the item
  carries no redundancy signal either way.
- unique_contribution(j) = (decisive items where verdict_j differs from the
  majority of the others) / (decisive items). 0 means j never breaks from the
  bloc; 1 means it always does.
- When two judges agree perfectly, both score 0: verdict data alone cannot
  tell who copies whom, so both are reported and the panel keeps one.
- Panel verdict (for ablation): majority over the given judges, ties broken
  deterministically by the lowest judge_id (the panel chair), so even-sized
  panels always resolve.
"""
from __future__ import annotations


def majority_of_others(
    votes: dict[str, str], exclude: str
) -> str | None:
    """Majority verdict among all judges except `exclude`.

    Returns None when the other judges tie or when there are no other votes:
    a split panel determines nothing, so the item is skipped rather than
    counted for or against the judge.
    """
    counts: dict[str, int] = {}
    total = 0
    for judge_id, verdict in votes.items():
        if judge_id == exclude:
            continue
        counts[verdict] = counts.get(verdict, 0) + 1
        total += 1
    if total == 0:
        return None
    top = max(counts.values())
    leaders = [v for v, c in counts.items() if c == top]
    if len(leaders) != 1:
        return None
    return leaders[0]


def unique_contribution(
    votes_by_judge: dict[str, dict[str, str]], judge_id: str
) -> tuple[float, int, int]:
    """(rate, differs, decisive) for one judge over common items.

    `votes_by_judge` maps judge_id -> {task_id: verdict}; every judge must
    have a verdict for every task_id in the mapping (the check enforces
    common items before calling). Rate is 0.0 when there are no decisive
    items.
    """
    own = votes_by_judge.get(judge_id, {})
    differs = 0
    decisive = 0
    for task_id, verdict in own.items():
        votes = {j: v[task_id] for j, v in votes_by_judge.items()}
        majority = majority_of_others(votes, judge_id)
        if majority is None:
            continue
        decisive += 1
        if verdict != majority:
            differs += 1
    rate = differs / decisive if decisive else 0.0
    return rate, differs, decisive


def max_agreement_partner(
    votes_by_judge: dict[str, dict[str, str]], judge_id: str
) -> tuple[str | None, float]:
    """(other judge_id, agreement rate) the judge mirrors most closely.

    Agreement is over common items. Returns (None, 0.0) for a lone judge.
    Used for evidence: it names the bloc a redundant judge belongs to.
    """
    own = votes_by_judge.get(judge_id, {})
    best: str | None = None
    best_rate = 0.0
    for other, theirs in votes_by_judge.items():
        if other == judge_id or not own:
            continue
        agree = sum(1 for t, v in own.items() if theirs.get(t) == v)
        rate = agree / len(own)
        if rate > best_rate:
            best, best_rate = other, rate
    return best, best_rate


def panel_verdict(votes: dict[str, str]) -> str | None:
    """Majority verdict of the given judges; None when no votes.

    Ties are broken deterministically by the lowest judge_id (the panel
    chair). With more than two verdict options and the chair's own verdict
    outside the tie, the lowest tied verdict id wins -- still deterministic.
    """
    if not votes:
        return None
    counts: dict[str, int] = {}
    for verdict in votes.values():
        counts[verdict] = counts.get(verdict, 0) + 1
    top = max(counts.values())
    leaders = [v for v, c in counts.items() if c == top]
    if len(leaders) == 1:
        return leaders[0]
    chair_vote = votes[min(votes)]
    if chair_vote in leaders:
        return chair_vote
    return min(leaders)


def panel_agreement(
    votes_by_judge: dict[str, dict[str, str]], truth: dict[str, str]
) -> float:
    """Fraction of common items where the panel majority matches truth.

    Only items present in `truth` are counted; 0.0 when none are.
    """
    hits = 0
    total = 0
    judge_ids = sorted(votes_by_judge)
    if not judge_ids:
        return 0.0
    for task_id in votes_by_judge[judge_ids[0]]:
        label = truth.get(task_id)
        if label is None:
            continue
        votes = {j: votes_by_judge[j][task_id] for j in judge_ids}
        total += 1
        if panel_verdict(votes) == label:
            hits += 1
    return hits / total if total else 0.0


def ablation_value(
    votes_by_judge: dict[str, dict[str, str]],
    truth: dict[str, str],
    judge_id: str,
) -> float:
    """How much panel truth-agreement drops when `judge_id` is removed.

    Positive means the judge improves the panel's truth-tracking; zero means
    the panel decides just as well without it (the copier case). Needs at
    least 2 judges; 0.0 otherwise.
    """
    if len(votes_by_judge) < 2 or judge_id not in votes_by_judge:
        return 0.0
    full = panel_agreement(votes_by_judge, truth)
    reduced = {j: v for j, v in votes_by_judge.items() if j != judge_id}
    return full - panel_agreement(reduced, truth)
