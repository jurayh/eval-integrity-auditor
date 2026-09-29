"""JUDGE-001 .. JUDGE-008 tests: positive, negative, and precision cases."""
from __future__ import annotations

import random

from evalwarden.checks.judge import (
    EnsembleRedundancyCheck,
    JudgeCalibrationCheck,
    PositionBiasCheck,
    ReferenceAgreementCheck,
    SelfConsistencyCheck,
    UnvalidatedJudgeCheck,
    VerbosityBiasCheck,
    PairOrderCheck,
)
from evalwarden.model import Confidence, Grader, Severity

from .conftest import make_judge_grader, make_judgment, make_model

c1 = UnvalidatedJudgeCheck()
c2 = PairOrderCheck()
c3 = SelfConsistencyCheck()
c4 = ReferenceAgreementCheck()
c5 = PositionBiasCheck()
c6 = VerbosityBiasCheck()
c7 = JudgeCalibrationCheck()
c8 = EnsembleRedundancyCheck()


def _ids(findings):
    return sorted({f.id for f in findings})


# ---- JUDGE-001: unvalidated model judge ----


def test_judge001_no_labels_flagged():
    model = make_model(grader=make_judge_grader(reference_labels={}))
    findings = [f for f in c1.run(model) if "calibration set" in f.title]
    assert len(findings) == 1
    assert findings[0].severity == Severity.HIGH
    assert findings[0].confidence == Confidence.HIGH


def test_judge001_validated_judge_not_flagged():
    # Labels + anchors + pinned temp: the clean configuration.
    model = make_model(grader=make_judge_grader())
    assert c1.run(model) == []


def test_judge001_missing_anchors_flagged():
    model = make_model(grader=make_judge_grader(scale_anchors={}))
    findings = [f for f in c1.run(model) if "scale anchors" in f.title]
    assert len(findings) == 1
    assert findings[0].severity == Severity.MEDIUM


def test_judge001_hot_single_sample_flagged():
    model = make_model(grader=make_judge_grader(temperature=0.7, repeats=1))
    findings = [f for f in c1.run(model) if "temperature" in f.title]
    assert len(findings) == 1
    assert findings[0].confidence == Confidence.MEDIUM


def test_judge001_hot_with_repeats_not_flagged():
    # Temperature with enough repeats is a documented tradeoff, not a finding.
    model = make_model(grader=make_judge_grader(temperature=0.7, repeats=5))
    assert all("temperature" not in f.title for f in c1.run(model))


def test_judge001_script_grader_not_flagged():
    model = make_model(grader=Grader(kind="script"))
    assert c1.run(model) == []


# ---- JUDGE-002: pair order not counterbalanced ----


def test_judge002_ab_only_flagged():
    model = make_model(grader=make_judge_grader(counterbalanced=False))
    findings = c2.run(model)
    assert len(findings) == 1
    assert findings[0].id == "JUDGE-002"
    assert findings[0].severity == Severity.HIGH
    assert findings[0].confidence == Confidence.HIGH


def test_judge002_counterbalanced_not_flagged():
    model = make_model(grader=make_judge_grader(counterbalanced=True))
    assert c2.run(model) == []


def test_judge002_pointwise_not_flagged():
    # No presentation order in a pointwise protocol: nothing to counterbalance.
    model = make_model(grader=make_judge_grader(protocol="pointwise", counterbalanced=False))
    assert c2.run(model) == []


def test_judge002_undeclared_protocol_not_flagged():
    model = make_model(grader=make_judge_grader(protocol=None, counterbalanced=False))
    assert c2.run(model) == []


# ---- JUDGE-003: self-consistency ----


def test_judge003_flipped_repeats_flagged():
    judgments = [
        make_judgment(task_id="t1", winner="sol-a", repeat_index=0),
        make_judgment(task_id="t1", winner="sol-b", repeat_index=1),
        make_judgment(task_id="t1", winner="sol-a", repeat_index=2),
    ]
    model = make_model(grader=make_judge_grader(), judgments=judgments)
    findings = c3.run(model)
    assert len(findings) == 1
    assert findings[0].id == "JUDGE-003"
    assert findings[0].severity == Severity.HIGH
    assert "t1" in findings[0].title


def test_judge003_consistent_repeats_not_flagged():
    judgments = [make_judgment(task_id="t1", winner="sol-a", repeat_index=i) for i in range(3)]
    model = make_model(grader=make_judge_grader(), judgments=judgments)
    assert c3.run(model) == []


def test_judge003_two_repeats_not_enough():
    # Two draws can disagree by chance; the check needs three.
    judgments = [
        make_judgment(task_id="t1", winner="sol-a", repeat_index=0),
        make_judgment(task_id="t1", winner="sol-b", repeat_index=1),
    ]
    model = make_model(grader=make_judge_grader(), judgments=judgments)
    assert c3.run(model) == []


def test_judge003_unstable_scores_flagged():
    judgments = [
        make_judgment(task_id="t1", winner=None, scores={"sol-a": 1.0}, repeat_index=0),
        make_judgment(task_id="t1", winner=None, scores={"sol-a": 5.0}, repeat_index=1),
        make_judgment(task_id="t1", winner=None, scores={"sol-a": 4.5}, repeat_index=2),
    ]
    model = make_model(grader=make_judge_grader(), judgments=judgments)
    findings = c3.run(model)
    assert len(findings) == 1
    assert "unstable" in findings[0].title


def test_judge003_stable_scores_not_flagged():
    judgments = [
        make_judgment(task_id="t1", winner=None, scores={"sol-a": 4.0}, repeat_index=i)
        for i in range(3)
    ]
    model = make_model(grader=make_judge_grader(), judgments=judgments)
    assert c3.run(model) == []


# ---- JUDGE-004: reference agreement ----


def _labeled_judgments(n, agree):
    labels = {f"t{i:02d}": "sol-a" for i in range(n)}
    judgments = []
    for i in range(n):
        winner = "sol-a" if i < agree else "sol-b"
        judgments.append(make_judgment(task_id=f"t{i:02d}", winner=winner))
    return labels, judgments


def test_judge004_low_agreement_flagged():
    labels, judgments = _labeled_judgments(12, 7)
    model = make_model(grader=make_judge_grader(reference_labels=labels), judgments=judgments)
    findings = c4.run(model)
    assert len(findings) == 1
    assert findings[0].id == "JUDGE-004"
    assert findings[0].severity == Severity.HIGH
    assert "7/12" in findings[0].title


def test_judge004_high_agreement_not_flagged():
    labels, judgments = _labeled_judgments(12, 11)
    model = make_model(grader=make_judge_grader(reference_labels=labels), judgments=judgments)
    assert c4.run(model) == []


def test_judge004_too_few_labels_not_flagged():
    # Five labeled items cannot support a calibration verdict.
    labels, judgments = _labeled_judgments(5, 2)
    model = make_model(grader=make_judge_grader(reference_labels=labels), judgments=judgments)
    assert c4.run(model) == []


def test_judge004_no_labels_not_flagged():
    # No labels is JUDGE-001's finding, not JUDGE-004's.
    model = make_model(grader=make_judge_grader(reference_labels={}))
    assert c4.run(model) == []


# ---- JUDGE-005: position bias ----


def _biased_pairs(n, first_wins):
    judgments = []
    for i in range(n):
        first_is_a = i % 2 == 0
        order = ("a", "b") if first_is_a else ("b", "a")
        winner_first = i < first_wins
        winner = order[0] if winner_first else order[1]
        judgments.append(make_judgment(winner=f"sol-{winner}", order=order))
    return judgments


def test_judge005_position_bias_flagged():
    model = make_model(grader=make_judge_grader(), judgments=_biased_pairs(24, 18))
    findings = c5.run(model)
    assert len(findings) == 1
    assert findings[0].id == "JUDGE-005"
    assert findings[0].severity == Severity.MEDIUM
    assert findings[0].confidence == Confidence.MEDIUM


def test_judge005_balanced_order_not_flagged():
    model = make_model(grader=make_judge_grader(), judgments=_biased_pairs(24, 12))
    assert c5.run(model) == []


def test_judge005_too_few_pairs_not_flagged():
    model = make_model(grader=make_judge_grader(), judgments=_biased_pairs(10, 9))
    assert c5.run(model) == []


# ---- JUDGE-006: verbosity bias ----


def _length_pairs(n, longer_wins):
    judgments = []
    for i in range(n):
        a_longer = i % 2 == 0
        la, lb = (700, 300) if a_longer else (300, 700)
        winner_is_longer = i < longer_wins
        if winner_is_longer:
            winner = "a" if a_longer else "b"
        else:
            winner = "b" if a_longer else "a"
        judgments.append(make_judgment(winner=f"sol-{winner}", la=la, lb=lb))
    return judgments


def test_judge006_verbosity_bias_flagged():
    model = make_model(grader=make_judge_grader(), judgments=_length_pairs(24, 18))
    findings = c6.run(model)
    assert len(findings) == 1
    assert findings[0].id == "JUDGE-006"
    assert findings[0].severity == Severity.MEDIUM
    assert findings[0].confidence == Confidence.MEDIUM


def test_judge006_balanced_lengths_not_flagged():
    model = make_model(grader=make_judge_grader(), judgments=_length_pairs(24, 12))
    assert c6.run(model) == []


def test_judge006_missing_lengths_not_flagged():
    judgments = _length_pairs(24, 18)
    for x in judgments:
        x.lengths = {}
    model = make_model(grader=make_judge_grader(), judgments=judgments)
    assert c6.run(model) == []


def test_judge006_script_grader_not_flagged():
    model = make_model(grader=Grader(kind="script"), judgments=_length_pairs(24, 20))
    for check in (c1, c2, c3, c4, c5, c6):
        assert check.run(model) == []


# ---- JUDGE-007: judge confidence calibration ----


def _calibration_judgments(n, confidences, accuracy_fn, seed, confidence=True):
    rng = random.Random(seed)
    judgments = []
    for i in range(n):
        conf = confidences[i % len(confidences)]
        correct = rng.random() < accuracy_fn(conf)
        judgments.append(
            make_judgment(
                task_id=f"cal-{i:03d}",
                winner="sol-a" if correct else "sol-b",
                confidence=conf if confidence else None,
            )
        )
    return judgments


def _calibration_model(n, confidences, accuracy_fn, seed, **kwargs):
    labels = {f"cal-{i:03d}": "sol-a" for i in range(n)}
    grader = make_judge_grader(reference_labels=labels, **kwargs)
    judgments = _calibration_judgments(n, confidences, accuracy_fn, seed)
    return make_model(grader=grader, judgments=judgments)


def test_judge007_overconfident_flagged():
    model = _calibration_model(100, [0.9], lambda c: 0.6, seed=11)
    findings = c7.run(model)
    assert len(findings) == 1
    assert findings[0].id == "JUDGE-007"
    assert "overconfident" in findings[0].title
    assert findings[0].severity == Severity.MEDIUM
    assert findings[0].confidence == Confidence.MEDIUM


def test_judge007_underconfident_flagged():
    model = _calibration_model(100, [0.4], lambda c: 0.9, seed=11)
    findings = c7.run(model)
    assert len(findings) == 1
    assert findings[0].id == "JUDGE-007"
    assert "underconfident" in findings[0].title


def test_judge007_calibrated_not_flagged():
    model = _calibration_model(100, [0.6, 0.7, 0.8, 0.9], lambda c: c, seed=11)
    assert c7.run(model) == []


def test_judge007_silent_without_confidence():
    labels = {f"cal-{i:03d}": "sol-a" for i in range(100)}
    grader = make_judge_grader(reference_labels=labels)
    judgments = _calibration_judgments(100, [0.9], lambda c: 0.6, seed=11, confidence=False)
    model = make_model(grader=grader, judgments=judgments)
    assert c7.run(model) == []


def test_judge007_silent_without_labels():
    grader = make_judge_grader(reference_labels={})
    judgments = _calibration_judgments(100, [0.9], lambda c: 0.6, seed=11)
    model = make_model(grader=grader, judgments=judgments)
    assert c7.run(model) == []


def test_judge007_silent_below_min_samples():
    # 10 overconfident judgments: clear signal, too little data to trust.
    model = _calibration_model(10, [0.9], lambda c: 0.6, seed=11)
    assert c7.run(model) == []


def test_judge007_script_grader_not_flagged():
    judgments = _calibration_judgments(100, [0.9], lambda c: 0.6, seed=11)
    model = make_model(grader=Grader(kind="script"), judgments=judgments)
    assert c7.run(model) == []


def _panel_judgments(seed, specs, n=200):
    """Synthetic panel judgments. specs: judge_id -> accuracy or ("copy", id).

    Returns (judgments, truth). Binary verdicts against random truth.
    """
    rng = random.Random(seed)
    judgments = []
    winners = {}
    for task_i in range(n):
        task_id = f"t{task_i:03d}"
        truth = rng.choice(["sol-a", "sol-b"])
        for jid, spec in specs.items():
            if isinstance(spec, tuple):
                winner = winners[(spec[1], task_id)]
            else:
                winner = truth if rng.random() < spec else (
                    "sol-b" if truth == "sol-a" else "sol-a"
                )
            winners[(jid, task_id)] = winner
            judgments.append(make_judgment(task_id=task_id, winner=winner, judge_id=jid))
    truth_map = {}
    rng2 = random.Random(seed)
    for task_i in range(n):
        truth_map[f"t{task_i:03d}"] = rng2.choice(["sol-a", "sol-b"])
    return judgments, truth_map


def _panel_model(seed, specs, n=200, with_labels=False):
    judgments, truth = _panel_judgments(seed, specs, n)
    grader = make_judge_grader(reference_labels=truth if with_labels else {})
    return make_model(grader=grader, judgments=judgments)


def test_judge008_flags_copier():
    # C always copies A: C must be flagged. A is flagged too -- with perfect
    # mutual agreement the data cannot tell who copies whom.
    model = _panel_model(7, {"A": 0.7, "B": 0.7, "C": ("copy", "A")})
    findings = c8.run(model)
    assert len(findings) == 1
    f = findings[0]
    assert f.id == "JUDGE-008"
    assert "C" in f.title and "A" in f.title
    assert f.severity == Severity.MEDIUM
    assert f.confidence == Confidence.MEDIUM
    assert any("Mirrors judge 'A'" in e for e in f.evidence)


def test_judge008_silent_on_independent_panel():
    model = _panel_model(11, {"A": 0.7, "B": 0.7, "D": 0.7})
    assert c8.run(model) == []


def test_judge008_silent_single_judge():
    model = _panel_model(7, {"A": 0.7})
    assert c8.run(model) == []


def test_judge008_silent_without_judge_ids():
    judgments = [make_judgment(task_id=f"t{i:03d}") for i in range(100)]
    model = make_model(grader=make_judge_grader(), judgments=judgments)
    assert c8.run(model) == []


def test_judge008_silent_below_min_common_items():
    model = _panel_model(7, {"A": 0.7, "B": 0.7, "C": ("copy", "A")}, n=20)
    assert c8.run(model) == []


def test_judge008_silent_between_30_and_50_items():
    # Measurable but below the flag threshold: never flag.
    model = _panel_model(7, {"A": 0.7, "B": 0.7, "C": ("copy", "A")}, n=40)
    assert c8.run(model) == []


def test_judge008_silent_non_judge_grader():
    judgments, _ = _panel_judgments(7, {"A": 0.7, "B": 0.7, "C": ("copy", "A")})
    model = make_model(grader=Grader(kind="script"), judgments=judgments)
    assert c8.run(model) == []


def test_judge008_ablation_reported_with_labels():
    model = _panel_model(7, {"A": 0.7, "B": 0.7, "C": ("copy", "A")}, with_labels=True)
    findings = c8.run(model)
    assert len(findings) == 1
    assert any("Ablation value" in e for e in findings[0].evidence)
