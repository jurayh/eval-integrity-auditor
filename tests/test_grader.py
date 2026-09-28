"""GRAD-001 / GRAD-002 tests."""
from __future__ import annotations

from evalint.checks.grader import EmptyPathCheck, VerifierWritableCheck
from evalint.model import Attempt, Confidence, Grader, Severity

from .conftest import make_model

writable_check = VerifierWritableCheck()
empty_check = EmptyPathCheck()


def test_writable_verifier_flagged():
    model = make_model(
        grader=Grader(verifier_path="/grader/verify.sh", verifier_writable_by_agent=True)
    )
    findings = writable_check.run(model)
    assert len(findings) == 1
    assert findings[0].id == "GRAD-001"
    assert findings[0].severity == Severity.ERROR
    assert findings[0].confidence == Confidence.HIGH


def test_readonly_verifier_not_flagged():
    model = make_model(
        grader=Grader(verifier_path="/grader/verify.sh", verifier_writable_by_agent=False)
    )
    assert writable_check.run(model) == []


def test_explicit_accept_empty_output_flagged():
    model = make_model(grader=Grader(accepts_empty_output=True))
    findings = empty_check.run(model)
    assert len(findings) == 1
    assert findings[0].id == "GRAD-002"
    assert findings[0].severity == Severity.ERROR


def test_empty_submission_scored_pass_flagged():
    model = make_model(
        attempts=[Attempt(task_id="t1", status="pass", score=1.0, empty_submission=True)]
    )
    findings = empty_check.run(model)
    assert len(findings) == 1
    assert "t1" in findings[0].title


def test_empty_submission_scored_fail_not_flagged():
    # A failed empty submission means the grader did its job.
    model = make_model(
        attempts=[Attempt(task_id="t1", status="fail", score=0.0, empty_submission=True)]
    )
    assert empty_check.run(model) == []


def test_zero_tool_call_pass_not_flagged():
    # A legitimate one-shot pass (no tools needed) must not be flagged.
    model = make_model(
        attempts=[Attempt(task_id="t1", status="pass", score=1.0, tool_calls=0, empty_submission=False)]
    )
    assert empty_check.run(model) == []


def test_normal_grader_clean():
    model = make_model(grader=Grader(verifier_path="/grader/verify.sh"))
    assert writable_check.run(model) == []
    assert empty_check.run(model) == []
