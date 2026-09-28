"""COST-001 tests."""
from __future__ import annotations

from evalint.checks.cost import CostReportingCheck, summarize_cost
from evalint.model import Attempt, Confidence, Severity

from .conftest import make_model


def _priced_attempt(task_id: str, status: str = "pass") -> Attempt:
    return Attempt(
        task_id=task_id,
        status=status,
        score=1.0 if status == "pass" else 0.0,
        tool_calls=4,
        tokens_in=1_000_000,
        tokens_out=1_000_000,
    )


def test_cost_math():
    attempts = [_priced_attempt("t1"), _priced_attempt("t2")]
    summary = summarize_cost(attempts, price_in_per_1m=3.0, price_out_per_1m=15.0)
    assert summary is not None
    assert summary.total_tokens_in == 2_000_000
    assert summary.total_tokens_out == 2_000_000
    assert summary.estimated_usd == 36.0
    assert summary.cost_per_success_usd == 18.0
    assert summary.avg_tool_calls_per_success == 4.0


def test_cost_per_success_with_failures():
    attempts = [_priced_attempt("t1"), _priced_attempt("t2", status="fail")]
    summary = summarize_cost(attempts)
    assert summary is not None
    assert summary.successes == 1
    assert summary.cost_per_success_usd == 36.0


def test_missing_token_data_flagged():
    model = make_model(attempts=[Attempt(task_id="t1", status="pass", score=1.0)])
    findings = CostReportingCheck().run(model)
    assert len(findings) == 1
    assert findings[0].id == "COST-001"
    assert findings[0].severity == Severity.MEDIUM
    assert findings[0].confidence == Confidence.HIGH


def test_present_token_data_no_finding_and_summary_attached():
    model = make_model(attempts=[_priced_attempt("t1")])
    findings = CostReportingCheck().run(model)
    assert findings == []
    assert model.cost_summary is not None
    assert model.cost_summary.cost_per_success_usd == 18.0


def test_no_attempts_no_finding():
    model = make_model(attempts=[])
    assert CostReportingCheck().run(model) == []
