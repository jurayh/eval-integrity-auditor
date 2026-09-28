"""Engine tests: scoring, verdicts, and end-to-end precision on the fixtures."""
from __future__ import annotations

import pytest

from evalint.adapters import AuditError
from evalint.engine import audit, integrity_score
from evalint.model import Confidence, Finding, Severity

from .conftest import DEMO_HARDENED, DEMO_LEAKY


def _finding(severity: Severity) -> Finding:
    return Finding(
        id="TEST-001",
        title="t",
        severity=severity,
        confidence=Confidence.HIGH,
        description="d",
    )


def test_score_deductions():
    assert integrity_score([]) == 100
    assert integrity_score([_finding(Severity.ERROR)]) == 75
    assert integrity_score([_finding(Severity.HIGH)]) == 90
    assert integrity_score([_finding(Severity.MEDIUM)]) == 95
    assert integrity_score([_finding(Severity.LOW)]) == 99
    assert integrity_score([_finding(Severity.ERROR)] * 10) == 0  # floored


def test_leaky_fixture_blocked():
    result = audit(DEMO_LEAKY)
    assert result.verdict == "BLOCKED"
    ids = {f.id for f in result.findings}
    assert "ENV-001" in ids
    assert "GRAD-001" in ids
    # No false COST-001: the fixture records token usage.
    assert "COST-001" not in ids
    assert result.score == 0  # 5 errors x 25


def test_hardened_fixture_passes_clean():
    result = audit(DEMO_HARDENED)
    assert result.verdict == "PASS"
    assert result.findings == []
    assert result.score == 100


def test_audit_missing_path_raises():
    with pytest.raises(AuditError):
        audit("/does/not/exist")
