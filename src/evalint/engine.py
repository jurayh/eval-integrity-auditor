"""Audit engine: adapter -> checks -> score -> verdict.

The pipeline is deliberately boring: detect the harness, translate to the
normalized model, run every registered check, score the findings. No network,
no mutation, no surprises.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import adapters  # noqa: F401  (import registers the Inspect adapter)
from .adapters import AuditError, autodetect
from .adapters import inspect_ai  # noqa: F401  (registers itself on import)
from .checks import REGISTRY
from .checks.cost import CostReportingCheck
from .model import SEVERITY_ORDER, Finding, IntegrityModel, Severity


@dataclass
class AuditResult:
    model: IntegrityModel
    findings: list[Finding]
    score: int  # 0-100, diagnostic only: an explainable prioritization aid.
    verdict: str  # "BLOCKED" | "PASS"
    blocked_by: list[Finding]


def integrity_score(findings: list[Finding]) -> int:
    """Explainable, fixed deduction per severity. Diagnostic, not a certification."""
    deductions = {Severity.ERROR: 25, Severity.HIGH: 10, Severity.MEDIUM: 5, Severity.LOW: 1}
    score = 100 - sum(deductions.get(f.severity, 0) for f in findings)
    return max(0, score)


def sort_findings(findings: list[Finding]) -> list[Finding]:
    order = {sev: i for i, sev in enumerate(SEVERITY_ORDER)}
    return sorted(findings, key=lambda f: (order[f.severity], f.id, f.title))


_FAIL_ON_LEVELS = {
    "error": {Severity.ERROR},
    "high": {Severity.ERROR, Severity.HIGH},
    "medium": {Severity.ERROR, Severity.HIGH, Severity.MEDIUM},
    "low": {Severity.ERROR, Severity.HIGH, Severity.MEDIUM, Severity.LOW},
}


def audit(
    path: str | Path,
    adapter_name: str = "auto",
    price_in_per_1m: float = 3.0,
    price_out_per_1m: float = 15.0,
) -> AuditResult:
    target = Path(path)
    if not target.exists():
        raise AuditError(f"no such eval artifact: {target}")

    if adapter_name == "auto":
        adapter = autodetect(target)
    else:
        matches = [a for a in adapters.REGISTRY if a.name == adapter_name]
        if not matches:
            raise AuditError(f"unknown adapter {adapter_name!r}")
        adapter = matches[0]

    bundle = adapter.collect(target)
    model = adapter.normalize(bundle)

    checks = [c for c in REGISTRY if not isinstance(c, CostReportingCheck)]
    checks.append(CostReportingCheck(price_in_per_1m, price_out_per_1m))
    findings: list[Finding] = []
    for check in checks:
        findings.extend(check.run(model))
    findings = sort_findings(findings)

    score = integrity_score(findings)
    blocked_by = [f for f in findings if f.severity in _FAIL_ON_LEVELS["high"]]
    verdict = "BLOCKED" if blocked_by else "PASS"
    return AuditResult(
        model=model, findings=findings, score=score, verdict=verdict, blocked_by=blocked_by
    )


def audit_with_policy(
    path: str | Path,
    adapter_name: str = "auto",
    fail_on: str = "high",
    price_in_per_1m: float = 3.0,
    price_out_per_1m: float = 15.0,
) -> tuple[AuditResult, bool]:
    """Run the audit; return (result, policy_failed)."""
    result = audit(path, adapter_name, price_in_per_1m, price_out_per_1m)
    levels = _FAIL_ON_LEVELS[fail_on]
    policy_failed = any(f.severity in levels for f in result.findings)
    return result, policy_failed
