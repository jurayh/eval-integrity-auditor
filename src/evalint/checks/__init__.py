"""Rule registry. v0.1 ships deterministic, high-precision checks only.

Sequencing rule: earn trust with deterministic evidence before adding
probabilistic signals. Every finding carries a confidence label; a linter that
cries contamination on a clean eval is worse than no auditor.
"""
from __future__ import annotations

from .base import Check
from .cost import CostReportingCheck
from .env_leakage import EnvLeakageCheck
from .grader import EmptyPathCheck, VerifierWritableCheck

REGISTRY: list[Check] = [
    EnvLeakageCheck(),  # ENV-001
    VerifierWritableCheck(),  # GRAD-001
    EmptyPathCheck(),  # GRAD-002
    CostReportingCheck(),  # COST-001
]

BY_ID: dict[str, Check] = {check.meta.id: check for check in REGISTRY}


def get_check(check_id: str) -> Check | None:
    return BY_ID.get(check_id.upper())
