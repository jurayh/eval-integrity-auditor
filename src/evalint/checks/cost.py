"""COST-001: cost per success.

"Agent cost per completed task is the only unit a finance review can act on."
This check does two things:

1. If run records carry token/latency/tool-call data, it computes a cost
   summary (total, per-success, tool calls per success) and attaches it to the
   model for the report. No finding -- the data is present.
2. If attempts exist but none carry token data, it emits a Medium finding:
   cost per success cannot be computed because usage was not recorded.

Prices are estimates, configurable, and always labeled as such in the report.
"""
from __future__ import annotations

from ..model import Attempt, Confidence, CostSummary, Finding, IntegrityModel, Severity, SourceLocation
from .base import Check, CheckMeta


def summarize_cost(
    attempts: list[Attempt],
    price_in_per_1m: float = 3.0,
    price_out_per_1m: float = 15.0,
) -> CostSummary | None:
    priced = [a for a in attempts if a.tokens_in is not None and a.tokens_out is not None]
    if not priced:
        return None
    total_in = sum(a.tokens_in or 0 for a in priced)
    total_out = sum(a.tokens_out or 0 for a in priced)
    estimated = total_in / 1_000_000 * price_in_per_1m + total_out / 1_000_000 * price_out_per_1m
    successes = [a for a in priced if a.status == "pass"]
    tool_calls = [a.tool_calls for a in successes]
    return CostSummary(
        attempts=len(priced),
        successes=len(successes),
        total_tokens_in=total_in,
        total_tokens_out=total_out,
        estimated_usd=round(estimated, 6),
        cost_per_success_usd=round(estimated / len(successes), 6) if successes else None,
        avg_tool_calls_per_success=round(sum(tool_calls) / len(tool_calls), 2) if tool_calls else None,
        price_in_per_1m=price_in_per_1m,
        price_out_per_1m=price_out_per_1m,
    )


class CostReportingCheck(Check):
    meta = CheckMeta(
        id="COST-001",
        title="Cost per success not reported",
        threat=(
            "A pass rate without cost hides the real trade-off: an agent that "
            "succeeds after 50 retries at 15x token burn is not the same system "
            "as one that succeeds first try. Without usage data, cost per "
            "completed task -- the unit a budget review can act on -- is unknown."
        ),
        remediation=(
            "Record tokens in/out, latency, tool calls, and retry counts per "
            "attempt in the run log, and report cost per successful task "
            "alongside the pass rate."
        ),
    )

    def __init__(self, price_in_per_1m: float = 3.0, price_out_per_1m: float = 15.0):
        self.price_in_per_1m = price_in_per_1m
        self.price_out_per_1m = price_out_per_1m

    def run(self, model: IntegrityModel) -> list[Finding]:
        summary = summarize_cost(model.attempts, self.price_in_per_1m, self.price_out_per_1m)
        if summary is not None:
            model.cost_summary = summary
            return []
        if not model.attempts:
            # No runs at all: nothing to cost. The report shows this as a
            # coverage gap, not a finding.
            return []
        return [
            Finding(
                id=self.meta.id,
                title="Token usage not recorded: cost per success unknown",
                severity=Severity.MEDIUM,
                confidence=Confidence.HIGH,
                description=self.meta.threat,
                evidence=[
                    f"{len(model.attempts)} attempt(s) recorded, none with tokens_in/tokens_out.",
                    "Cost per completed task cannot be computed from this run log.",
                ],
                locations=[SourceLocation(file="run.json", excerpt="attempts[] tokens_in/tokens_out")],
                remediation=self.meta.remediation,
            )
        ]
