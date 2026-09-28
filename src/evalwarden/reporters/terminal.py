"""Terminal reporter: the linter-style output."""
from __future__ import annotations

from ..engine import AuditResult
from ..model import Severity

_SEV_LABEL = {
    Severity.ERROR: "E",
    Severity.HIGH: "H",
    Severity.MEDIUM: "M",
    Severity.LOW: "L",
}


def render_terminal(result: AuditResult) -> str:
    lines: list[str] = []
    model = result.model
    status = "BLOCKED" if result.verdict == "BLOCKED" else "PASS"
    counts = {sev: 0 for sev in Severity}
    for f in result.findings:
        counts[f.severity] += 1

    lines.append(f"evalwarden audit: {model.eval_id}")
    lines.append(f"adapter: {model.adapter_name} {model.adapter_version}")
    lines.append(
        f"Integrity: {result.score} / 100 {status} "
        f"({counts[Severity.ERROR]} errors, {counts[Severity.HIGH]} high, "
        f"{counts[Severity.MEDIUM]} medium, {counts[Severity.LOW]} low)"
    )
    if result.verdict == "BLOCKED":
        blocked_ids = sorted({f.id for f in result.blocked_by})
        lines.append(f"claim blocked by: {', '.join(blocked_ids)}")
    lines.append("")

    if not result.findings:
        lines.append("no blocking findings observed under this policy.")
    for finding in result.findings:
        label = _SEV_LABEL[finding.severity]
        lines.append(
            f"{label} {finding.id} [{finding.severity.value}|confidence:{finding.confidence.value}] "
            f"{finding.title}"
        )
        for ev in finding.evidence:
            lines.append(f"    evidence: {ev}")
        for loc in finding.locations:
            lines.append(f"    at: {loc.render()}")
        if finding.remediation:
            lines.append(f"    fix: {finding.remediation}")
        lines.append("")

    if model.cost_summary:
        c = model.cost_summary
        lines.append("cost summary (estimates, not metered billing):")
        lines.append(
            f"    attempts={c.attempts} successes={c.successes} "
            f"tokens_in={c.total_tokens_in} tokens_out={c.total_tokens_out}"
        )
        lines.append(f"    estimated total: ${c.estimated_usd:.4f}")
        if c.cost_per_success_usd is not None:
            lines.append(f"    cost per success: ${c.cost_per_success_usd:.4f}")
        if c.avg_tool_calls_per_success is not None:
            lines.append(f"    avg tool calls per success: {c.avg_tool_calls_per_success}")
        if c.avg_tries_per_success is not None:
            lines.append(
                f"    avg tries per success: {c.avg_tries_per_success} "
                f"(max {c.max_tries} on one attempt)"
            )
        lines.append(
            f"    wasted on non-passing attempts: ${c.wasted_usd:.4f} "
            f"({c.wasted_share:.0%} of spend)"
        )
        lines.append(
            f"    output tokens per attempt: p50={c.p50_tokens_out:,} "
            f"p90={c.p90_tokens_out:,} max={c.max_tokens_out:,} ({c.max_tokens_out_task})"
        )
        if c.budget_per_task_usd is not None:
            lines.append(
                f"    within ${c.budget_per_task_usd:.2f}/task budget: "
                f"{c.tasks_within_budget}/{c.tasks_total} tasks passed"
            )
        lines.append("")

    if model.unsupported:
        lines.append("coverage gaps (seen but not translated):")
        for item in model.unsupported:
            lines.append(f"    - {item}")
        lines.append("")

    lines.append("Integrity score is diagnostic, not a certification.")
    return "\n".join(lines).rstrip() + "\n"
