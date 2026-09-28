"""Grader checks.

GRAD-001: the verifier/oracle is writable by the agent. If the agent can edit
the thing that decides pass/fail, the score is whatever the agent wants.

GRAD-002: the grader grants credit without completion. Two deterministic
signals: the grader is explicitly configured to accept empty output, or a run
record shows an explicitly empty submission scored as a pass.
"""
from __future__ import annotations

from ..model import Confidence, Finding, IntegrityModel, Severity, SourceLocation
from .base import Check, CheckMeta


class VerifierWritableCheck(Check):
    meta = CheckMeta(
        id="GRAD-001",
        title="Verifier writable by the agent",
        threat=(
            "If the agent can write to the verifier, oracle, or test files, it can "
            "change the measurement itself: weaken assertions, plant expected "
            "outputs, or mark itself passed. The score then reflects the agent's "
            "access, not its capability."
        ),
        remediation=(
            "Mount the verifier and all test/oracle files read-only for the agent "
            "(writable only by the harness), and verify ownership and permissions "
            "in CI."
        ),
    )

    def run(self, model: IntegrityModel) -> list[Finding]:
        grader = model.grader
        if not grader.verifier_writable_by_agent:
            return []
        path = grader.verifier_path or "<unknown verifier path>"
        return [
            Finding(
                id=self.meta.id,
                title=f"Verifier writable by the agent: {path}",
                severity=Severity.ERROR,
                confidence=Confidence.HIGH,
                description=self.meta.threat,
                evidence=[
                    f"grader declares verifier {path!r} writable by the agent.",
                    "An agent with write access to the oracle can alter pass/fail "
                    "outcomes directly.",
                ],
                locations=[SourceLocation(file="grader.json", excerpt="verifier.writable_by_agent")],
                remediation=self.meta.remediation,
            )
        ]


class EmptyPathCheck(Check):
    meta = CheckMeta(
        id="GRAD-002",
        title="Grader grants credit without completion",
        threat=(
            "A grader that passes empty, errored, or trivial submissions inflates "
            "scores: agents get credit for doing nothing. Common causes are "
            "misconfigured scorers, tests that assert nothing, or pipelines that "
            "treat missing output as success."
        ),
        remediation=(
            "Require non-empty, task-relevant output before scoring; add a control "
            "case that submits nothing and assert it fails; fix the scorer "
            "configuration."
        ),
    )

    def run(self, model: IntegrityModel) -> list[Finding]:
        findings: list[Finding] = []
        if model.grader.accepts_empty_output:
            findings.append(
                Finding(
                    id=self.meta.id,
                    title="Grader is configured to accept empty output",
                    severity=Severity.ERROR,
                    confidence=Confidence.HIGH,
                    description=self.meta.threat,
                    evidence=[
                        "grader.json sets accepts_empty_output=true.",
                        "Any submission -- including no submission at all -- can score as a pass.",
                    ],
                    locations=[SourceLocation(file="grader.json", excerpt="accepts_empty_output")],
                    remediation=self.meta.remediation,
                )
            )
        for attempt in model.attempts:
            # Only flag when the run record explicitly marks the submission as
            # empty AND the grader passed it. A zero-tool-call pass on a real
            # answer is legitimate and must not be flagged.
            if attempt.empty_submission and attempt.status == "pass":
                findings.append(
                    Finding(
                        id=self.meta.id,
                        title=f"Empty submission scored as pass: {attempt.task_id}",
                        severity=Severity.ERROR,
                        confidence=Confidence.HIGH,
                        description=self.meta.threat,
                        evidence=[
                            f"task {attempt.task_id!r}: submission marked empty, "
                            f"status={attempt.status!r}, score={attempt.score}.",
                            "Credit was granted without any completed work.",
                        ],
                        locations=[SourceLocation(file="run.json", excerpt=f"attempts[] task_id={attempt.task_id}")],
                        remediation=self.meta.remediation,
                    )
                )
        return findings
