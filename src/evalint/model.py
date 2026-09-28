"""Normalized integrity model: the framework-neutral description of an evaluation.

Adapters translate harness-specific artifacts into this model. Checks only ever
see this model, never harness internals. That boundary is what keeps this tool
a linter instead of another eval framework.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    """How much a finding undermines the reported score."""

    ERROR = "error"  # Direct evidence the score can be invalid.
    HIGH = "high"  # A concrete exploit path exists.
    MEDIUM = "medium"  # Weakens trust; needs context to judge.
    LOW = "low"  # Hygiene note.


class Confidence(str, Enum):
    """Every finding carries a confidence label. Precision over recall."""

    HIGH = "high"  # Deterministic, directly observed evidence.
    MEDIUM = "medium"  # Strong heuristic signal; corroborate before blocking.
    LOW = "low"  # Weak/indirect signal; advisory only.


SEVERITY_ORDER = [Severity.ERROR, Severity.HIGH, Severity.MEDIUM, Severity.LOW]


@dataclass
class SourceLocation:
    file: str
    line: int | None = None
    excerpt: str | None = None

    def render(self) -> str:
        base = self.file
        if self.line is not None:
            base += f":{self.line}"
        if self.excerpt:
            base += f" -- {self.excerpt}"
        return base


@dataclass
class Finding:
    id: str  # Stable rule ID, e.g. "ENV-001".
    title: str
    severity: Severity
    confidence: Confidence
    description: str  # Why it matters.
    evidence: list[str] = field(default_factory=list)
    locations: list[SourceLocation] = field(default_factory=list)
    remediation: str = ""
    fingerprint: str = ""  # Stable hash for baselining across runs.

    def __post_init__(self) -> None:
        if not self.fingerprint:
            key = "|".join(
                [
                    self.id,
                    self.title,
                    ",".join(sorted(loc.file for loc in self.locations)),
                ]
            )
            self.fingerprint = hashlib.sha256(key.encode()).hexdigest()[:16]


@dataclass
class Mount:
    path: str
    mode: str = "ro"  # "ro" | "rw"
    agent_access: str = "read"  # "none" | "read" | "write"


@dataclass
class Environment:
    # Names of env vars visible to the agent/solver. Values are NEVER stored:
    # secret values must not end up in reports.
    env_vars: dict[str, str] = field(default_factory=dict)  # name -> "<redacted>"
    mounts: list[Mount] = field(default_factory=list)


@dataclass
class TaskSample:
    id: str
    prompt: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Grader:
    kind: str = "script"
    verifier_path: str | None = None
    verifier_writable_by_agent: bool = False
    accepts_empty_output: bool = False
    tests: list[str] = field(default_factory=list)


@dataclass
class Attempt:
    task_id: str
    status: str  # "pass" | "fail" | "error" | "incomplete"
    score: float | None = None
    tool_calls: int = 0
    actions: list[str] = field(default_factory=list)
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_s: float | None = None
    tries: int = 1
    empty_submission: bool = False


@dataclass
class CostSummary:
    attempts: int
    successes: int
    total_tokens_in: int
    total_tokens_out: int
    estimated_usd: float
    cost_per_success_usd: float | None
    avg_tool_calls_per_success: float | None
    price_in_per_1m: float
    price_out_per_1m: float


@dataclass
class IntegrityModel:
    eval_id: str
    adapter_name: str
    adapter_version: str
    tasks: list[TaskSample] = field(default_factory=list)
    environment: Environment = field(default_factory=Environment)
    grader: Grader = field(default_factory=Grader)
    attempts: list[Attempt] = field(default_factory=list)
    # Harness fields the adapter saw but could not translate. Shown in the
    # report as coverage gaps, never silently dropped.
    unsupported: list[str] = field(default_factory=list)
    # Artifact file -> sha256, for the reproducibility manifest.
    digests: dict[str, str] = field(default_factory=dict)
    cost_summary: CostSummary | None = None
