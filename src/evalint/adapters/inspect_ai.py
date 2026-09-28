"""Inspect AI adapter (v0.1).

Reads an Inspect-style eval artifact directory -- the v0.1 normalized input
format, modeled on Inspect's Task / dataset / scorer / .eval-log concepts:

    eval-artifact/
      dataset.json       samples: [{id, prompt, metadata}]
      environment.json   env vars visible to the solver, mounts
      grader.json        verifier config, pass conditions, tests
      run.json           per-task attempts with status, usage, actions
      judge_run.json     (optional) model-judge config, judgments, reference labels

This is a read-only translation layer. Full-fidelity parsing of real Inspect
`.eval` logs is a later milestone; the adapter pins the schema version it
understands and fails clearly on anything else.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..model import (
    Attempt,
    Confidence,
    Environment,
    Grader,
    IntegrityModel,
    Judgment,
    Mount,
    TaskSample,
)
from . import AuditError, register

ADAPTER_NAME = "inspect"
ADAPTER_VERSION = "0.1.0"
SCHEMA_VERSION = "evalint-artifact-v1"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AuditError(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AuditError(f"invalid JSON in {path}: {exc}") from exc


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@register
class InspectAdapter:
    name = ADAPTER_NAME
    version = ADAPTER_VERSION

    def detect(self, path: Path) -> Confidence:
        if not path.is_dir():
            return Confidence.LOW
        has_dataset = (path / "dataset.json").is_file()
        has_grader = (path / "grader.json").is_file()
        if has_dataset and has_grader:
            return Confidence.HIGH
        if has_dataset:
            return Confidence.MEDIUM
        return Confidence.LOW

    def collect(self, path: Path) -> dict:
        """Read-only: files are opened for reading and never modified."""
        bundle: dict[str, Any] = {"root": path}
        digests: dict[str, str] = {}
        for fname in (
            "dataset.json",
            "environment.json",
            "grader.json",
            "run.json",
            "judge_run.json",
        ):
            fpath = path / fname
            if fpath.is_file():
                bundle[fname] = _read_json(fpath)
                digests[fname] = _digest(fpath)
        for fname in ("gold_map.json",):
            fpath = path / fname
            if fpath.is_file():
                # Recorded for the digest manifest only; never parsed for content
                # beyond its existence (it is the leak, not the evidence).
                digests[fname] = _digest(fpath)
                bundle.setdefault("extra_files", []).append(fname)
        bundle["digests"] = digests
        return bundle

    def normalize(self, bundle: dict) -> IntegrityModel:
        root: Path = bundle["root"]
        dataset = bundle.get("dataset.json", {})
        environment = bundle.get("environment.json", {})
        grader_cfg = bundle.get("grader.json", {})
        run = bundle.get("run.json", {})

        if dataset.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION and "tasks" not in dataset:
            raise AuditError(
                f"unsupported dataset schema (expected {SCHEMA_VERSION!r} with a 'tasks' list)"
            )

        tasks = [
            TaskSample(
                id=str(t.get("id", f"task-{i}")),
                prompt=str(t.get("prompt", "")),
                metadata=dict(t.get("metadata", {})),
            )
            for i, t in enumerate(dataset.get("tasks", []))
        ]

        env_vars = {
            str(name): "<redacted>"  # values are never stored; names are the signal
            for name in (environment.get("env") or {})
        }
        mounts = [
            Mount(
                path=str(m.get("path", "")),
                mode=str(m.get("mode", "ro")),
                agent_access=str(m.get("agent_access", "read")),
            )
            for m in (environment.get("mounts") or [])
        ]

        verifier = grader_cfg.get("verifier") or {}
        judge = grader_cfg.get("judge") or {}
        grader = Grader(
            kind=str(grader_cfg.get("kind", "script")),
            verifier_path=verifier.get("path"),
            verifier_writable_by_agent=bool(verifier.get("writable_by_agent", False)),
            accepts_empty_output=bool(grader_cfg.get("accepts_empty_output", False)),
            tests=[str(t) for t in (grader_cfg.get("tests") or [])],
            judge_model=judge.get("model"),
            judge_family=judge.get("family"),
            protocol=judge.get("protocol"),
            counterbalanced=judge.get("counterbalanced"),
            temperature=judge.get("temperature"),
            repeats=int(judge.get("repeats", 1)),
            rubric_criteria=[str(c) for c in (judge.get("rubric_criteria") or [])],
            scale_anchors={str(k): str(v) for k, v in (judge.get("scale_anchors") or {}).items()},
        )

        attempts = [
            Attempt(
                task_id=str(a.get("task_id", "")),
                status=str(a.get("status", "error")),
                score=a.get("score"),
                tool_calls=int(a.get("tool_calls", 0)),
                actions=[str(x) for x in (a.get("actions") or [])],
                tokens_in=a.get("tokens_in"),
                tokens_out=a.get("tokens_out"),
                latency_s=a.get("latency_s"),
                tries=int(a.get("tries", 1)),
                empty_submission=bool(a.get("empty_submission", False)),
            )
            for a in (run.get("attempts") or [])
        ]

        unsupported: list[str] = []

        judge_run = bundle.get("judge_run.json", {})
        judgments: list[Judgment] = []
        for j in judge_run.get("judgments", []):
            candidates = sorted(str(c) for c in (j.get("candidates") or []))
            judgments.append(
                Judgment(
                    task_id=str(j.get("task_id", "")),
                    candidates=candidates,
                    presentation_order=[str(c) for c in (j.get("presentation_order") or candidates)],
                    winner=j.get("winner"),
                    scores={str(k): float(v) for k, v in (j.get("scores") or {}).items()},
                    lengths={str(k): int(v) for k, v in (j.get("lengths") or {}).items()},
                    repeat_index=int(j.get("repeat_index", 0)),
                )
            )
        if any("rationale" in j for j in judge_run.get("judgments", [])):
            # Raw judge text is redacted at the boundary by design, not parsed.
            # Reported here so the redaction is explicit, not silent.
            unsupported.append("judge_run.json:judgments[].rationale (redacted: raw judge text not stored)")
        grader.reference_labels = {
            str(k): str(v) for k, v in (judge_run.get("reference_labels") or {}).items()
        }
        for fname, content in bundle.items():
            if fname in ("root", "digests", "extra_files"):
                continue
            if isinstance(content, dict):
                known = {
                    "dataset.json": {"schema_version", "eval_id", "tasks"},
                    "environment.json": {"env", "mounts", "notes"},
                    "grader.json": {"kind", "verifier", "accepts_empty_output", "tests", "judge"},
                    "run.json": {"solver", "attempts", "notes"},
                    "judge_run.json": {"judge", "judgments", "reference_labels", "notes"},
                }.get(fname, set())
                for key in content:
                    if key not in known:
                        unsupported.append(f"{fname}:{key}")

        return IntegrityModel(
            eval_id=str(dataset.get("eval_id", root.name)),
            adapter_name=self.name,
            adapter_version=self.version,
            tasks=tasks,
            environment=Environment(env_vars=env_vars, mounts=mounts),
            grader=grader,
            attempts=attempts,
            judgments=judgments,
            unsupported=unsupported,
            digests=dict(bundle.get("digests", {})),
        )
