#!/usr/bin/env python3
"""Build evalwarden audit artifacts from real public benchmark definitions.

This script translates PUBLIC, downloadable eval definitions into evalwarden's
artifact format (dataset.json / environment.json / grader.json) WITHOUT
inventing anything:

- Only facts present in the public sources are encoded. Anything the
  definition does not say is left out, and the omission is documented in
  PROVENANCE.json next to each artifact.
- No traces are fabricated: there is no run.json. Checks that need run
  records (COST-002..004, GRAD-002's attempt leg, JUDGE-003..006) simply do
  not fire, and COST-001 treats "no runs at all" as a coverage gap, not a
  finding.
- Answer-bearing material (gold patches, test patches, full rubrics) is
  EXCLUDED from the translated artifact even though it is public: the
  integrity question is whether the AGENT can see it, and the task
  definitions keep it harness-side. The exclusion is documented.

Pinned sources (all public, no auth):
- inspect_evals @ 244e43cc924d1de5a78dad7db259bbd4471c97e7
  (https://github.com/UKGovernmentBEIS/inspect_evals)
- SWE-bench_Verified: princeton-nlp/SWE-bench_Verified
  rev c104f840cc67f8b6eec6f759ebc8b2693d585d4a (pinned in the task source)
- HealthBench: https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl
  (URL pinned in the task source)
- WritingBench: benchmark_all.jsonl inside the inspect_evals repo at the pinned
  commit (fetched via raw.githubusercontent.com at that commit)
- MMLU: cais/mmlu (config "all", test split) rev
  c30699e8356da336a370243923dbaf21066bb9fe (revision pinned in the task
  source); prompt template MultipleChoiceTemplate.SINGLE_ANSWER from
  UKGovernmentBEIS/inspect_ai src/inspect_ai/solver/_multiple_choice.py

Usage:
    python3 build_real_cards.py --work-dir /tmp/real_work
    # then:
    evalwarden report-cards /tmp/real_work/swe-bench-verified /tmp/real_work/healthbench \
        --output-dir examples/report-cards/real/

Re-running the script reproduces the artifacts byte-for-byte as long as the
pinned sources are unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import urllib.request
from pathlib import Path

SCHEMA_VERSION = "evalwarden-artifact-v1"
N_SAMPLES = 12

INSPECT_EVALS_COMMIT = "244e43cc924d1de5a78dad7db259bbd4471c97e7"
SWE_BENCH_DATASET = "princeton-nlp/SWE-bench_Verified"
SWE_BENCH_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
HEALTHBENCH_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/"
    "2025-05-07-06-14-12_oss_eval.jsonl"
)
# Exact input prompt from inspect_evals/swe_bench/swe_bench.py
SWE_INPUT_PROMPT = "Please solve the following coding issue:\n\n{issue_text}"
# Docker image template from inspect_evals/swe_bench/swe_bench.py
SWE_IMAGE_TEMPLATE = "ghcr.io/epoch-research/swe-bench.eval.{arch}.{id}:latest"


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "evalwarden-card-builder/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_jsonl_head(url: str, want_lines: int, chunk_bytes: int = 2_000_000) -> list[str]:
    """Fetch the head of a JSONL file in small ranged chunks until enough
    complete lines are collected. Returns the complete lines."""
    buf = b""
    offset = 0
    for _ in range(12):  # cap: 12 chunks x 2MB
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "evalwarden-card-builder/1.0",
                "Range": f"bytes={offset}-{offset + chunk_bytes - 1}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = resp.read()
        except Exception as exc:  # noqa: BLE001 - network flakiness, not fatal yet
            raise RuntimeError(f"download failed at offset {offset}: {exc}") from exc
        if not data:
            break
        buf += data
        offset += len(data)
        text = buf.decode("utf-8", errors="replace")
        complete = [ln for ln in text.splitlines() if ln.strip().endswith("}")]
        if len(complete) >= want_lines:
            return complete
        if len(data) < chunk_bytes:
            break  # EOF
    text = buf.decode("utf-8", errors="replace")
    return [ln for ln in text.splitlines() if ln.strip().endswith("}")]


def build_swe_bench(work: Path) -> Path:
    """Translate the inspect_evals swe_bench (SWE-bench Verified) task definition."""
    out = work / "swe-bench-verified"
    out.mkdir(parents=True, exist_ok=True)

    data = _get_json(
        f"https://datasets-server.huggingface.co/rows?dataset={SWE_BENCH_DATASET}"
        f"&config=default&split=test&offset=0&length={N_SAMPLES}"
    )
    rows = [r["row"] for r in data["rows"]]
    assert len(rows) == N_SAMPLES, f"expected {N_SAMPLES} rows, got {len(rows)}"

    tasks = []
    for r in rows:
        tasks.append(
            {
                "id": r["instance_id"],
                "prompt": SWE_INPUT_PROMPT.format(issue_text=r["problem_statement"]),
                "metadata": {
                    "repo": r["repo"],
                    "version": r["version"],
                    "base_commit": r["base_commit"],
                    # Template only: the task resolves {id}/{arch} per sample at
                    # definition time (swe_bench.py resolve_image_name).
                    "image_name_template": SWE_IMAGE_TEMPLATE,
                    "sandbox": "docker",
                },
            }
        )

    (out / "dataset.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "eval_id": "SWE-bench Verified via inspect_evals",
                "tasks": tasks,
                "notes": (
                    f"Definition sample: {N_SAMPLES} of 500 instances from "
                    f"{SWE_BENCH_DATASET} rev {SWE_BENCH_REVISION[:12]}. "
                    "Gold patch, test patch, and FAIL_TO_PASS/PASS_TO_PASS "
                    "contents are public in the dataset but EXCLUDED here: "
                    "they live in harness-side sample metadata (TaskState), "
                    "which the react agent's sandbox tools cannot read. "
                    "hints_text likewise stays metadata-side and is not in "
                    "the agent prompt."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "environment.json").write_text(
        json.dumps(
            {
                "env": {},
                "mounts": [],
                "notes": (
                    "The task definition (swe_bench.py, solvers.py) declares "
                    "no agent-visible environment variables and no mounts. "
                    "The agent (swe_bench_agent_with_inspect_tool_support: "
                    "bash/python/text_editor) runs inside a Docker container "
                    "holding the repo at base_commit. The container image "
                    "name embeds the instance_id, but the definition "
                    "provides no mechanism exposing it inside the container "
                    "(no docker socket, no env passthrough). An oracle "
                    "solver (swe_bench_oracle_solver) ships in solvers.py for "
                    "pipeline validation only; it runs harness-side and "
                    "applies the gold patch from TaskState metadata, which "
                    "the evaluated agent cannot reach."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "grader.json").write_text(
        json.dumps(
            {
                "kind": "script",
                "verifier": {
                    "path": (
                        "swe_bench_scorer: writes eval script into the sandbox "
                        "and runs FAIL_TO_PASS/PASS_TO_PASS via the upstream "
                        "swebench harness (get_eval_report)"
                    ),
                    "writable_by_agent": False,
                },
                "tests": ["FAIL_TO_PASS", "PASS_TO_PASS"],
                "notes": (
                    "Scoring is delegated to the upstream swebench harness. "
                    "The test patch is applied by the SCORER after the agent "
                    "submits; the agent never sees the graded tests and has "
                    "no write path to the scoring. Gradeability requires "
                    "both test-output markers (the harness ignores the eval "
                    "script's exit code by design)."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "benchmark": "SWE-bench Verified",
                "task_definition": "inspect_evals/swe_bench @ " + INSPECT_EVALS_COMMIT,
                "task_files_read": ["swe_bench.py", "solvers.py", "scorers.py"],
                "dataset": SWE_BENCH_DATASET,
                "dataset_revision": SWE_BENCH_REVISION,
                "dataset_split": "test",
                "sample": f"first {N_SAMPLES} rows via HuggingFace datasets-server",
                "translation": "mechanical: prompt = DEFAULT_INPUT_PROMPT, metadata = public non-answer fields",
                "excluded": [
                    "patch (gold patch): public but harness-side; agent cannot read it",
                    "test_patch: applied by scorer post-submission; agent never sees it",
                    "FAIL_TO_PASS / PASS_TO_PASS test bodies: same as test_patch",
                    "hints_text: metadata-side, not in the agent prompt",
                ],
                "no_traces": True,
                "generated_by": "examples/report-cards/real/build_real_cards.py",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"swe-bench-verified: {len(tasks)} tasks -> {out}")
    return out


def build_healthbench(work: Path) -> Path:
    """Translate the inspect_evals healthbench task definition (default args)."""
    out = work / "healthbench"
    out.mkdir(parents=True, exist_ok=True)

    # Pull the head of the JSONL in small chunks until N_SAMPLES complete
    # examples are collected (examples vary widely in size).
    lines = _get_jsonl_head(HEALTHBENCH_URL, N_SAMPLES)
    examples = [json.loads(ln) for ln in lines[:N_SAMPLES]]
    assert len(examples) == N_SAMPLES, f"expected {N_SAMPLES} examples, got {len(examples)}"

    tasks = []
    criterion_texts: list[str] = []
    for ex in examples:
        user_msg = next(m["content"] for m in ex["prompt"] if m["role"] == "user")
        rubrics = ex["rubrics"]
        tasks.append(
            {
                "id": ex["prompt_id"],
                "prompt": user_msg,
                "metadata": {
                    "n_rubric_items": len(rubrics),
                    "total_positive_points": sum(r["points"] for r in rubrics if r["points"] > 0),
                    "total_negative_points": sum(r["points"] for r in rubrics if r["points"] < 0),
                    "axes": sorted({t.split(":", 1)[1] for r in rubrics for t in r["tags"] if t.startswith("axis:")}),
                    "example_tags": ex.get("example_tags", []),
                },
            }
        )
        for r in rubrics:
            if len(criterion_texts) < 4:
                criterion_texts.append(r["criterion"])

    (out / "dataset.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "eval_id": "HealthBench via inspect_evals",
                "tasks": tasks,
                "notes": (
                    f"Definition sample: {N_SAMPLES} of 5000 examples from the "
                    "HealthBench full split (URL pinned in the task source). "
                    "The solver sees only the conversation. Rubric criterion "
                    "texts are grader-side and summarized here as counts/axes; "
                    "four verbatim criteria are carried in grader.json as the "
                    "rubric_criteria sample."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "environment.json").write_text(
        json.dumps(
            {
                "env": {},
                "mounts": [],
                "notes": (
                    "The solver is plain `generate` (no tools, no sandbox). "
                    "The task definition (healthbench.py) declares no "
                    "agent-visible environment. Rubrics are grader-side; the "
                    "solver never sees them."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "grader.json").write_text(
        json.dumps(
            {
                "kind": "judge",
                "judge": {
                    # Defaults from the healthbench() task signature.
                    "model": "openai/gpt-4o-mini",
                    "protocol": "pointwise",
                    "temperature": 0.0,
                    "repeats": 1,
                    "rubric_criteria": criterion_texts,
                    # No scale_anchors: the rubric is a boolean checklist
                    # (criteria_met per item x points), not a scalar scale.
                    "scale_anchors": {},
                },
                "notes": (
                    "Judge defaults from healthbench() in healthbench.py. "
                    "Each (conversation, rubric item) pair is graded "
                    "criteria_met=true/false by the judge model per "
                    "GRADER_TEMPLATE (scorer.py); the score is the "
                    "points-weighted sum. The task package also ships a 29,511-item "
                    "physician-graded meta_eval subset (DATASET_URLS['meta_eval'] "
                    "in dataset.py, with a judge-vs-physician macro-F1 agreement "
                    "scorer in meta_evaluation.py); this artifact translates the "
                    "main eval subset only and does not encode those labels "
                    "(audited separately as healthbench-meta-eval-via-inspect-evals). "
                    "The rubric is a boolean checklist with point "
                    "weights, not a scalar scale with anchored levels."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "benchmark": "HealthBench (OpenAI, 2025)",
                "task_definition": "inspect_evals/healthbench @ " + INSPECT_EVALS_COMMIT,
                "task_files_read": ["healthbench.py", "scorer.py", "dataset.py", "eval.yaml"],
                "dataset_url": HEALTHBENCH_URL,
                "dataset_split": "full (5000 examples per eval.yaml)",
                "sample": f"first {N_SAMPLES} JSONL records",
                "judge_defaults_source": "healthbench() task signature (judge_model, judge_temperature)",
                "grader_template": "GRADER_TEMPLATE in scorer.py",
                "judge_validation_note": (
                    "The task package ships the healthbench_meta_eval subset "
                    "(29,511 physician-graded items, URL pinned in dataset.py, "
                    "judge-vs-physician agreement scorer in meta_evaluation.py); "
                    "this artifact translates the main eval subset only, which "
                    "does not encode those labels. The meta_eval subset is "
                    "audited as its own card (healthbench-meta-eval-via-inspect-evals)."
                ),
                "no_traces": True,
                "generated_by": "examples/report-cards/real/build_real_cards.py",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"healthbench: {len(tasks)} tasks -> {out}")
    return out


# Pinned in DATASET_URLS["meta_eval"] in inspect_evals/healthbench/dataset.py
# at the pinned commit (read by a human; the machine link is this file's hash).
HB_META_EVAL_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/"
    "2025-05-07-06-14-12_oss_meta_eval.jsonl"
)
# Exact judge defaults from meta_evaluation_scorer() in
# inspect_evals/healthbench/meta_evaluation.py at the pinned commit.
HB_META_JUDGE_MODEL = "openai/gpt-4o-mini"
HB_META_JUDGE_TEMPERATURE = 0.0


def _healthbench_meta_id(prompt_id: str, completion_id: str) -> str:
    """Replicate create_stable_id(prompt_id, completion_id, prefix="healthbench_meta")
    from inspect_evals/utils/deps_utils.py at the pinned commit."""
    combined = "\0".join([prompt_id, completion_id])
    return "healthbench_meta_" + hashlib.md5(combined.encode()).hexdigest()[:8]


def _meta_eval_judge_input(record: dict) -> str:
    """Replicate _create_conversation_string(state, completion) from
    inspect_evals/healthbench/scorer.py: the conversation turns plus the
    pre-recorded completion as the final assistant turn. This is exactly what
    the meta_evaluation_scorer feeds the judge model."""
    turns = list(record["prompt"]) + [{"role": "assistant", "content": record["completion"]}]
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in turns)


def _physician_majority(grades: list[bool]) -> bool:
    """Replicate calculate_physician_majority() in meta_evaluation.py."""
    return sum(grades) > len(grades) / 2


def build_healthbench_meta_eval(work: Path) -> Path:
    """Translate the inspect_evals healthbench meta_eval subset (default args).

    This subset is the benchmark's own judge-validation instrument: 29,511
    records, each carrying 2-5 physician boolean grades (binary_labels) for a
    pre-recorded assistant completion. meta_evaluation_scorer grades the judge
    model against the physician majority and reports macro F1. The physician
    labels are the ground truth the judge is validated against, so they are
    attached as the grader's reference_labels (via judge_run.json) and kept
    out of the task prompts: the integrity question is what the JUDGE can see.
    """
    out = work / "healthbench-meta-eval"
    out.mkdir(parents=True, exist_ok=True)

    lines = _get_jsonl_head(HB_META_EVAL_URL, N_SAMPLES)
    records = [json.loads(ln) for ln in lines[:N_SAMPLES]]
    assert len(records) == N_SAMPLES, f"expected {N_SAMPLES} records, got {len(records)}"

    tasks = []
    reference_labels: dict[str, str] = {}
    criterion_texts: list[str] = []
    for rec in records:
        tid = _healthbench_meta_id(rec["prompt_id"], rec["completion_id"])
        grades = rec["binary_labels"]
        reference_labels[tid] = "true" if _physician_majority(grades) else "false"
        tasks.append(
            {
                "id": tid,
                "prompt": _meta_eval_judge_input(rec),
                "metadata": {
                    "category": rec["category"],
                    "n_physician_grades": len(grades),
                    "prompt_id": rec["prompt_id"],
                    "completion_id": rec["completion_id"],
                },
            }
        )
        if len(criterion_texts) < 4:
            criterion_texts.append(rec["rubric"])

    (out / "dataset.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "eval_id": "HealthBench meta_eval via inspect_evals",
                "tasks": tasks,
                "notes": (
                    f"Definition sample: {N_SAMPLES} of 29,511 meta_eval records "
                    "(URL pinned in DATASET_URLS['meta_eval'], dataset.py). Each "
                    "record is a (conversation, pre-recorded completion) pair the "
                    "judge grades against one rubric criterion; the task prompt "
                    "is exactly the judge's input per _create_conversation_string. "
                    "The completion is the response under evaluation, not an "
                    "answer: the ground truth is the physician majority, which is "
                    "harness-side (judge_run.json reference_labels) and excluded "
                    "here."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "environment.json").write_text(
        json.dumps(
            {
                "env": {},
                "mounts": [],
                "notes": (
                    "The meta_eval subset declares no agent solver: completions "
                    "are pre-recorded responses under evaluation, not agent "
                    "outputs. The task definition declares no agent-visible "
                    "environment. The instrument under audit is the judge model "
                    "itself, validated against the physician labels."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "grader.json").write_text(
        json.dumps(
            {
                "kind": "judge",
                "judge": {
                    # Defaults from meta_evaluation_scorer().
                    "model": HB_META_JUDGE_MODEL,
                    "protocol": "pointwise",
                    "temperature": HB_META_JUDGE_TEMPERATURE,
                    "repeats": 1,
                    "rubric_criteria": criterion_texts,
                    # No scale_anchors: the criterion is graded
                    # criteria_met=true/false, not a scalar scale.
                    "scale_anchors": {},
                },
                "notes": (
                    "Judge defaults from meta_evaluation_scorer() in "
                    "meta_evaluation.py. The scorer grades the judge model's "
                    "criteria_met verdict on (conversation, completion, rubric) "
                    "against the physician majority of binary_labels and "
                    "reports judge-vs-physician agreement via macro_f1_metric. "
                    "The 12 sampled records' physician majorities are attached "
                    "as reference_labels in judge_run.json; the full 29,511-item "
                    "labeled set is the calibration evidence this card's "
                    "JUDGE-001 finding (or lack thereof) rests on."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "judge_run.json").write_text(
        json.dumps(
            {
                "judge": {
                    "model": HB_META_JUDGE_MODEL,
                    "protocol": "pointwise",
                    "temperature": HB_META_JUDGE_TEMPERATURE,
                },
                "judgments": [],
                "reference_labels": reference_labels,
                "notes": (
                    "No judge runs exist: this is a definition-level card. The "
                    "reference_labels are the physician majorities for the 12 "
                    "sampled records (\"true\"/\"false\"), drawn from the full "
                    "29,511-item physician-graded set pinned in the task "
                    "definition. Their presence is what clears JUDGE-001's "
                    "labeled-calibration-set finding."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "benchmark": "HealthBench meta_eval subset (OpenAI, 2025)",
                "task_definition": "inspect_evals/healthbench @ " + INSPECT_EVALS_COMMIT,
                "task_files_read": [
                    "healthbench.py",
                    "dataset.py",
                    "meta_evaluation.py",
                    "scorer.py",
                ],
                "dataset_url": HB_META_EVAL_URL,
                "dataset_split": "meta_eval (29,511 records per DATASET_URLS)",
                "sample": f"first {N_SAMPLES} JSONL records via ranged head fetch",
                "judge_defaults_source": (
                    "meta_evaluation_scorer() signature (judge_model, "
                    "judge_temperature)"
                ),
                "judge_input_format_source": (
                    "_create_conversation_string(state, completion) in scorer.py"
                ),
                "agreement_metric": (
                    "macro_f1_metric in meta_evaluation.py: judge criteria_met "
                    "vs physician majority over binary_labels"
                ),
                "reference_label_rule": (
                    "physician majority per calculate_physician_majority(): "
                    "sum(binary_labels) > len(binary_labels)/2, encoded as "
                    "\"true\"/\"false\""
                ),
                "excluded": [
                    "binary_labels (physician boolean grades): public in the dataset but harness-side; they are the ground truth the judge is validated against, attached as reference_labels via judge_run.json",
                    "anonymized_physician_ids: not needed to describe the instrument",
                ],
                "no_traces": True,
                "generated_by": "examples/report-cards/real/build_real_cards.py",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"healthbench-meta-eval: {len(tasks)} tasks -> {out}")
    return out


# Exact judge defaults from writingbench() / multi_scorer_wrapper() in
# writingbench.py at the pinned commit (read by a human; the machine link is
# this file's content hash).
WB_URL = (
    "https://raw.githubusercontent.com/UKGovernmentBEIS/inspect_evals/"
    "244e43cc924d1de5a78dad7db259bbd4471c97e7/src/inspect_evals/writingbench/"
    "benchmark_all.jsonl"
)
WB_JUDGE_MODEL = "anthropic/claude-3-5-haiku-latest"
WB_GRADE_PATTERN = r'"score"\s*:\s*(10|[1-9])'
WB_SCALE_ANCHORS = {
    "1-2": "Low score description: Critical deficiencies and major issues that prevent adequate functionality.",
    "3-4": "Below average score description: Lacking with noticeable shortcomings that impact overall effectiveness and require improvement.",
    "5-6": "Average score description: Adequate but not exemplary, Baseline performance that meets essential requirements. Most models may achieve this score.",
    "7-8": "Above average score description: Strong performance characterized by competent execution, though minor refinements are needed to achieve excellence.",
    "9-10": "High score description: Exceptional performance with all aspects optimally addressed, demonstrating superior effectiveness and quality without any flaws.",
}


def build_writingbench(work: Path) -> Path:
    """Translate the inspect_evals writingbench task definition (default args)."""
    out = work / "writingbench"
    out.mkdir(parents=True, exist_ok=True)

    lines = _get_jsonl_head(WB_URL, N_SAMPLES)
    records = [json.loads(ln) for ln in lines[:N_SAMPLES]]
    assert len(records) == N_SAMPLES, f"expected {N_SAMPLES} records, got {len(records)}"

    tasks = []
    for rec in records:
        checklist = rec["checklist"]
        tasks.append(
            {
                "id": str(rec["index"]),
                "prompt": rec["query"],
                "metadata": {
                    "domain1": rec["domain1"],
                    "domain2": rec["domain2"],
                    "n_checklist_items": len(checklist),
                    "checklist_names": [item["name"] for item in checklist],
                },
            }
        )
    # One verbatim checklist item, carried as grader-side evidence that the
    # per-criterion anchors exist (each item also carries its own 1-2..9-10
    # level texts beyond the generic scoring rules).
    criteria_sample = records[0]["checklist"][0]

    (out / "dataset.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "eval_id": "WritingBench via inspect_evals",
                "tasks": tasks,
                "notes": (
                    f"Definition sample: {N_SAMPLES} of 1000 queries from "
                    "benchmark_all.jsonl (URL pinned in the task source; the "
                    "file lives in the inspect_evals repo at the pinned "
                    "commit). The solver sees only the query. Each record "
                    "carries a 5-item checklist of grading criteria in its "
                    "metadata; those criterion texts are grader-side and "
                    "summarized here as names, with one verbatim item carried "
                    "in grader.json as the rubric_criteria sample."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "environment.json").write_text(
        json.dumps(
            {
                "env": {},
                "mounts": [],
                "notes": (
                    "The solver is plain `generate` (no tools, no sandbox) "
                    "with GenerateConfig(top_p=0.8, top_k=20, temperature=0.7, "
                    "max_tokens=16000). The task definition (writingbench.py) "
                    "declares no agent-visible environment. The checklist is "
                    "sample metadata used only in the scoring template; the "
                    "solver never sees it."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "grader.json").write_text(
        json.dumps(
            {
                "kind": "judge",
                "judge": {
                    # Defaults from writingbench() and multi_scorer_wrapper().
                    "model": WB_JUDGE_MODEL,
                    "protocol": "pointwise",
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "max_tokens": 2048,
                    "repeats": 1,
                    "n_criteria_per_sample": 5,
                    "aggregation": "mean",
                    "grade_pattern": WB_GRADE_PATTERN,
                    "scale_anchors": WB_SCALE_ANCHORS,
                    "rubric_criteria": criteria_sample,
                },
                "notes": (
                    "Five model_graded_qa scorers via multi_scorer, one per "
                    "checklist item, reduced by mean. The scoring template "
                    "carries the generic 1-10 anchored level descriptions "
                    "plus the per-item criterion text; each checklist item "
                    "additionally ships its own 1-2..9-10 level texts. Judge "
                    "completions that do not match the grade pattern are "
                    "dropped from the per-sample criterion mean rather than "
                    "zeroed (changelog 3-A); a sample whose criteria all fail "
                    "is unscored, visible only via unscored_samples. No "
                    "labeled calibration set ships in the eval definition: "
                    "the README's Validation Notes cover pipeline execution "
                    "(end-to-end runs, metric math, multilingual handling, "
                    "parse-failure surfacing), not judge-vs-human agreement."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "benchmark": "WritingBench (2025)",
                "task_definition": "inspect_evals/writingbench @ " + INSPECT_EVALS_COMMIT,
                "task_files_read": ["writingbench.py", "eval.yaml", "README.md"],
                "dataset_url": WB_URL,
                "dataset_split": "benchmark_all.jsonl (1000 queries per eval.yaml)",
                "sample": f"first {N_SAMPLES} JSONL records via ranged head fetch",
                "judge_defaults_source": (
                    "writingbench() task signature (judge_model) and "
                    "multi_scorer_wrapper() (grade pattern, judge GenerateConfig, "
                    "5 scorers, mean reducer); anchored level texts from "
                    "create_scoring_prompt()"
                ),
                "excluded": [
                    "checklist criterion texts (criteria_description + per-level anchors): public in the dataset but grader-side; the solver prompt carries only the query",
                ],
                "no_traces": True,
                "generated_by": "examples/report-cards/real/build_real_cards.py",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"writingbench: {len(tasks)} tasks -> {out}")
    return out


# Exact input prompt template: MultipleChoiceTemplate.SINGLE_ANSWER from
# inspect_ai (src/inspect_ai/solver/_multiple_choice.py), based on
# openai/simple-evals mmlu_eval.py. Read from UKGovernmentBEIS/inspect_ai
# main on 2026-09-29; the task resolves it via MMLU_MULTISHOT_QUESTION_TEMPLATE.
MMLU_PROMPT_TEMPLATE = (
    "Answer the following multiple choice question. The entire content of "
    "your response should be of the following format: 'ANSWER: $LETTER' "
    "(without quotes) where LETTER is one of {letters}.\n"
    "\n"
    "{question}\n"
    "\n"
    "{choices}"
)
MMLU_DATASET = "cais/mmlu"
MMLU_CONFIG = "all"
MMLU_SPLIT = "test"
# Revision pinned in the task source (MMLU_REVISION in mmlu.py).
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
MMLU_ROWS_URL = (
    f"https://datasets-server.huggingface.co/rows?dataset={MMLU_DATASET}"
    f"&config={MMLU_CONFIG}&split={MMLU_SPLIT}&offset=0&length={N_SAMPLES}"
)


def _format_mmlu_prompt(question: str, choices: list[str]) -> str:
    # Letters and choice lines verbatim from format_mmlu_question /
    # format_mmlu_choices in mmlu.py.
    letters = ",".join(chr(ord("a") + i) for i in range(len(choices)))
    choice_lines = "\n".join(
        f"({chr(ord('a') + i)}) {c}" for i, c in enumerate(choices)
    )
    return MMLU_PROMPT_TEMPLATE.format(
        letters=letters, question=question, choices=choice_lines
    )


def build_mmlu(work: Path) -> Path:
    """Translate the inspect_evals mmlu_0_shot task definition (default args)."""
    out = work / "mmlu"
    out.mkdir(parents=True, exist_ok=True)

    data = _get_json(MMLU_ROWS_URL)
    rows = [r["row"] for r in data["rows"]]
    assert len(rows) == N_SAMPLES, f"expected {N_SAMPLES} rows, got {len(rows)}"

    tasks = []
    for i, r in enumerate(rows):
        tasks.append(
            {
                "id": f"mmlu-test-{i:05d}",
                "prompt": _format_mmlu_prompt(r["question"], r["choices"]),
                "metadata": {
                    "subject": r["subject"],
                    "n_choices": len(r["choices"]),
                },
            }
        )

    (out / "dataset.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "eval_id": "MMLU via inspect_evals",
                "tasks": tasks,
                "notes": (
                    f"Definition sample: {N_SAMPLES} of 14042 test questions "
                    f"(57 subjects) from {MMLU_DATASET} config '{MMLU_CONFIG}' "
                    f"rev {MMLU_REVISION[:12]}. The solver sees only the "
                    "formatted multiple-choice question. The answer index is "
                    "public in the dataset but EXCLUDED here: it is the "
                    "harness-side sample target that choice() grades against, "
                    "which the agent never sees. Sample = first 12 rows in "
                    "dataset order via the datasets-server rows API; the "
                    "task default (mmlu_0_shot) shuffles with seed 42 after "
                    "dedup — the sample illustrates the instrument, not a "
                    "particular eval draw."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "environment.json").write_text(
        json.dumps(
            {
                "env": {},
                "mounts": [],
                "notes": (
                    "The solver is multiple_choice over plain `generate` (no "
                    "tools, no sandbox): GenerateConfig(temperature=0.0), "
                    "non-CoT max tokens default GPT_5_MIN_TOKENS (16) with a "
                    "model-dependent floor (get_max_tokens in mmlu.py). The "
                    "task definition (mmlu.py) declares no agent-visible "
                    "environment variables and no mounts."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "grader.json").write_text(
        json.dumps(
            {
                "kind": "script",
                "verifier": {
                    "path": (
                        "inspect_ai.scorer.choice: parses the model's selected "
                        "letter from the completion and compares it to the "
                        "harness-side target letter"
                    ),
                    "writable_by_agent": False,
                },
                "tests": ["letter-match (A/B/C/D)"],
                "notes": (
                    "choice() grades the selected letter against the sample "
                    "target, which lives in harness-side sample metadata. "
                    "Scoring runs harness-side after the agent submits; the "
                    "agent never sees the target and has no write path to the "
                    "scoring. Completions with no parseable letter score as "
                    "incorrect (0.0); there is no empty-output credit path. "
                    "No judge model is involved: JUDGE-001..006 do not apply."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "benchmark": "MMLU (Hendrycks et al., 2020)",
                "task_definition": "inspect_evals/mmlu @ " + INSPECT_EVALS_COMMIT,
                "task_files_read": ["mmlu.py", "eval.yaml"],
                "prompt_template_source": (
                    "MultipleChoiceTemplate.SINGLE_ANSWER from "
                    "UKGovernmentBEIS/inspect_ai "
                    "src/inspect_ai/solver/_multiple_choice.py (read "
                    "2026-09-29; based on openai/simple-evals mmlu_eval.py); "
                    "choice letters formatted per format_mmlu_choices in mmlu.py"
                ),
                "dataset": MMLU_DATASET,
                "dataset_config": MMLU_CONFIG,
                "dataset_split": MMLU_SPLIT,
                "dataset_revision": MMLU_REVISION,
                "sample": (
                    f"first {N_SAMPLES} rows via HuggingFace datasets-server "
                    "(offset 0, length 12)"
                ),
                "translation": (
                    "mechanical: prompt = SINGLE_ANSWER template over "
                    "question + lettered choices; metadata = subject only"
                ),
                "excluded": [
                    "answer (answer index 0-3): public in the dataset but "
                    "harness-side; it is the sample target that choice() "
                    "grades against"
                ],
                "no_traces": True,
                "generated_by": "examples/report-cards/real/build_real_cards.py",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"mmlu: {len(tasks)} tasks -> {out}")
    return out


# GPQA Diamond: the dataset is a single CSV whose URL and sha256 are pinned
# in the task source itself (gpqa.py: GPQA_DIAMOND_DATASET_URL,
# GPQA_DIAMOND_DATASET_SHA256, computed 2026-04-07).
GPQA_CSV_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
)
GPQA_CSV_SHA256 = (
    "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305"
)
# Fixed choice-shuffle seed from the task (DEFAULT_SHUFFLE_SEED in gpqa.py).
# The raw CSV lists the correct answer first, so the seeded shuffle makes the
# presented exam identical on every build.
GPQA_SHUFFLE_SEED = 42
# Exact prompt template: MultipleChoiceTemplate.SINGLE_ANSWER_COT from
# inspect_ai (src/inspect_ai/solver/_multiple_choice.py); the task's solver is
# multiple_choice(cot=True) by default. Choices are lettered per
# answer_options() in the same module ("A) ...").
GPQA_PROMPT_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your "
    "response should be of the following format: 'ANSWER: $LETTER' "
    "(without quotes) where LETTER is one of {letters}. Think step by step "
    "before answering.\n"
    "\n"
    "{question}\n"
    "\n"
    "{choices}"
)


def _get_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "evalwarden-card-builder/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _gpqa_shuffled_choices(
    records: list[dict[str, str]],
) -> list[tuple[list[str], str]]:
    """Replicate the task's seeded choice shuffle.

    Mirrors inspect_ai MemoryDataset.shuffle_choices(seed=42): ONE
    random.Random(42) shared across samples in dataset order; per sample the
    choice positions are shuffled and the target letter remapped (the correct
    answer is choices[0] in the raw CSV). Returns (shuffled choices,
    target letter) per record.
    """
    rand = random.Random(GPQA_SHUFFLE_SEED)
    out = []
    for rec in records:
        choices = [
            rec["Correct Answer"],
            rec["Incorrect Answer 1"],
            rec["Incorrect Answer 2"],
            rec["Incorrect Answer 3"],
        ]
        positions = list(range(len(choices)))
        rand.shuffle(positions)
        shuffled = [choices[i] for i in positions]
        target_letter = chr(ord("A") + positions.index(0))
        out.append((shuffled, target_letter))
    return out


def _format_gpqa_prompt(question: str, choices: list[str]) -> str:
    letters = ",".join(chr(ord("A") + i) for i in range(len(choices)))
    choice_lines = "\n".join(
        f"{chr(ord('A') + i)}) {c}" for i, c in enumerate(choices)
    )
    return GPQA_PROMPT_TEMPLATE.format(
        letters=letters, question=question, choices=choice_lines
    )


def build_gpqa(work: Path) -> Path:
    """Translate the inspect_evals gpqa_diamond task definition (default args)."""
    out = work / "gpqa"
    out.mkdir(parents=True, exist_ok=True)

    raw = _get_bytes(GPQA_CSV_URL)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != GPQA_CSV_SHA256:
        raise RuntimeError(
            f"gpqa_diamond.csv hash mismatch: got {digest}, "
            f"task source pins {GPQA_CSV_SHA256}"
        )
    records = list(csv.DictReader(raw.decode("utf-8").splitlines()))
    assert len(records) == 198, f"expected 198 rows, got {len(records)}"
    sample = records[:N_SAMPLES]
    shuffled = _gpqa_shuffled_choices(sample)

    tasks = []
    for rec, (choices, _target) in zip(sample, shuffled):
        tasks.append(
            {
                "id": rec["Record ID"],
                "prompt": _format_gpqa_prompt(rec["Question"], choices),
                "metadata": {
                    "high_level_domain": rec["High-level domain"],
                    "subdomain": rec["Subdomain"],
                },
            }
        )

    (out / "dataset.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "eval_id": "GPQA Diamond via inspect_evals",
                "tasks": tasks,
                "notes": (
                    f"Definition sample: {N_SAMPLES} of 198 questions from "
                    "gpqa_diamond.csv (URL and sha256 pinned in the task "
                    "source; hash verified before use). The solver sees only "
                    "the formatted multiple-choice question with the "
                    "seed-42-shuffled choices. The correct-answer identity "
                    "(target letter) is public in the dataset but EXCLUDED "
                    "here: it is the harness-side sample target that "
                    "choice() grades against, which the agent never sees. "
                    "Sample = first 12 CSV rows in file order; the task "
                    "applies no dataset-level shuffle, so this matches the "
                    "task's own sample order."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "environment.json").write_text(
        json.dumps(
            {
                "env": {},
                "mounts": [],
                "notes": (
                    "The solver is multiple_choice(cot=True) over plain "
                    "`generate` (no tools, no sandbox). The task definition "
                    "(gpqa.py) declares no agent-visible environment "
                    "variables and no mounts."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "grader.json").write_text(
        json.dumps(
            {
                "kind": "script",
                "verifier": {
                    "path": (
                        "inspect_ai.scorer.choice: parses the model's selected "
                        "letter from the completion and compares it to the "
                        "harness-side target letter"
                    ),
                    "writable_by_agent": False,
                },
                "tests": ["letter-match (A/B/C/D)"],
                "notes": (
                    "choice() grades the selected letter against the sample "
                    "target, which lives in harness-side sample metadata. "
                    "Scoring runs harness-side after the agent submits; the "
                    "agent never sees the target and has no write path to the "
                    "scoring. Completions with no parseable letter score as "
                    "incorrect (0.0); there is no empty-output credit path. "
                    "No judge model is involved: JUDGE-001..006 do not apply."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "benchmark": "GPQA Diamond (Rein et al., 2023)",
                "task_definition": "inspect_evals/gpqa @ " + INSPECT_EVALS_COMMIT,
                "task_files_read": ["gpqa.py", "eval.yaml"],
                "dataset_url": GPQA_CSV_URL,
                "dataset_sha256": GPQA_CSV_SHA256,
                "dataset_sha256_source": (
                    "GPQA_DIAMOND_DATASET_SHA256 in gpqa.py (pinned in the "
                    "task source; verified by this builder before use)"
                ),
                "dataset_split": "gpqa_diamond.csv (198 questions)",
                "sample": f"first {N_SAMPLES} CSV rows in file order",
                "choice_shuffle": (
                    "replicated from inspect_ai MemoryDataset.shuffle_choices "
                    "(single random.Random(42) across samples in order; "
                    "target letter remapped); seed from DEFAULT_SHUFFLE_SEED "
                    "in gpqa.py"
                ),
                "prompt_template_source": (
                    "MultipleChoiceTemplate.SINGLE_ANSWER_COT from "
                    "UKGovernmentBEIS/inspect_ai "
                    "src/inspect_ai/solver/_multiple_choice.py (read "
                    "2026-09-29); task solver is multiple_choice(cot=True)"
                ),
                "translation": (
                    "mechanical: prompt = SINGLE_ANSWER_COT template over "
                    "question + seed-shuffled lettered choices; metadata = "
                    "record id, high-level domain, subdomain"
                ),
                "excluded": [
                    "correct-answer identity (target letter): public in the dataset but harness-side; it is the sample target that choice() grades against",
                    "validator metadata columns (expert/non-expert validator accuracy, feedback, etc.): collected during benchmark construction; not read by the task definition",
                    "pre-revision fields: superseded by the revised columns the task reads",
                ],
                "no_traces": True,
                "generated_by": "examples/report-cards/real/build_real_cards.py",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"gpqa: {len(tasks)} tasks -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True, help="Where to write the artifacts.")
    parser.add_argument(
        "--only",
        choices=["swe-bench-verified", "healthbench", "healthbench-meta-eval", "writingbench", "mmlu", "gpqa"],
        default=None,
        help="Build just one artifact (default: all).",
    )
    args = parser.parse_args()
    work = Path(args.work_dir)
    builders = {
        "swe-bench-verified": build_swe_bench,
        "healthbench": build_healthbench,
        "healthbench-meta-eval": build_healthbench_meta_eval,
        "writingbench": build_writingbench,
        "mmlu": build_mmlu,
        "gpqa": build_gpqa,
    }
    for name, fn in builders.items():
        if args.only is None or args.only == name:
            fn(work)
    print("done. Next: evalwarden audit <dir> to review, then evalwarden report-cards <dirs>.")


if __name__ == "__main__":
    main()
