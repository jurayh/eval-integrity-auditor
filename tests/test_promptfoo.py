"""Tests for the promptfoo adapter: detection, normalization, redaction, CLI."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalint.adapters import AuditError, Confidence, autodetect
from evalint.adapters.promptfoo import ADAPTER_NAME, ADAPTER_VERSION, PromptfooAdapter
from evalint.cli import app
from evalint.engine import audit_with_policy

from .conftest import DEMO_PROMPTFOO_BAD, DEMO_PROMPTFOO_CLEAN

runner = CliRunner()
adapter = PromptfooAdapter()


def _audit(path: Path):
    result, _policy_failed = audit_with_policy(path, adapter_name="promptfoo")
    return result


# --- detection ------------------------------------------------------------


def test_detect_high_for_config_plus_results():
    assert adapter.detect(DEMO_PROMPTFOO_BAD) is Confidence.HIGH
    assert adapter.detect(DEMO_PROMPTFOO_CLEAN) is Confidence.HIGH


def test_detect_medium_for_config_only(tmp_path: Path):
    (tmp_path / "promptfooconfig.yaml").write_text("description: cfg only\ntests: []\n")
    assert adapter.detect(tmp_path) is Confidence.MEDIUM


def test_detect_low_for_empty_dir(tmp_path: Path):
    assert adapter.detect(tmp_path) is Confidence.LOW


def test_autodetect_picks_promptfoo_for_fixture_dir():
    found = autodetect(DEMO_PROMPTFOO_BAD)
    assert found.name == ADAPTER_NAME
    assert found.version == ADAPTER_VERSION


def test_autodetect_accepts_yml_extension(tmp_path: Path):
    (tmp_path / "promptfooconfig.yml").write_text("description: yml variant\n")
    assert adapter.detect(tmp_path) is Confidence.MEDIUM


# --- normalization ---------------------------------------------------------


def test_normalize_bad_fixture_shape():
    model = adapter.normalize(adapter.collect(DEMO_PROMPTFOO_BAD))
    assert model.adapter_name == ADAPTER_NAME
    assert model.adapter_version == ADAPTER_VERSION
    assert model.eval_id == "support-ticket classifier eval (demo)"
    assert len(model.tasks) == 2
    assert model.tasks[0].id == "billing ticket"
    assert set(model.tasks[0].metadata["assertions"]) == {"llm-rubric"}
    assert model.tasks[0].metadata["vars"] == ["expected", "ticket"]


def test_env_names_kept_values_redacted():
    model = adapter.normalize(adapter.collect(DEMO_PROMPTFOO_BAD))
    assert set(model.environment.env_vars) == {"TASK_ID", "RUN_ID", "MODEL_NAME"}
    assert all(v == "<redacted>" for v in model.environment.env_vars.values())


def test_llm_rubric_maps_to_judge_grader():
    model = adapter.normalize(adapter.collect(DEMO_PROMPTFOO_BAD))
    grader = model.grader
    assert grader.kind == "judge"
    assert grader.judge_model == "openai:gpt-4o"
    assert "llm-rubric" in grader.tests
    assert grader.rubric_criteria == [
        "The predicted label must match the true category of the ticket."
    ] * 2


def test_attempts_map_tokens_latency_status():
    model = adapter.normalize(adapter.collect(DEMO_PROMPTFOO_BAD))
    assert len(model.attempts) == 2
    first = model.attempts[0]
    assert first.status == "pass"
    assert first.score == 1.0
    assert first.tokens_in == 42
    assert first.tokens_out == 3
    assert first.latency_s == pytest.approx(0.812)
    assert not first.empty_submission


def test_failed_and_errored_rows_map_status():
    bundle = adapter.collect(DEMO_PROMPTFOO_CLEAN)
    rows = bundle["results"]["results"]["results"]
    rows[0]["success"] = False
    rows[1]["error"] = "provider timeout"
    model = adapter.normalize(bundle)
    assert model.attempts[0].status == "fail"
    assert model.attempts[1].status == "error"


def test_results_only_dir_uses_embedded_config(tmp_path: Path):
    envelope = json.loads((DEMO_PROMPTFOO_BAD / "results.json").read_text())
    envelope["config"] = {
        "description": "from envelope",
        "env": {"TASK_ID": "x"},
        "tests": [{"description": "t1", "assert": [{"type": "equals"}]}],
    }
    (tmp_path / "results.json").write_text(json.dumps(envelope))
    assert adapter.detect(tmp_path) is Confidence.MEDIUM
    model = adapter.normalize(adapter.collect(tmp_path))
    assert model.eval_id == "from envelope"
    assert "TASK_ID" in model.environment.env_vars
    assert any("embedded in the results envelope" in u for u in model.unsupported)


def test_unsupported_version_rejected(tmp_path: Path):
    envelope = json.loads((DEMO_PROMPTFOO_BAD / "results.json").read_text())
    envelope["results"]["version"] = 2
    (tmp_path / "results.json").write_text(json.dumps(envelope))
    (tmp_path / "promptfooconfig.yaml").write_text("description: x\n")
    with pytest.raises(AuditError, match="unsupported results version"):
        adapter.normalize(adapter.collect(tmp_path))


def test_unknown_key_surfaces_in_unsupported(tmp_path: Path):
    (tmp_path / "promptfooconfig.yaml").write_text(
        "description: x\nfutureField: 1\ntests: []\n"
    )
    model = adapter.normalize(adapter.collect(tmp_path))
    assert any("futureField" in u for u in model.unsupported)


def test_config_only_audit_degrades_gracefully(tmp_path: Path):
    (tmp_path / "promptfooconfig.yaml").write_text(
        "description: cfg only\nenv:\n  TASK_ID: planted\ntests: []\n"
    )
    model = adapter.normalize(adapter.collect(tmp_path))
    assert model.attempts == []
    assert model.environment.env_vars == {"TASK_ID": "<redacted>"}
    assert model.digests  # every file read is digested


# --- adapter contract ------------------------------------------------------


def test_collect_is_read_only():
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in DEMO_PROMPTFOO_BAD.iterdir()
    }
    adapter.normalize(adapter.collect(DEMO_PROMPTFOO_BAD))
    after = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in DEMO_PROMPTFOO_BAD.iterdir()
    }
    assert before == after


def test_no_secret_values_anywhere_in_model():
    model = adapter.normalize(adapter.collect(DEMO_PROMPTFOO_BAD))
    dump = repr(model)
    assert "support-classifier-v3" not in dump
    assert "run-demo-001" not in dump
    assert "The predicted label matches the true category" not in dump  # grader reason


def test_findings_point_at_real_filenames():
    result = _audit(DEMO_PROMPTFOO_BAD)
    env_finding = next(f for f in result.findings if f.id == "ENV-001")
    files = {loc.file for loc in env_finding.locations}
    assert files == {"promptfooconfig.yaml"}


# --- end to end via the checks ----------------------------------------------


def test_bad_fixture_triggers_env_and_judge_findings():
    result = _audit(DEMO_PROMPTFOO_BAD)
    ids = [f.id for f in result.findings]
    assert ids.count("ENV-001") == 2
    assert result.verdict == "BLOCKED"
    judge = [f for f in result.findings if f.id == "JUDGE-001"]
    assert len(judge) == 2  # no calibration set, no scale anchors
    assert all(f.confidence == "high" for f in judge)


def test_clean_fixture_produces_zero_findings():
    result = _audit(DEMO_PROMPTFOO_CLEAN)
    assert result.findings == []
    assert result.verdict == "PASS"
    assert result.score == 100
    assert result.model.grader.kind == "script"


# --- CLI --------------------------------------------------------------------


def test_cli_audit_bad_fixture_exits_1(tmp_path: Path):
    report = tmp_path / "report.html"
    result = runner.invoke(app, ["audit", str(DEMO_PROMPTFOO_BAD), "--output", str(report)])
    assert result.exit_code == 1, result.output
    assert "BLOCKED" in result.output
    assert "ENV-001" in result.output
    assert "JUDGE-001" in result.output
    assert "promptfoo 0.5.0" in result.output
    assert report.is_file()


def test_cli_audit_clean_fixture_exits_0():
    result = runner.invoke(app, ["audit", str(DEMO_PROMPTFOO_CLEAN)])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_cli_demo_fixtures_exit_0():
    for fixture in ("promptfoo_bad", "promptfoo_clean"):
        result = runner.invoke(app, ["demo", "--fixture", fixture])
        assert result.exit_code == 0, (fixture, result.output)
