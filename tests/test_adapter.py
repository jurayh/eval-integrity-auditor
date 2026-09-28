"""Adapter tests: detection, normalization, and the read-only guarantee."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from evalint.adapters import AuditError, autodetect
from evalint.adapters.inspect_ai import InspectAdapter
from evalint.model import Confidence

from .conftest import DEMO_HARDENED, DEMO_LEAKY

adapter = InspectAdapter()


def _hashes(path: Path) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.iterdir())
        if p.is_file()
    }


def test_detect_leaky_fixture_high():
    assert adapter.detect(DEMO_LEAKY) == Confidence.HIGH


def test_detect_hardened_fixture_high():
    assert adapter.detect(DEMO_HARDENED) == Confidence.HIGH


def test_detect_unknown_dir_low(tmp_path: Path):
    assert adapter.detect(tmp_path) == Confidence.LOW


def test_autodetect_unknown_dir_raises(tmp_path: Path):
    with pytest.raises(AuditError):
        autodetect(tmp_path)


def test_normalize_leaky_fixture():
    bundle = adapter.collect(DEMO_LEAKY)
    model = adapter.normalize(bundle)
    assert model.eval_id == "tinycode-leaky-1.0"
    assert model.adapter_name == "inspect"
    assert len(model.tasks) == 3
    assert set(model.environment.env_vars) == {"TASK_ID", "RUN_ID", "AGENT_TOKEN", "MODEL"}
    # Values are redacted: names are the signal, secrets never enter the model.
    assert all(v == "<redacted>" for v in model.environment.env_vars.values())
    assert model.grader.verifier_writable_by_agent is True
    assert len(model.attempts) == 3
    assert all(a.status == "pass" for a in model.attempts)


def test_collect_is_read_only(leaky_dir: Path):
    before = _hashes(leaky_dir)
    bundle = adapter.collect(leaky_dir)
    adapter.normalize(bundle)
    assert _hashes(leaky_dir) == before


def test_unsupported_fields_reported_not_dropped(tmp_path: Path):
    (tmp_path / "dataset.json").write_text(
        '{"schema_version": "evalint-artifact-v1", "eval_id": "x", "tasks": [], "future_field": 1}'
    )
    (tmp_path / "grader.json").write_text('{"kind": "script"}')
    model = adapter.normalize(adapter.collect(tmp_path))
    assert "dataset.json:future_field" in model.unsupported
