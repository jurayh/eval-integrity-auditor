"""CLI tests via typer's CliRunner."""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from evalint.cli import app

from .conftest import DEMO_HARDENED, DEMO_LEAKY

runner = CliRunner()


def test_audit_leaky_exits_1_and_writes_report(tmp_path: Path):
    report = tmp_path / "report.html"
    result = runner.invoke(app, ["audit", str(DEMO_LEAKY), "--output", str(report)])
    assert result.exit_code == 1, result.output
    assert "BLOCKED" in result.output
    assert "ENV-001" in result.output
    assert "GRAD-001" in result.output
    assert report.is_file()
    html = report.read_text()
    assert "ENV-001" in html
    assert "GRAD-001" in html
    assert "confidence" in html


def test_audit_hardened_exits_0(tmp_path: Path):
    report = tmp_path / "report.html"
    result = runner.invoke(app, ["audit", str(DEMO_HARDENED), "--output", str(report)])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output
    assert "no blocking findings" in result.output


def test_audit_json_output(tmp_path: Path):
    report = tmp_path / "report.html"
    payload = tmp_path / "findings.json"
    result = runner.invoke(
        app, ["audit", str(DEMO_LEAKY), "--output", str(report), "--json", str(payload)]
    )
    assert result.exit_code == 1
    import json

    data = json.loads(payload.read_text())
    assert data["verdict"] == "BLOCKED"
    assert any(f["id"] == "ENV-001" for f in data["findings"])
    assert all("confidence" in f for f in data["findings"])


def test_audit_unknown_path_exits_2():
    result = runner.invoke(app, ["audit", "/does/not/exist"])
    assert result.exit_code == 2


def test_explain_known_check():
    result = runner.invoke(app, ["explain", "ENV-001"])
    assert result.exit_code == 0
    assert "Eval-detection" in result.output


def test_explain_unknown_check():
    result = runner.invoke(app, ["explain", "NOPE-999"])
    assert result.exit_code == 2


def test_fail_on_error_still_blocks_leaky(tmp_path: Path):
    report = tmp_path / "report.html"
    result = runner.invoke(
        app, ["audit", str(DEMO_LEAKY), "--output", str(report), "--fail-on", "error"]
    )
    assert result.exit_code == 1
