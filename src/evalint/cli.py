"""CLI: evalint audit | demo | explain. A linter-style interface.

Exit codes: 0 = policy passes, 1 = findings cross the --fail-on threshold,
2 = the audit could not complete.
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

import evalint
from .adapters import AuditError
from .checks import BY_ID
from .engine import AuditResult, audit_with_policy
from .reporters import render_html, render_terminal

app = typer.Typer(
    help="A linter for agent evaluations. Not another eval framework.",
    no_args_is_help=True,
)


def _findings_json(result: AuditResult) -> dict:
    return {
        "eval_id": result.model.eval_id,
        "adapter": f"{result.model.adapter_name} {result.model.adapter_version}",
        "verdict": result.verdict,
        "score": result.score,
        "findings": [
            {
                "id": f.id,
                "title": f.title,
                "severity": f.severity.value,
                "confidence": f.confidence.value,
                "description": f.description,
                "evidence": f.evidence,
                "locations": [loc.render() for loc in f.locations],
                "remediation": f.remediation,
                "fingerprint": f.fingerprint,
            }
            for f in result.findings
        ],
        "digests": result.model.digests,
        "unsupported": result.model.unsupported,
    }


@app.command()
def audit(
    path: Path = typer.Argument(..., help="Path to the eval artifact directory."),
    adapter: str = typer.Option("auto", help="Adapter to use: auto, inspect."),
    output: Path = typer.Option(
        Path("evalint-report.html"), help="Where to write the self-contained HTML report."
    ),
    json_output: Path | None = typer.Option(
        None, "--json", help="Also write machine-readable JSON findings."
    ),
    fail_on: str = typer.Option(
        "high", help="Minimum finding severity that fails the audit: error|high|medium|low."
    ),
    price_in: float = typer.Option(3.0, help="Estimated USD per 1M input tokens."),
    price_out: float = typer.Option(15.0, help="Estimated USD per 1M output tokens."),
) -> None:
    """Audit an eval artifact and write an integrity report."""
    if fail_on not in ("error", "high", "medium", "low"):
        typer.echo("error: --fail-on must be one of error|high|medium|low", err=True)
        raise typer.Exit(2)
    try:
        result, policy_failed = audit_with_policy(
            path,
            adapter_name=adapter,
            fail_on=fail_on,
            price_in_per_1m=price_in,
            price_out_per_1m=price_out,
        )
    except AuditError as exc:
        typer.echo(f"error: audit could not complete: {exc}", err=True)
        raise typer.Exit(2)
    typer.echo(render_terminal(result))
    output.write_text(render_html(result, evalint.__version__), encoding="utf-8")
    typer.echo(f"Report: {output}")
    if json_output is not None:
        json_output.write_text(json.dumps(_findings_json(result), indent=2), encoding="utf-8")
        typer.echo(f"JSON: {json_output}")
    raise typer.Exit(1 if policy_failed else 0)


def _demo_dir() -> Path:
    here = Path(evalint.__file__).resolve().parent  # <repo>/src/evalint
    candidates = [
        here.parent.parent / "demo" / "leaky",  # editable install: repo root
        Path.cwd() / "demo" / "leaky",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise AuditError("demo fixture not found (expected demo/leaky/ next to the repo root)")


@app.command()
def demo(
    output: Path = typer.Option(
        Path("evalint-demo-report.html"), help="Where to write the demo HTML report."
    ),
) -> None:
    """Run the flagship demo: audit the deliberately leaky eval fixture."""
    try:
        fixture = _demo_dir()
        result, policy_failed = audit_with_policy(fixture)
    except AuditError as exc:
        typer.echo(f"error: demo could not run: {exc}", err=True)
        raise typer.Exit(2)
    typer.echo("beat 1 -- run: the cheating solver reports 3/3 PASS (see demo/leaky/run.json).")
    typer.echo("beat 2 -- audit: the linter shows why that score is invalid.")
    typer.echo("")
    typer.echo(render_terminal(result))
    output.write_text(render_html(result, evalint.__version__), encoding="utf-8")
    typer.echo(f"Report: {output}")
    typer.echo("beat 3 -- harden: see demo/hardened/ for the fixed eval (opaque IDs, no gold access).")
    raise typer.Exit(1 if policy_failed else 0)


@app.command()
def explain(check_id: str = typer.Argument(..., help="Check ID, e.g. ENV-001.")) -> None:
    """Explain a check: the threat, what evidence it needs, how to fix it."""
    check = BY_ID.get(check_id.upper())
    if check is None:
        known = ", ".join(sorted(BY_ID))
        typer.echo(f"unknown check {check_id!r} (known: {known})", err=True)
        raise typer.Exit(2)
    meta = check.meta
    typer.echo(f"{meta.id}: {meta.title}")
    typer.echo("")
    typer.echo(f"threat: {meta.threat}")
    typer.echo("")
    typer.echo(f"fix: {meta.remediation}")


if __name__ == "__main__":
    app()
