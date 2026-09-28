# evalint — a linter for agent evaluations

**This is a linter for evals, not another eval framework.** It audits the
measurement system around a score — the dataset, the evidence boundary, the
grader, the environment, the run records — and tells you whether the score
can be trusted. It never runs your evals and never changes your harness.

One minute from "100% pass" to "invalid score," with the exact leak highlighted.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# The flagship demo: a deliberately broken benchmark
evalint demo
# or audit it directly:
evalint audit demo/leaky --output evalint-report.html
open evalint-report.html

# The hardened version passes clean:
evalint audit demo/hardened
```

## The flagship demo

A tiny synthetic coding benchmark reports **3/3 PASS** — because the solver
reads `TASK_ID` from the environment, looks up `gold_map.json`, and submits
the gold patch. No real work. The auditor flags the exact leak channels:

- `ENV-001` — `TASK_ID`, `RUN_ID`, `AGENT_TOKEN` visible to the agent
  (eval-detection signals)
- `ENV-001` — `gold_map.json` mounted where the agent can read it
  (answer-bearing state)
- `GRAD-001` — the verifier is writable by the agent

Then see `demo/hardened/`: opaque task IDs, no gold mounted, read-only
verifier. The cheat fails there while a genuine solver passes — and the audit
is clean. All fixtures are synthetic; no model key required.

## Checks (v0.1)

Deterministic, high-precision checks only. A linter that cries contamination
on a clean eval is worse than no auditor — so every finding carries a
**confidence** label, and statistical probes stay out until v0.2.

| ID | Check | Default severity |
|----|-------|------------------|
| ENV-001 | Eval-detection signal or leaked state visible to the agent | Error |
| GRAD-001 | Verifier writable by the agent | Error |
| GRAD-002 | Grader grants credit without completion | Error |
| COST-001 | Cost per success not reported | Medium |

`evalint explain ENV-001` prints any check's threat model, evidence, and fix.

## How it works

Adapters translate harness artifacts into a framework-neutral integrity model;
checks only ever see that model. That boundary is what keeps this a linter.

```
eval artifact/ ──▶ adapter (inspect) ──▶ integrity model ──▶ checks ──▶ report
     read-only, offline              data · boundary · grader · runs
```

v0.1 ships one adapter (`inspect`, for Inspect-style eval artifact
directories). Promptfoo, Harbor, and BrowserGym adapters plug into the same
registry — see `src/evalint/adapters/__init__.py`.

## CLI

```
evalint audit <eval-artifact> [--adapter auto|inspect] [--output report.html]
                              [--json findings.json] [--fail-on high]
evalint demo                  # run the flagship demo
evalint explain <CHECK-ID>     # threat, evidence, remediation
```

Exit codes: `0` policy passes · `1` findings cross `--fail-on` ·
`2` the audit could not complete. Same policy runs locally and as a CI gate.

## Reports

Self-contained HTML (inline CSS, no JavaScript, no remote assets) plus
terminal and JSON output. Secret values are never stored — only variable
*names* enter the model. Integrity scores are diagnostic, not a certification:
the report says "no blocking findings observed under this policy," never
"certified safe."

## Non-goals

Running or scheduling evaluations · replacing task/solver/scorer APIs ·
trace observability · generic red-teaming · public leaderboards · declaring
any benchmark contamination-free.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The test suite is the product's credibility: every rule has positive,
negative, and precision fixtures (clean evals must *not* be flagged), the
adapter has a read-only contract test, and the leaky/hardened fixtures run
end to end.
