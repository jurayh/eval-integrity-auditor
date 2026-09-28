# Hardened fixture: the same benchmark, fixed

The fixes, relative to `../leaky/`:

- Task IDs are opaque (`a3f9c1`, ...) instead of stable, guessable names.
- No `TASK_ID` / `RUN_ID` / `AGENT_TOKEN` in the solver's environment.
- No gold material is mounted where the solver can read it.
- The verifier is mounted read-only (`writable_by_agent: false`).

`run.json` is a genuine solver doing real work (3/3 PASS). `run_cheat.json`
shows the same cheating solver failing here: with no task ID and no gold map,
there is nothing to exploit.

`evalint audit demo/hardened` reports no blocking findings.
