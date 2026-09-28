# Flagship demo: a deliberately broken benchmark

**Thesis:** a benchmark can report 100% because the solver recognizes task IDs.
The auditor shows why that score is invalid.

## The three beats

1. **Run.** The cheating solver prints 3/3 PASS. See `run.json`: it reads
   `TASK_ID` from the environment, looks up `gold_map.json`, and submits the
   gold patch. No real work.
2. **Audit.** Run `evalwarden audit demo/leaky` (or `evalwarden demo`). The linter
   flags the exact leak channels:
   - `ENV-001`: `TASK_ID`, `RUN_ID`, `AGENT_TOKEN` visible to the agent
     (eval-detection signals)
   - `ENV-001`: `gold_map.json` mounted where the agent can read it
     (answer-bearing state)
   - `GRAD-001`: the verifier at `/grader` is writable by the agent
3. **Harden.** See `../hardened/`: task IDs are opaque, no gold is mounted,
   the verifier is read-only. The cheat fails there while a genuine solver
   passes -- and the auditor reports no blocking findings.

All fixtures are synthetic. No proprietary data or model key is required.
