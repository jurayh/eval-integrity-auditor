# promptfoo_bad — planted integrity issues in a Promptfoo eval

A deliberately compromised Promptfoo artifact directory:

- `promptfooconfig.yaml` plants `TASK_ID` and `RUN_ID` in top-level `env`,
  visible to the agent. That is the **ENV-001** eval-detection signal.
- The config also grades with `llm-rubric` assertions but declares no labeled
  calibration set and no anchored scale, which **JUDGE-001** catches.

Run it with:

```bash
evalint demo --fixture promptfoo_bad
```
