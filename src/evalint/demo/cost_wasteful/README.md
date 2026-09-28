# Demo: a wasteful run

**Thesis:** a pass rate without cost data tells you nothing about what the
score cost. This synthetic run passes 3 of 7 tasks but burns budget badly:

- `COST-002` — each success took 2.67 attempts on average (retries on t1, t6)
- `COST-003` — 91% of estimated spend went to attempts that never passed
  (expensive failures on t2 and t5, plus the runaway below)
- `COST-004` — one attempt on t7 burned 22,000 output tokens, 22x the typical
  attempt: an unbounded retry loop aborted by the wall clock

Compare `../cost_clean/`: the same tasks solved first try, no cost findings.

```bash
evalint demo --fixture cost_wasteful
evalint audit demo/cost_wasteful --budget-per-task 0.05
```

All fixtures are synthetic. Token prices are estimates, labeled as such.
