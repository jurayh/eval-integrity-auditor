# promptfoo_clean — a clean Promptfoo eval

The counterpart to `promptfoo_bad`: same support-ticket classifier shape, but

- `promptfooconfig.yaml` declares only an innocuous `MODEL_NAME` in `env`,
- grading is fully deterministic (`equals` / `contains`), no model judge,
- every run reports token usage and latency.

The audit produces zero findings. Run it with:

```bash
evalwarden demo --fixture promptfoo_clean
```
