# Real benchmark report cards

These cards audit **real public benchmarks** from their published eval
definitions, not constructed fixtures. They are what the tool says about
real artifacts, with nothing invented: no traces were fabricated, no evals
were run, and every input is pinned and reproducible.

## The cards

| Card | Result | What it means |
|------|--------|---------------|
| [SWE-bench Verified via inspect_evals](swe-bench-verified-via-inspect-evals.html) | 100/100 PASS | The definition keeps gold patches, test patches, and grading harness-side where the agent cannot reach them. No findings. |
| [HealthBench via inspect_evals](healthbench-via-inspect-evals.html) | 85/100 BLOCKED | The model judge (`openai/gpt-4o-mini`) ships in the definition with no calibration set attached. |

## How to read the HealthBench card

The BLOCKED verdict needs one paragraph of context, and here it is.

JUDGE-001 fires because the **eval definition artifact** carries no labeled
calibration set for its judge. That statement is precise and checkable: open
`artifacts/healthbench/grader.json` and you will find no reference labels.
The card is about the artifact, and the artifact is what a downstream user
actually audits.

What the card does **not** claim: that HealthBench's authors never validated
their grader. They did, in the paper, through a separate `healthbench_meta_eval`
maintenance task (29,511 physician-graded items scored against the judge with
macro F1). That validation lives outside the eval definition, which is exactly
the gap the finding points at: if you pick up the definition alone, the judge
is unvalidated *as far as you can verify*. The fix is correspondingly small:
attach the meta-eval agreement numbers, or a pointer to them, to the artifact.

The second JUDGE-001 line (no scale anchors, medium severity) is the softest
signal on the card and is labeled as such: HealthBench grades boolean
`criteria_met` per rubric item rather than a scalar scale, so anchored scale
levels apply less directly, and the grader template already ships worked
examples of criterion application. It stays on the card because the linter is
mechanical: it reports what the definition declares, with a confidence label,
and the human decides. That is the intended workflow, not a false positive
to be tuned away.

## Provenance and reproduction

Each artifact directory carries a `PROVENANCE.json`: pinned sources,
exactly what was translated, what was excluded and why. Rebuild with:

```bash
python3 build_real_cards.py --work-dir /tmp/real_work
evalwarden report-cards /tmp/real_work/swe-bench-verified /tmp/real_work/healthbench \
    --output-dir /tmp/real_cards/
```

Pinned inputs:

- `inspect_evals` @ `244e43cc924d1de5a78dad7db259bbd4471c97e7`
  (task definitions: `swe_bench`, `healthbench`)
- SWE-bench Verified: `princeton-nlp/SWE-bench_Verified`
  rev `c104f840cc67f8b6eec6f759ebc8b2693d585d4a` (pinned by the task source)
- HealthBench full split: the public Azure blob URL pinned in the task source

Definition samples are 12 instances each; the cards audit the measurement
setup (environment boundary, grader, judge config), which does not vary
across instances.

## Scope honesty

These cards audit definitions, not runs. Without run records, the
cost checks, the attempt-based grader checks, and the behavioral judge
checks (repeats, agreement, bias) cannot fire, and the tool correctly stays
silent about them. A definition-level card is a floor, not a ceiling: it
catches structural problems in the measurement setup, which is where the
cheapest and most embarrassing failures live.
