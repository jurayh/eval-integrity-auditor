"""ENV-001 tests. Precision matters more than recall: clean evals must not be flagged."""
from __future__ import annotations

from evalint.checks.env_leakage import EnvLeakageCheck
from evalint.model import Confidence, Mount, Severity

from .conftest import make_model

check = EnvLeakageCheck()


def test_exact_detection_signals_flagged_as_error():
    model = make_model(env_vars={"TASK_ID": "<redacted>", "RUN_ID": "<redacted>", "AGENT_TOKEN": "<redacted>"})
    findings = check.run(model)
    assert len(findings) == 3
    assert all(f.severity == Severity.ERROR for f in findings)
    assert all(f.confidence == Confidence.HIGH for f in findings)
    assert all(f.id == "ENV-001" for f in findings)


def test_clean_env_not_flagged():
    # INTERVAL_MS must not match the EVAL heuristic: token boundaries matter.
    model = make_model(env_vars={"PATH": "<redacted>", "HOME": "<redacted>", "MODEL_NAME": "<redacted>", "INTERVAL_MS": "<redacted>"})
    assert check.run(model) == []


def test_matching_is_case_insensitive():
    model = make_model(env_vars={"task_id": "<redacted>"})
    findings = check.run(model)
    assert len(findings) == 1
    assert findings[0].severity == Severity.ERROR


def test_heuristic_token_is_high_medium_not_error():
    model = make_model(env_vars={"MY_EVAL_RUN": "<redacted>"})
    findings = check.run(model)
    assert len(findings) == 1
    assert findings[0].severity == Severity.HIGH
    assert findings[0].confidence == Confidence.MEDIUM


def test_answer_bearing_mount_flagged():
    model = make_model(
        mounts=[Mount(path="/fixtures/gold_map.json", mode="ro", agent_access="read")]
    )
    findings = check.run(model)
    assert len(findings) == 1
    assert findings[0].severity == Severity.ERROR
    assert "gold_map.json" in findings[0].title


def test_similar_but_innocent_filename_not_flagged():
    # "goldenretriever.py" tokenizes to ["goldenretriever"] -- no match.
    model = make_model(
        mounts=[Mount(path="/data/goldenretriever.py", mode="ro", agent_access="read")]
    )
    assert check.run(model) == []


def test_inaccessible_mount_not_flagged():
    model = make_model(
        mounts=[Mount(path="/grader/gold.json", mode="ro", agent_access="none")]
    )
    assert check.run(model) == []


def test_secret_values_never_stored():
    # The adapter redacts values; the check only ever sees names.
    model = make_model(env_vars={"TASK_ID": "<redacted>"})
    findings = check.run(model)
    assert findings
    assert "tok-" not in findings[0].evidence[0]
