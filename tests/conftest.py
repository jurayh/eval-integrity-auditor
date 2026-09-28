"""Shared fixtures for the evalint test suite."""
from __future__ import annotations

from pathlib import Path

import pytest

from evalint.model import Attempt, Environment, Grader, IntegrityModel, Judgment, TaskSample

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_LEAKY = REPO_ROOT / "demo" / "leaky"
DEMO_HARDENED = REPO_ROOT / "demo" / "hardened"
DEMO_JUDGE_BAD = REPO_ROOT / "demo" / "judge_bad"
DEMO_JUDGE_CLEAN = REPO_ROOT / "demo" / "judge_clean"


def make_model(
    env_vars: dict[str, str] | None = None,
    mounts: list | None = None,
    grader: Grader | None = None,
    attempts: list[Attempt] | None = None,
    judgments: list[Judgment] | None = None,
) -> IntegrityModel:
    return IntegrityModel(
        eval_id="test-eval",
        adapter_name="test",
        adapter_version="0.0.0",
        tasks=[TaskSample(id="t1", prompt="do the thing")],
        environment=Environment(env_vars=env_vars or {}, mounts=mounts or []),
        grader=grader or Grader(),
        attempts=attempts or [],
        judgments=judgments or [],
    )


def make_judge_grader(**kwargs) -> Grader:
    defaults = dict(
        kind="judge",
        judge_model="judge-7b",
        protocol="pairwise",
        counterbalanced=True,
        temperature=0.0,
        repeats=3,
        rubric_criteria=["helpfulness"],
        scale_anchors={"1": "bad", "5": "good"},
        reference_labels={f"t{i:02d}": "sol-a" for i in range(1, 13)},
    )
    defaults.update(kwargs)
    return Grader(**defaults)


def make_judgment(task_id="t01", winner="sol-a", order=("a", "b"),
                 la=600, lb=400, repeat_index=0, scores=None) -> Judgment:
    return Judgment(
        task_id=task_id,
        candidates=["sol-a", "sol-b"],
        presentation_order=[f"sol-{c}" for c in order],
        winner=winner,
        scores=scores or {},
        lengths={"sol-a": la, "sol-b": lb},
        repeat_index=repeat_index,
    )


@pytest.fixture
def leaky_dir() -> Path:
    assert DEMO_LEAKY.is_dir(), "demo/leaky fixture missing"
    return DEMO_LEAKY


@pytest.fixture
def hardened_dir() -> Path:
    assert DEMO_HARDENED.is_dir(), "demo/hardened fixture missing"
    return DEMO_HARDENED
