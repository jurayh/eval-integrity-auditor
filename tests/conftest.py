"""Shared fixtures for the evalint test suite."""
from __future__ import annotations

from pathlib import Path

import pytest

from evalint.model import Attempt, Environment, Grader, IntegrityModel, TaskSample

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_LEAKY = REPO_ROOT / "demo" / "leaky"
DEMO_HARDENED = REPO_ROOT / "demo" / "hardened"


def make_model(
    env_vars: dict[str, str] | None = None,
    mounts: list | None = None,
    grader: Grader | None = None,
    attempts: list[Attempt] | None = None,
) -> IntegrityModel:
    return IntegrityModel(
        eval_id="test-eval",
        adapter_name="test",
        adapter_version="0.0.0",
        tasks=[TaskSample(id="t1", prompt="do the thing")],
        environment=Environment(env_vars=env_vars or {}, mounts=mounts or []),
        grader=grader or Grader(),
        attempts=attempts or [],
    )


@pytest.fixture
def leaky_dir() -> Path:
    assert DEMO_LEAKY.is_dir(), "demo/leaky fixture missing"
    return DEMO_LEAKY


@pytest.fixture
def hardened_dir() -> Path:
    assert DEMO_HARDENED.is_dir(), "demo/hardened fixture missing"
    return DEMO_HARDENED
