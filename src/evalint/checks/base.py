"""Check protocol: every rule implements this, and registers in checks/__init__."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..model import Finding, IntegrityModel


@dataclass
class CheckMeta:
    id: str  # Stable rule ID, e.g. "ENV-001".
    title: str
    threat: str  # Why it matters.
    remediation: str


class Check:
    """A single integrity rule. Pure function over the normalized model."""

    meta: CheckMeta

    def run(self, model: IntegrityModel) -> list[Finding]:
        raise NotImplementedError

    def with_options(self, **kwargs: Any) -> "Check":
        """Return this check with runtime options applied.

        The default implementation ignores every option and returns self.
        Checks that need CLI-supplied parameters (token prices, budgets)
        override this instead of being special-cased in the engine: the
        engine calls with_options on every registered check, so adding a
        parameterized check needs no engine changes.
        """
        return self
