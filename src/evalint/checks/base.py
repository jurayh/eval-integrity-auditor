"""Check protocol: every rule implements this, and registers in checks/__init__."""
from __future__ import annotations

from dataclasses import dataclass

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
