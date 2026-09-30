"""Immutable, engine-free count contract for one fresh audited PPO run.

This is a count specification, not permission to perform any training or
physics. The root-owned execution freeze must bind its serialized identity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

BUDGET_SCHEMA = "d1-course-rl16-learning-budget-v1"
_INPUT_KEYS = frozenset(("total_controls", "n_steps", "batch_size", "n_epochs"))
_DERIVED_KEYS = frozenset((
    "rollouts", "train_calls", "minibatches_per_epoch", "epochs", "optimizer_steps",
))


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    """One-env PPO count bounds, with exact minibatch and rollout divisions."""

    total_controls: int
    n_steps: int
    batch_size: int
    n_epochs: int

    def __post_init__(self) -> None:
        for name in _INPUT_KEYS:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive non-bool integer")
        if self.total_controls % self.n_steps:
            raise ValueError("total_controls must be divisible by n_steps")
        if self.n_steps % self.batch_size:
            raise ValueError("n_steps must be divisible by batch_size")

    @property
    def rollouts(self) -> int:
        return self.total_controls // self.n_steps

    @property
    def train_calls(self) -> int:
        return self.rollouts

    @property
    def minibatches_per_epoch(self) -> int:
        return self.n_steps // self.batch_size

    @property
    def epochs(self) -> int:
        return self.train_calls * self.n_epochs

    @property
    def optimizer_steps(self) -> int:
        return self.epochs * self.minibatches_per_epoch

    def as_dict(self) -> dict[str, int | str]:
        """Canonical JSON-ready input and derived count claims."""
        return {
            "schema": BUDGET_SCHEMA,
            "total_controls": self.total_controls,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "rollouts": self.rollouts,
            "train_calls": self.train_calls,
            "minibatches_per_epoch": self.minibatches_per_epoch,
            "epochs": self.epochs,
            "optimizer_steps": self.optimizer_steps,
        }

    def canonical_sha256(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("ascii")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_dict(cls, value: Any) -> BudgetSpec:
        """Recompute every claimed count when reading a frozen sidecar."""
        if not isinstance(value, dict) or set(value) != _INPUT_KEYS | _DERIVED_KEYS | {"schema"}:
            raise ValueError("budget sidecar has missing or extra fields")
        if value["schema"] != BUDGET_SCHEMA:
            raise ValueError("budget sidecar has a different schema")
        if any(type(value[name]) is not int for name in _INPUT_KEYS | _DERIVED_KEYS):
            raise ValueError("budget sidecar counts must be exact non-bool integers")
        spec = cls(*(value[name] for name in (
            "total_controls", "n_steps", "batch_size", "n_epochs",
        )))
        if value != spec.as_dict():
            raise ValueError("budget sidecar derived counts differ from the inputs")
        return spec


__all__ = ("BUDGET_SCHEMA", "BudgetSpec")
