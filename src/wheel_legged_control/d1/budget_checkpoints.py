"""Budget-scheduled PPO checkpoint writer for the D1 locomotion experiment.

This module implements a deterministic, audit-consistent checkpoint scheduler
that saves the learner exactly at pre-declared timestep budgets. It owns its own
``manifest.json`` and ``phases.jsonl`` artefacts and never loads models or resets
environments itself.

Adapted from a Claude Opus implementation; integration and tests verified locally.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from .locomotion_checkpoint import write_checkpoint_metadata
from .ppo_update_audit import parameter_sha256

MANIFEST_SCHEMA = "d1-budget-checkpoints-v1"


def _require_positive_int(value: object, name: str) -> int:
    """Return ``value`` iff it is a strictly positive, non-bool Python ``int``.

    No silent coercion: ``bool`` is rejected (``type is int`` excludes it) and
    non-int types raise ``TypeError`` rather than being cast.
    """

    if type(value) is not int:
        raise TypeError(f"{name} must be a non-bool Python int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be strictly positive, got {value}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_checkpoint_budgets(
    budgets,
    *,
    total_steps,
    workers,
    n_steps: int = 128,
) -> tuple[int, ...] | None:
    """Validate a checkpoint-budget declaration.

    ``None`` preserves legacy (no budget scheduling) and is returned as-is.
    Otherwise returns a tuple of strictly positive non-bool ints that is
    non-empty, strictly increasing, aligned to whole ``workers * n_steps``
    rollouts, with the last budget equal to ``total_steps``.
    """

    total_steps = _require_positive_int(total_steps, "total_steps")
    workers = _require_positive_int(workers, "workers")
    n_steps = _require_positive_int(n_steps, "n_steps")

    if budgets is None:
        return None

    budgets = tuple(budgets)
    if len(budgets) == 0:
        raise ValueError("checkpoint_budgets must be non-empty when provided")

    rollout = workers * n_steps
    previous = 0
    for budget in budgets:
        _require_positive_int(budget, "checkpoint budget")
        if budget <= previous:
            raise ValueError("checkpoint_budgets must be strictly increasing")
        if budget % rollout != 0:
            raise ValueError(
                f"checkpoint budget {budget} is not a whole multiple of the "
                f"rollout size {rollout} (workers*n_steps)"
            )
        previous = budget

    if budgets[-1] != total_steps:
        raise ValueError(
            f"the last checkpoint budget must equal total_steps={total_steps}, got {budgets[-1]}"
        )

    return budgets


class BudgetCheckpointWriter:
    """Save PPO checkpoints exactly at declared timestep budgets.

    The writer is bound to a single learner instance on first use and drives one
    PPO update per :meth:`execute_update` call. Progress and failures are written
    to ``manifest.json``; phase timings are appended to ``phases.jsonl``.
    ``complete`` covers scheduled saves, not the enclosing experiment or reload.
    """

    def __init__(
        self,
        directory,
        budgets,
        *,
        workers,
        n_steps,
        training_seed,
        reference_env,
        audit_directory,
    ):
        self._workers = _require_positive_int(workers, "workers")
        self._n_steps = _require_positive_int(n_steps, "n_steps")
        self._rollout = self._workers * self._n_steps

        budgets = tuple(budgets)
        if len(budgets) == 0:
            raise ValueError("checkpoint_budgets must be non-empty")
        self._budgets = validate_checkpoint_budgets(
            budgets, total_steps=budgets[-1], workers=workers, n_steps=n_steps
        )
        self._budget_set = set(budgets)

        directory = Path(directory)
        if ".." in directory.parts:
            raise ValueError("checkpoints directory must not contain parent traversal")
        if directory.is_symlink() or directory.exists():
            raise FileExistsError(f"checkpoints directory must not already exist: {directory}")
        parent = directory.parent
        if not parent.exists():
            raise FileNotFoundError(f"parent directory does not exist: {parent}")
        if not parent.is_dir():
            raise NotADirectoryError(f"parent is not a directory: {parent}")
        if parent.is_symlink():
            raise ValueError(f"parent directory must not be a symlink: {parent}")
        self._reject_symlink_ancestors(parent)

        self._directory = directory
        if type(training_seed) is not int or training_seed < 0:
            raise ValueError("training_seed must be a nonnegative non-bool Python int")
        self._training_seed = training_seed
        self._reference_env = reference_env
        self._audit_updates_path = Path(audit_directory) / "updates.jsonl"

        # Create the checkpoints directory exclusively; never overwrite.
        self._directory.mkdir(exist_ok=False)
        self._manifest_path = self._directory / "manifest.json"
        self._phases_path = self._directory / "phases.jsonl"

        self._learner = None
        self._last_num_timesteps = 0
        self._last_n_updates = 0
        self._updates_done = 0

        self._saved = []
        self._pending = set(budgets)
        self._complete = False
        self._failed = False
        self._failure = None
        self._learning_start = None

        self._write_manifest()

    # -- properties -------------------------------------------------------
    @property
    def budgets(self) -> tuple[int, ...]:
        return self._budgets

    @property
    def saved(self):
        return list(self._saved)

    @property
    def complete(self) -> bool:
        return self._complete

    @property
    def directory(self) -> Path:
        return self._directory

    # -- helpers ----------------------------------------------------------
    def begin_learning(self):
        """Start the actual continuous-learn clock after learner initialization."""
        if self._learning_start is not None:
            raise RuntimeError("continuous learning clock may only be started once")
        self._learning_start = perf_counter()
        return self._learning_start

    @staticmethod
    def _reject_symlink_ancestors(path: Path) -> None:
        for ancestor in [path, *path.parents]:
            if ancestor.is_symlink():
                raise ValueError(f"symlink ancestor is not allowed: {ancestor}")

    def _read_last_audit_row(self) -> dict:
        last_line = None
        with self._audit_updates_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    last_line = stripped
        if last_line is None:
            raise RuntimeError("audit updates.jsonl is empty after update")
        return json.loads(last_line)

    def _append_phase(
        self, phase: str, *, budget, start: float, end: float, wall_start: str
    ) -> None:
        entry = {
            "phase": phase,
            "budget": budget,
            "start_perf": start,
            "end_perf": end,
            "duration_s": end - start,
            "wall_start_utc": wall_start,
            "wall_end_utc": _utcnow_iso(),
        }
        with self._phases_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def _write_manifest(self) -> None:
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "budgets": list(self._budgets),
            "saved": self._saved,
            "complete": self._complete,
            "failed": self._failed,
            "failure": self._failure,
            "pending_budgets": sorted(self._pending),
        }
        tmp_path = self._manifest_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, self._manifest_path)

    def record_failure(self, exc: BaseException) -> None:
        """Retain unfinished state when learning, audit, or saving is interrupted."""
        self._failed = True
        self._complete = False
        self._failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "num_timesteps": getattr(self._learner, "num_timesteps", None),
            "updates_done": self._updates_done,
            "failed_at_utc": _utcnow_iso(),
        }
        try:
            self._write_manifest()
        except OSError:
            # Report storage failure without masking the original interruption.
            logging.getLogger(__name__).exception("could not persist budget checkpoint failure")

    def _save_checkpoint(
        self,
        learner,
        *,
        budget: int,
        after_update_index: int,
        n_updates: int,
        param_sha256_after: str,
    ) -> dict:
        wall_start = _utcnow_iso()
        save_start = perf_counter()

        folder = self._directory / f"budget{budget}"
        folder.mkdir(exist_ok=False)
        zip_path = folder / "checkpoint.zip"
        metadata_path = folder / "checkpoint.json"

        learner.save(zip_path)
        if zip_path.is_symlink() or not (zip_path.is_file() and zip_path.stat().st_size > 0):
            raise RuntimeError(f"checkpoint zip is missing or empty: {zip_path}")

        write_checkpoint_metadata(
            zip_path,
            self._reference_env,
            metadata_path,
            extra={
                "training_seed": self._training_seed,
                "num_timesteps": learner.num_timesteps,
                "checkpoint_budget": budget,
                "after_update_index": after_update_index,
                "param_sha256_after": param_sha256_after,
            },
        )
        if metadata_path.is_symlink() or not (
            metadata_path.is_file() and metadata_path.stat().st_size > 0
        ):
            raise RuntimeError(f"checkpoint metadata is missing or empty: {metadata_path}")

        zip_sha256 = _sha256_file(zip_path)
        metadata_sha256 = _sha256_file(metadata_path)

        save_end = perf_counter()
        record = {
            "budget": budget,
            "num_timesteps": learner.num_timesteps,
            "after_update_index": after_update_index,
            "n_updates": n_updates,
            "param_sha256_after": param_sha256_after,
            "zip_sha256": zip_sha256,
            "metadata_sha256": metadata_sha256,
            "model_path": f"budget{budget}/checkpoint.zip",
            "metadata_path": f"budget{budget}/checkpoint.json",
            "saved_at_utc": _utcnow_iso(),
            "elapsed_s": save_end - self._learning_start,
            "save_duration_s": save_end - save_start,
        }
        self._append_phase(
            "save", budget=budget, start=save_start, end=save_end, wall_start=wall_start
        )
        self._saved.append(record)
        self._pending.discard(budget)
        self._write_manifest()
        return record

    # -- public API -------------------------------------------------------
    def execute_update(self, learner, update_callable: Callable):
        """Save only after one complete PPO ``train()`` call returns.

        SB3 has already counted the collected rollout at entry. ``train()``
        changes optimizer counters and policy weights, not ``num_timesteps``.
        One call may contain several individual optimizer steps.
        """

        if self._failed:
            raise RuntimeError("failed budget writer cannot resume; use a new directory")
        if self._complete:
            raise RuntimeError("all budget checkpoints already saved; no extra update allowed")
        if self._learning_start is None:
            raise RuntimeError("begin_learning must precede the first PPO train call")
        if not callable(update_callable):
            raise TypeError("update_callable must be callable")
        if self._learner is None:
            self._learner = learner
        elif learner is not self._learner:
            raise ValueError("BudgetCheckpointWriter is bound to a different learner")

        before_timesteps = learner.num_timesteps
        before_n_updates = learner._n_updates
        if type(before_timesteps) is not int:
            raise TypeError("learner.num_timesteps must be an int")
        if type(before_n_updates) is not int:
            raise TypeError("learner._n_updates must be an int")
        if before_timesteps != self._last_num_timesteps + self._rollout:
            raise RuntimeError(
                f"unexpected num_timesteps before update: {before_timesteps} != "
                f"{self._last_num_timesteps + self._rollout}"
            )
        if before_n_updates != self._last_n_updates:
            raise RuntimeError(
                f"unexpected _n_updates before update: {before_n_updates} != {self._last_n_updates}"
            )
        if type(learner._audit_index) is not int or learner._audit_index != self._updates_done:
            raise RuntimeError("unexpected zero-based audit index before update")

        wall_start = _utcnow_iso()
        update_start = perf_counter()
        try:
            result = update_callable()
            update_end = perf_counter()
            self._append_phase(
                "update", budget=None, start=update_start, end=update_end, wall_start=wall_start
            )

            after_timesteps = learner.num_timesteps
            after_n_updates = learner._n_updates
            if type(after_timesteps) is not int or after_timesteps != before_timesteps:
                raise RuntimeError(
                    "PPO train() must not change the already collected rollout timestep count"
                )
            if type(after_n_updates) is not int or not after_n_updates > before_n_updates:
                raise RuntimeError("_n_updates must strictly increase after update")

            if (
                type(learner._audit_index) is not int
                or learner._audit_index != self._updates_done + 1
            ):
                raise RuntimeError("exactly one completed audit record is required per train()")
            after_update_index = learner._audit_index - 1
            audit_row = self._read_last_audit_row()
            if (
                type(audit_row["num_timesteps"]) is not int
                or audit_row["num_timesteps"] != after_timesteps
            ):
                raise RuntimeError("audit num_timesteps mismatch")
            if (
                type(audit_row["audit_index"]) is not int
                or audit_row["audit_index"] != after_update_index
            ):
                raise RuntimeError("audit_index mismatch")
            if type(audit_row["n_updates"]) is not int or audit_row["n_updates"] != after_n_updates:
                raise RuntimeError("audit n_updates mismatch")
            actual_hash = parameter_sha256(learner.policy)
            if audit_row["param_sha256_after"] != actual_hash:
                raise RuntimeError("param_sha256_after mismatch with audit row")

            self._updates_done += 1
            self._last_num_timesteps = after_timesteps
            self._last_n_updates = after_n_updates

            if after_timesteps in self._budget_set:
                self._save_checkpoint(
                    learner,
                    budget=after_timesteps,
                    after_update_index=after_update_index,
                    n_updates=after_n_updates,
                    param_sha256_after=actual_hash,
                )

            if not self._pending:
                self._complete = True
                self._write_manifest()

            return result
        except BaseException as exc:  # includes KeyboardInterrupt
            self.record_failure(exc)
            raise
