"""Shared definitions and audited source derivations for the 48 budget experiment.

The frozen training protocol pins 32768 controls / 32 rollouts inside ``learning21`` and
``read_training22``. This revision doubles the budget (65536 controls, 64 rollouts) and derives
those two modules by exact, counted string replacements so any upstream drift fails loudly.
The reward stays at the 44 ``ffyaw`` shape (beta 0); only the budget and the seed change.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b22_yawtrain44 import (
    ALPHA,
    FEEDFORWARD,
    YAW_EMPHASIS_SLOTS,
    apply_curriculum,
    derived_verifier_source,
    keep_record_consistent,
)

CONTROLS = 65536
ROLLOUTS = 64
N_STEPS = 1024
BATCH_SIZE = 256
N_EPOCHS_MAX = 4
CURRICULUM = "yaw_emphasis"
SEEDS = (201031, 201041)
ARMS = {
    f"ffyaw_b2_s{seed % 1000:02d}": {
        "alpha": ALPHA,
        "beta": 0,
        "curriculum": CURRICULUM,
        "seed": seed,
        "controls": CONTROLS,
        "rollouts": ROLLOUTS,
    }
    for seed in SEEDS
}

# (old, new, expected count) — every replacement must occur exactly this many times.
LEARNING_REPLACEMENTS = (
    (
        'require(type(max_steps) is int and max_steps == 512, "arm ceiling must be 512 steps")',
        'require(type(max_steps) is int and max_steps == 1024, "arm ceiling must be 1024 steps")',
        1,
    ),
    (
        'actual["num_timesteps"] != 32768 or actual["train_calls"] != 32',
        'actual["num_timesteps"] != 65536 or actual["train_calls"] != 64',
        1,
    ),
    (
        'or actual["rollouts"] != 32 or actual["transitions"] != 32768',
        'or actual["rollouts"] != 64 or actual["transitions"] != 65536',
        1,
    ),
    (
        'violations.append("32 rollout/train-call and 32768 control closure differs")',
        'violations.append("64 rollout/train-call and 65536 control closure differs")',
        1,
    ),
    ('if (not 32 <= actual["epochs"] <= 128', 'if (not 64 <= actual["epochs"] <= 256', 1),
    (
        'or not 32 <= actual["optimizer_steps"] <= 512',
        'or not 64 <= actual["optimizer_steps"] <= 1024',
        1,
    ),
    (
        'or not actual["optimizer_steps"] <= actual["evaluated_minibatches"] <= 512',
        'or not actual["optimizer_steps"] <= actual["evaluated_minibatches"] <= 1024',
        1,
    ),
    ("BudgetSpec(32768, 1024, 256, 4).as_dict()", "BudgetSpec(65536, 1024, 256, 4).as_dict()", 1),
    (
        "require(len(clip_audit.train_records) == 32",
        "require(len(clip_audit.train_records) == 64",
        1,
    ),
    (
        "and 32 <= len(clip_audit.events) <= clip_audit.max_steps",
        "and 64 <= len(clip_audit.events) <= clip_audit.max_steps",
        1,
    ),
)

READER_REPLACEMENTS = (
    ('"train":32,"forward":N', '"train":64,"forward":N', 1),
    ("for block_index in range(32):", "for block_index in range(64):", 1),
    ("N = 32768", "N = 65536", 1),
    (
        'and len(trains) == len(learned["updates"]) == 32',
        'and len(trains) == len(learned["updates"]) == 64',
        1,
    ),
    (
        "and 32 <= len(events) <= len(batches) <= 512",
        "and 64 <= len(events) <= len(batches) <= 1024",
        1,
    ),
    (
        'and actual["num_timesteps"] == actual["transitions"] == 32768',
        'and actual["num_timesteps"] == actual["transitions"] == 65536',
        1,
    ),
    (
        'and actual["rollouts"] == actual["train_calls"] == 32',
        'and actual["rollouts"] == actual["train_calls"] == 64',
        1,
    ),
    (
        '== counts["predict_values"]["returned"] == 32+truncated_episodes',
        '== counts["predict_values"]["returned"] == 64+truncated_episodes',
        1,
    ),
    (
        'and counts["predict_values"]["rows_returned"] == 32+truncated_episodes',
        'and counts["predict_values"]["rows_returned"] == 64+truncated_episodes',
        1,
    ),
    (
        "== N+256*len(batches)+32+truncated_episodes",
        "== N+256*len(batches)+64+truncated_episodes",
        1,
    ),
    (
        'and len(blocks["numeric_blocks"]) == len(blocks["gaussian_blocks"]) == 32',
        'and len(blocks["numeric_blocks"]) == len(blocks["gaussian_blocks"]) == 64',
        1,
    ),
)


def derive(source: str, replacements, label: str) -> tuple[str, dict]:
    derived = source
    applied = []
    for old, new, expected in replacements:
        found = derived.count(old)
        if found != expected:
            raise RuntimeError(
                f"{label}: expected {expected} occurrence(s) of {old!r}, found {found}"
            )
        derived = derived.replace(old, new)
        applied.append({"old": old, "new": new, "count": found})
    ast.parse(derived)  # the derived module must still be valid Python
    return derived, {
        "label": label,
        "replacements": applied,
        "source_sha256": __import__("hashlib").sha256(source.encode()).hexdigest(),
        "derived_sha256": __import__("hashlib").sha256(derived.encode()).hexdigest(),
    }


def derive_learning(source: str) -> tuple[str, dict]:
    return derive(source, LEARNING_REPLACEMENTS, "learning21-budget48")


def derive_reader(source: str) -> tuple[str, dict]:
    return derive(source, READER_REPLACEMENTS, "read_training22-budget48")


__all__ = (
    "ALPHA",
    "ARMS",
    "CONTROLS",
    "CURRICULUM",
    "FEEDFORWARD",
    "N_EPOCHS_MAX",
    "N_STEPS",
    "ROLLOUTS",
    "SEEDS",
    "YAW_EMPHASIS_SLOTS",
    "apply_curriculum",
    "derive_learning",
    "derive_reader",
    "derived_verifier_source",
    "keep_record_consistent",
)
