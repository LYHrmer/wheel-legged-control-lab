"""Shared pure definitions for the 47 seed-replication pair.

Two arms, same curriculum, same parent policy and Adam, same feedforward baseline, same budget;
the only difference between arms is the reward shape (beta 0 versus beta 8). Both arms use the
same new PPO/environment seed 201021, while ``command_seed`` and ``measurement_seed`` stay at the
frozen values so the curriculum and measurement-noise stream are identical to seeds 201011.

The actual PPO seed reaches training through ``spec['training']['ppo_seed_by_stage'][1]``; both
the worker and the independent reader override that entry from this table.
"""

from __future__ import annotations

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

SEED = 201021
CURRICULUM = "yaw_emphasis"
ARMS = {
    "ffyaw_s21": {"alpha": ALPHA, "beta": 0, "curriculum": CURRICULUM, "seed": SEED},
    "ffsse_s21": {"alpha": ALPHA, "beta": 8, "curriculum": CURRICULUM, "seed": SEED},
}

__all__ = (
    "ALPHA",
    "ARMS",
    "CURRICULUM",
    "FEEDFORWARD",
    "SEED",
    "YAW_EMPHASIS_SLOTS",
    "apply_curriculum",
    "derived_verifier_source",
    "keep_record_consistent",
)
