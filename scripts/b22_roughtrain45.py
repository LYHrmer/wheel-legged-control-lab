"""Shared pure definitions for the 45 rough-terrain emphasis training arm.

One arm continues the completed B22 policy and Adam state on the yaw-feedforward baseline with
the deficit-8 tracking reward, exactly like the 44 arms; only the training command curriculum
differs. ``rough_emphasis`` is the 44 ``yaw_emphasis`` table with two slots moved onto rough
terrain: slot 0 (flat, low speed, yaw) and slot 6 (flat, 1.6 m/s, straight) become rough, low speed,
so rough episodes rise from 1/8 to 3/8 of the source schedule while the number of yaw slots stays 4.

Nothing here touches a model or the simulator.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b22_yawtrain44 import (
    ALPHA,
    FEEDFORWARD,
    ORIGINAL_SLOTS,
    YAW_EMPHASIS_SLOTS,
    derived_verifier_source,
    keep_record_consistent,
)

ARMS = {"ffrough": {"alpha": ALPHA, "curriculum": "rough_emphasis"}}
ROUGH_EMPHASIS_SLOTS = (
    ("rough", (0.25, 0.4), (0.15, 0.3)),
    YAW_EMPHASIS_SLOTS[1],
    YAW_EMPHASIS_SLOTS[2],
    YAW_EMPHASIS_SLOTS[3],
    YAW_EMPHASIS_SLOTS[4],
    YAW_EMPHASIS_SLOTS[5],
    ("rough", (0.25, 0.4), None),
    YAW_EMPHASIS_SLOTS[7],
)
CURRICULUM_SLOTS = {
    "original": ORIGINAL_SLOTS,
    "yaw_emphasis": YAW_EMPHASIS_SLOTS,
    "rough_emphasis": ROUGH_EMPHASIS_SLOTS,
}


def slots_for(curriculum: str):
    if curriculum not in CURRICULUM_SLOTS:
        raise ValueError(f"unknown curriculum: {curriculum}")
    return CURRICULUM_SLOTS[curriculum]


def apply_curriculum(recipes, curriculum: str) -> None:
    """Install the arm's slot table into an executed recipes module."""
    if tuple(recipes._SLOTS) != ORIGINAL_SLOTS:
        raise RuntimeError("frozen recipes slot table differs from the expected original")
    recipes._SLOTS = slots_for(curriculum)


def rough_episode_fraction(slots) -> float:
    return sum(1 for terrain, _, _ in slots if terrain == "rough") / len(slots)


__all__ = (
    "ALPHA",
    "ARMS",
    "FEEDFORWARD",
    "apply_curriculum",
    "derived_verifier_source",
    "keep_record_consistent",
    "slots_for",
)
