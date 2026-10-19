"""Shared pure definitions for the 46 logarithmic tracking-shape training arm.

One arm continues the completed B22 policy and Adam state on the yaw-feedforward baseline with
the 44 yaw-emphasis curriculum and the deficit-8 tracking reward; the only change is the shape of
the two tracking terms (``b22_reward46``). Nothing here touches a model or the simulator.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b22_yawtrain44 import (  # noqa: E402
    ALPHA,
    FEEDFORWARD,
    ORIGINAL_SLOTS,
    YAW_EMPHASIS_SLOTS,
    apply_curriculum,
    derived_verifier_source,
    keep_record_consistent,
)

BETA = 8
CURRICULUM = "yaw_emphasis"
ARMS = {"ffsse": {"alpha": ALPHA, "beta": BETA, "curriculum": CURRICULUM}}

__all__ = (
    "ALPHA",
    "ARMS",
    "BETA",
    "CURRICULUM",
    "FEEDFORWARD",
    "ORIGINAL_SLOTS",
    "YAW_EMPHASIS_SLOTS",
    "apply_curriculum",
    "derived_verifier_source",
    "keep_record_consistent",
)
