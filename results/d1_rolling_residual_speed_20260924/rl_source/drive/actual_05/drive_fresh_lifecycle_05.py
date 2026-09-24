"""Pure read-only guard for the actual never-reset D1 environment shape.

Construction initializes bookkeeping but intentionally does not create
``_steps``. The first reset creates it, even when its value is zero.
"""

from __future__ import annotations

import math
from typing import Any

_NONE_FIELDS = (
    "loop", "decision", "last_transition", "schedule",
    "_heading_context", "_reference_loop", "_reference_initial",
)


def assert_never_reset_inactive(inner: Any) -> dict[str, Any]:
    """Validate one actual fresh object's initialized, unprepared lifecycle.

    Uses instance attributes only; does not initialize/reset/prepare the object.
    The heading_decision property is checked separately because it is the
    public read-only view of the initialized ``_heading_context`` field.
    """
    attributes = vars(inner)
    if "_steps" in attributes or hasattr(inner, "_steps"):
        raise RuntimeError("fresh environment already entered reset lifecycle")
    for name in _NONE_FIELDS:
        if name not in attributes or attributes[name] is not None:
            raise RuntimeError(f"fresh environment field differs: {name}")
    if "_active" not in attributes or attributes["_active"] is not False:
        raise RuntimeError("fresh environment must be exactly inactive")
    if ("_episode_metadata" not in attributes
            or type(attributes["_episode_metadata"]) is not dict
            or attributes["_episode_metadata"]):
        raise RuntimeError("fresh environment already has episode metadata")
    if inner.heading_decision is not None:
        raise RuntimeError("fresh heading decision must be absent")
    if "plant" not in attributes or not hasattr(attributes["plant"], "data"):
        raise RuntimeError("fresh plant/data binding is absent")
    data = attributes["plant"].data
    if not hasattr(data, "time"):
        raise RuntimeError("fresh physical time is absent")
    try:
        physical_time = float(data.time)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("fresh physical time is malformed") from exc
    if not math.isfinite(physical_time) or physical_time != 0.0:
        raise RuntimeError("fresh physical time must be exactly zero")
    return {
        "schema": "d1-never-reset-inactive-lifecycle-v1",
        "steps_attribute_absent": True,
        "initialized_none_fields": list(_NONE_FIELDS),
        "exact_inactive": True,
        "episode_metadata_empty": True,
        "heading_decision_none": True,
        "physical_time_zero": True,
        "no_mutation_or_engine_call": True,
    }
