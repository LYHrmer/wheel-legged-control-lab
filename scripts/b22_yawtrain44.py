"""Shared pure definitions for the 44 yaw-feedforward training comparison.

Two arms continue the completed B22 policy and Adam state on the feedforward baseline with the
deficit-8 tracking reward of the 42 comparison; only the training command curriculum differs:

* ``ffcontrol``: the original arm-B curriculum.
* ``ffyaw``: the same curriculum with the slots 0, 1 and 4 (flat low speed, flat high speed,
  rough) also carrying the original three-segment yaw pattern, so about 20% instead of about 5%
  of the training control ticks have a non-zero yaw command.

Nothing here touches a model or the simulator.
"""

from __future__ import annotations

ALPHA = 8
FEEDFORWARD = 3.9
ARMS = {
    "ffcontrol": {"alpha": ALPHA, "curriculum": "original"},
    "ffyaw": {"alpha": ALPHA, "curriculum": "yaw_emphasis"},
}
# (terrain, speed rule, yaw-amplitude rule) in the frozen recipes22 slot order.
ORIGINAL_SLOTS = (
    ("flat", (0.4, 0.9), None),
    ("flat", (1.1, 1.6), None),
    ("flat", (0.6, 1.2), (0.15, 0.3)),
    ("bumps", (0.25, 0.4), None),
    ("rough", (0.25, 0.4), None),
    ("ramp", (0.35, 0.5), None),
    ("flat", 1.6, None),
    ("ramp", (0.35, 0.5), None),
)
YAW_EMPHASIS_SLOTS = (
    ("flat", (0.4, 0.9), (0.15, 0.3)),
    ("flat", (1.1, 1.6), (0.15, 0.3)),
    ORIGINAL_SLOTS[2],
    ORIGINAL_SLOTS[3],
    ("rough", (0.25, 0.4), (0.15, 0.3)),
    ORIGINAL_SLOTS[5],
    ORIGINAL_SLOTS[6],
    ORIGINAL_SLOTS[7],
)
CURRICULUM_SLOTS = {"original": ORIGINAL_SLOTS, "yaw_emphasis": YAW_EMPHASIS_SLOTS}


def slots_for(curriculum: str):
    if curriculum not in CURRICULUM_SLOTS:
        raise ValueError(f"unknown curriculum: {curriculum}")
    return CURRICULUM_SLOTS[curriculum]


def apply_curriculum(recipes, curriculum: str) -> None:
    """Install the arm's slot table into an executed recipes module (original is a no-op check)."""
    if tuple(recipes._SLOTS) != ORIGINAL_SLOTS:
        raise RuntimeError("frozen recipes slot table differs from the expected original")
    recipes._SLOTS = slots_for(curriculum)


def yaw_tick_fraction(slots, episodes: int = 72, episode_ticks: int = 1000) -> float:
    """Fraction of control ticks with a non-zero yaw command over source episodes 8..8+episodes."""
    yaw_slots = sum(1 for _, _, yaw in slots if yaw)
    cycles = episodes / len(slots)
    return yaw_slots * cycles * 400 / (episodes * episode_ticks)


def derived_verifier_source(source: str, feedforward: float) -> str:
    """Single audited edit of the frozen C18 verifier: the nominal unclipped yaw law."""
    old = "unclipped_yaw = servo_yaw_rps + 4.0*(servo_yaw_rps-body_yaw_rate)"
    if source.count(old) != 1:
        raise RuntimeError("frozen verifier no longer contains the expected single yaw-law line")
    return source.replace(
        old, f"unclipped_yaw = {feedforward!r}*servo_yaw_rps + 4.0*(servo_yaw_rps-body_yaw_rate)"
    )


def keep_record_consistent(controller, feedforward: float) -> None:
    """Restate the two yaw fields of the frozen C18 record, which knows nothing about the feedforward."""
    compute = controller.compute

    def wrapped(command, state, ground, action):
        torque = compute(command, state, ground, action)
        cmd = float(command.yaw_rate_rps)
        value = feedforward * cmd + controller._nominal.yaw_feedback_gain * (
            cmd - float(state.base_angular_velocity_body[2])
        )
        controller.last_record["nominal_unclipped_yaw_rps"] = value
        controller.last_record["nominal_unlimited_yaw_rps"] = value
        controller.last_record["command_feedforward_gain"] = feedforward
        return torque

    controller.compute = wrapped
