"""Versioned, proprioceptive 82-value decision encoding shared by v3 tasks.

No plant, terrain query, or evaluation truth is accepted by this module. The
current baseline and actual controller memories come from the decision context,
not from an ambiguously named previous total-force diagnostic.
"""

from __future__ import annotations

import numpy as np

from .control_context import D1ForceBaseline, D1JointTargetBaseline
from .control_loop import D1Decision
from .model import JOINT_VELOCITY_LIMIT, NOMINAL_JOINT_POSITION

LOCOMOTION_OBSERVATION_SCHEMA = "d1-proprio-current-control82-v1"
LOCOMOTION_OBSERVATION_SIZE = 82
LEG_INDICES = np.asarray([index for index in range(16) if index % 4 != 3])
LEG_POSITION_SCALE = np.tile((0.8, 2.0, 1.0), 4)

# Half-open slices describe the actual flattened array; names carry units where
# appropriate. Baseline targets and controller memory have explicit kind/masks.
OBSERVATION_SLICES = {
    "com_velocity_body_mps": (0, 3),
    "angular_velocity_body_over4": (3, 6),
    "projected_gravity": (6, 9),
    "height_error_over_0p08m": (9, 10),
    "leg_position_error_normalized": (10, 22),
    "joint_velocity_over_limit": (22, 38),
    "command_vx_yaw_clearance_roll_pitch": (38, 43),
    "measurement_age_over_0p05s": (43, 44),
    "baseline_force_or_leg_and_wheel_targets": (44, 60),
    "baseline_kind_force_joint": (60, 62),
    "distance_error_yaw_integral_wheel_integrals": (62, 68),
    "controller_memory_present": (68, 74),
    "previous_applied_action_padded8": (74, 82),
}


def encode_d1_locomotion_observation(decision: D1Decision) -> np.ndarray:
    """Encode one current decision; repeated reads have no stateful side effects."""
    context, command = decision.context, decision.world_command
    state, proposal = context.state, context.proposal
    if context.action_size not in (2, 8):
        raise ValueError("the v3 observation schema supports exactly 2 or 8 actions")
    baseline = np.zeros(16)
    kind = np.zeros(2)
    if isinstance(proposal.baseline, D1ForceBaseline):
        force = proposal.baseline
        baseline[:3] = (
            force.longitudinal_force_n / 180.0,
            force.support_vertical_force_n / 600.0,
            force.vertical_feedforward_force_n / 500.0,
        )
        kind[0] = 1
    elif isinstance(proposal.baseline, D1JointTargetBaseline):
        target = proposal.baseline
        baseline[:12] = (
            target.nominal_joint_target_rad[LEG_INDICES] - NOMINAL_JOINT_POSITION[LEG_INDICES]
        ) / LEG_POSITION_SCALE
        baseline[12:] = target.nominal_wheel_speed_rad_s / 30.0
        kind[1] = 1
    else:
        raise TypeError("unknown baseline semantics")
    memory, present = np.zeros(6), np.zeros(6)
    stored = proposal.memory
    if stored.distance_m is not None:
        memory[0] = (stored.distance_reference_m - stored.distance_m) / 0.55
        present[0] = 1
    if stored.yaw_integral_nm is not None:
        memory[1] = stored.yaw_integral_nm / 4.0
        present[1] = 1
    if stored.wheel_integral_nm is not None:
        memory[2:] = stored.wheel_integral_nm / 4.0
        present[2:] = 1
    previous = np.zeros(8)
    previous[: context.action_size] = context.previous_normalized_action
    observation = np.concatenate(
        (
            state.base_linear_velocity_body,
            state.base_angular_velocity_body / 4.0,
            state.projected_gravity_body,
            [(state.base_position[2] - command.base_height_m) / 0.08],
            (state.joint_position[LEG_INDICES] - NOMINAL_JOINT_POSITION[LEG_INDICES])
            / LEG_POSITION_SCALE,
            state.joint_velocity / JOINT_VELOCITY_LIMIT,
            [
                command.forward_velocity_mps / 0.6,
                command.yaw_rate_rps / 0.5,
                (decision.motion_command.clearance_m - 0.455) / 0.08,
                command.roll_rad / 0.3,
                command.pitch_rad / 0.3,
            ],
            [state.age_s / 0.05],
            baseline,
            kind,
            memory,
            present,
            previous,
        )
    )
    if observation.shape != (LOCOMOTION_OBSERVATION_SIZE,) or not np.isfinite(observation).all():
        raise ValueError("invalid v3 observation")
    return np.clip(observation, -5.0, 5.0).astype(np.float32)
