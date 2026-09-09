"""Short-horizon locomotion reward rates, separate from task quality gates.

Ordinary terms are rates integrated by control_dt. The terminal penalty is a
one-off event. No foot-speed penalty is used: wheel rotation is valid locomotion,
not automatically a slipping stance foot. Mechanical power is not battery power.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Real

import numpy as np

from .actuator_channel import ActuatorTrace
from .control_loop import D1ControlTransition
from .control_primitives import terrain_normal_to_rpy
from .training_terrain import TrainingGroundReference


@dataclass(frozen=True, slots=True)
class D1LocomotionRewardConfig:
    velocity_sigma_mps: float = 0.2
    yaw_sigma_rps: float = 0.2
    height_scale_m: float = 0.05
    attitude_scale_rad: float = 0.3
    mechanical_power_scale_w: float = 200.0
    action_change_scale: float = 0.1
    velocity_weight: float = 1.0
    yaw_weight: float = 1.0
    height_weight: float = 1.0
    attitude_weight: float = 0.2
    power_weight: float = 0.02
    action_change_weight: float = 0.01
    termination_cost: float = 2.0

    def __post_init__(self):
        for item in fields(self):
            name, value = item.name, getattr(self, item.name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
            if ("scale" in name or "sigma" in name) and value == 0:
                raise ValueError(f"{name} must be positive")


def d1_locomotion_reward_terms(
    transition: D1ControlTransition,
    ground_truth: TrainingGroundReference,
    actuator_traces: tuple[ActuatorTrace, ...],
    *,
    terminated: bool,
    config: D1LocomotionRewardConfig | None = None,
) -> dict[str, float]:
    """Return actual reward contributions, whose sum is the scalar reward.

    Action change is normalized to a change per 10 ms control tick, so changing
    control rate preserves its rate interpretation. It penalizes commanded
    policy changes, not an unobserved motor response or a torque derivative.
    """
    config = D1LocomotionRewardConfig() if config is None else config
    decision, truth = transition.decision, transition.truth
    dt = transition.receipt.end_time_s - transition.receipt.start_time_s
    if not actuator_traces:
        raise ValueError("reward power needs actual physical actuator traces")
    target = decision.motion_command
    vx_error = truth.base_linear_velocity_body[0] - target.forward_velocity_mps
    yaw_error = truth.base_angular_velocity_body[2] - target.yaw_rate_rps
    height_error = truth.base_position[2] - ground_truth.height_m - target.clearance_m
    # Correct endpoint terrain orientation; never confuse pre-step reward
    # targets with post-step height when moving across a sloped surface.
    roll_target, pitch_target = terrain_normal_to_rpy(
        -np.tan(ground_truth.pitch_rad), np.tan(ground_truth.roll_rad), float(truth.base_rpy[2])
    )
    attitude_error = truth.base_rpy[:2] - np.asarray((roll_target, pitch_target))
    power_w = float(
        np.mean(
            [
                np.sum(np.abs(trace.applied_nm * trace.joint_velocity_rps))
                for trace in actuator_traces
            ]
        )
    )
    change = (
        transition.receipt.normalized_action - decision.context.previous_normalized_action
    ) * (0.01 / dt)
    return {
        "tracking_velocity": dt
        * config.velocity_weight
        * float(np.exp(-((vx_error / config.velocity_sigma_mps) ** 2))),
        "tracking_yaw": dt
        * config.yaw_weight
        * float(np.exp(-((yaw_error / config.yaw_sigma_rps) ** 2))),
        "height": -dt * config.height_weight * float((height_error / config.height_scale_m) ** 2),
        "attitude": -dt
        * config.attitude_weight
        * float(np.sum((attitude_error / config.attitude_scale_rad) ** 2)),
        "mechanical_power": -dt * config.power_weight * power_w / config.mechanical_power_scale_w,
        "action_change": -dt
        * config.action_change_weight
        * float(np.mean((change / config.action_change_scale) ** 2)),
        "termination": -config.termination_cost if terminated else 0.0,
    }
