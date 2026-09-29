"""Fresh course/99D/16D Gym task; no old Heading or RollingPhysical inheritance.

Construction and reset are separate. Only the real factory creates a MuJoCo
model, inside the root-owned counted construction phase. The optional injected
factory is restricted to pure ``test_actor`` seam tests and must be rejected by
every production worker. Importing this module never constructs a model.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from full_drive_command_08 import FullDriveCommand, QualifiedCommandCaps
from full_drive_observation_08 import (
    OBSERVATION_SCHEMA,
    OBSERVATION_SIZE,
    ObservationPacket,
    encode_full_drive_observation,
    scan_compiled_course_ground,
)
from full_drive_servo_08 import FullDriveCommandServo, ServoReceipt

from wheel_legged_control.d1.control_primitives import terrain_normal_to_rpy

TASK_SCHEMA = "d1-course-pilot-rl16-v1"
REWARD_SCHEMA = "d1-course-pilot-bodycom-yaw-terrain-action-torque-v1"
SAFE_X_M = 11.0
SAFE_Y_M = 6.0
LEG_INDICES = tuple(index for index in range(16) if index % 4 != 3)
CommandSource = Callable[[int, float], FullDriveCommand]


def _spawn(value: Any) -> tuple[float, float, float]:
    raw = np.asarray(value)
    if raw.shape != (3,) or raw.dtype.kind not in "fiu":
        raise TypeError("course spawn must be a real xyz vector")
    point = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(point).all() or abs(point[0]) >= SAFE_X_M or abs(point[1]) >= SAFE_Y_M:
        raise ValueError("course spawn must be finite and inside the safety map")
    return tuple(float(item) for item in point)


class FullDriveCourseEnv(gym.Env):
    """One actual course plant, one control owner, one 16D policy action."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}
    task_schema = TASK_SCHEMA
    observation_schema = OBSERVATION_SCHEMA
    reward_schema = REWARD_SCHEMA

    def __init__(
        self, *, caps: QualifiedCommandCaps, command_source: CommandSource,
        spawn_position_m: tuple[float, float, float] = (-8.0, -4.7, 0.455),
        max_steps: int = 1600, mode: str = "train",
        enable_stationary_action_probe: bool = False,
        pure_components_factory: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(caps, QualifiedCommandCaps):
            raise TypeError("full-drive task requires qualified command caps")
        if not callable(command_source):
            raise TypeError("full-drive task requires an explicit command source")
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("max_steps must be a positive control-tick count")
        if mode not in ("test_actor", "train", "eval", "manual"):
            raise ValueError("unknown full-drive task mode")
        if pure_components_factory is not None and mode != "test_actor":
            raise ValueError("pure component injection is forbidden in production modes")
        self.pure_test_components = pure_components_factory is not None
        self.mode, self.caps = mode, caps
        self.command_source = command_source
        self.spawn_position_m = _spawn(spawn_position_m)
        self.max_steps = max_steps
        self._servo = FullDriveCommandServo(caps)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (16,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-5.0, 5.0, (OBSERVATION_SIZE,), dtype=np.float32)
        if pure_components_factory is None:
            from course_plant_08 import CoursePlant
            from full_drive_controller_08 import FullDriveControllerAdapter
            from full_drive_loop_08 import FullDriveLoop

            from wheel_legged_control.d1.actuator_channel import (
                ActuatorChannel,
                ActuatorChannelConfig,
            )
            from wheel_legged_control.d1.model import JOINT_TORQUE_LIMIT
            from wheel_legged_control.d1.state_provider import (
                D1StateProviderConfig,
                build_d1_state_provider,
            )

            channel = ActuatorChannel(ActuatorChannelConfig(
                torque_limit_nm=tuple(float(value) for value in JOINT_TORQUE_LIMIT),
            ))
            plant = CoursePlant(actuator_channel=channel)
            controller = FullDriveControllerAdapter(
                plant, mode=mode,
                enable_stationary_action_probe=enable_stationary_action_probe,
            )
            provider = build_d1_state_provider(
                plant, D1StateProviderConfig("oracle"),
                oracle_ground_query=plant.locomotion_ground_reference,
            )
            loop = FullDriveLoop(plant, provider, controller, caps=caps)
            self.plant, self.controller, self.loop = plant, controller, loop
            self.ground_map = plant.ground_map
            if type(plant) is not CoursePlant:
                raise RuntimeError("production task did not construct the exact CoursePlant")
        else:
            parts = pure_components_factory()
            if not isinstance(parts, dict) or set(parts) != {"plant", "controller", "loop", "ground_map"}:
                raise TypeError("pure test factory must provide plant/controller/loop/ground_map")
            self.plant, self.controller = parts["plant"], parts["controller"]
            self.loop, self.ground_map = parts["loop"], parts["ground_map"]

        # A fresh, never-reset env has no physical tick or prepared decision.
        # This is explicit so transfer/test guards cannot mistake absent
        # ``_steps`` for zero, the lifecycle bug found in the older 04 worker.
        self._steps: int | None = None
        self._active = False
        self._episode_index = -1
        self.decision = None
        self.last_transition = None
        self._last_observation: np.ndarray | None = None
        self._last_packet: ObservationPacket | None = None
        self._current_servo_receipt: ServoReceipt | None = None
        self._episode_metadata: dict[str, Any] = {}
        self._native_interval_reader: Callable[..., dict[str, Any]] | None = None
        self._native_interval_beginner: Callable[..., None] | None = None

    @property
    def episode_metadata(self) -> dict[str, Any]:
        return deepcopy(self._episode_metadata)

    def set_native_interval_reader(
        self, reader: Callable[..., dict[str, Any]], *, begin_interval: Callable[..., None],
    ) -> None:
        """Bind both sides of the audited five-substep interval atomically."""
        if (not callable(reader) or not callable(begin_interval)
                or self._native_interval_reader is not None):
            raise ValueError("native interval begin/read must be a one-time callable binding")
        if self._steps not in (None, 0) or self.last_transition is not None:
            raise RuntimeError("native interval reader must be installed before stepping")
        self._native_interval_reader = reader
        self._native_interval_beginner = begin_interval

    def _prepare(self, tick: int) -> np.ndarray:
        raw = self.command_source(tick, tick * self.plant.control_dt)
        if not isinstance(raw, FullDriveCommand):
            raise TypeError("command source must publish FullDriveCommand")
        servo = self._servo.advance(tick, raw)
        decision = self.loop.prepare(servo.applied, raw_operator_command=raw)
        state = decision.context.state
        scan = scan_compiled_course_ground(
            self.ground_map, x_m=float(state.base_position[0]),
            y_m=float(state.base_position[1]), yaw_rad=float(state.base_rpy[2]),
        )
        packet = encode_full_drive_observation(decision, scan, jump_phase=0)
        self.decision, self._last_packet = decision, packet
        self._current_servo_receipt = servo
        self._last_observation = packet.values.copy()
        return packet.values.copy()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        options = {} if options is None else dict(options)
        if set(options) - {"command_source", "spawn_position_m"}:
            raise ValueError("reset options support only command_source and spawn_position_m")
        source = options.get("command_source", self.command_source)
        if not callable(source):
            raise TypeError("next episode command source must be callable")
        spawn = _spawn(options.get("spawn_position_m", self.spawn_position_m))
        super().reset(seed=seed)
        self._active = False
        self._steps = 0
        self._episode_index += 1
        self.command_source, self.spawn_position_m = source, spawn
        self._servo.reset()
        measurement_seed = int(self.np_random.integers(0, 2**31))
        state = self.loop.reset(seed=measurement_seed, base_position=spawn)
        if state.sequence != 0 or not math.isclose(state.control_time_s, 0.0, abs_tol=1e-12):
            raise RuntimeError("reset did not publish the initial control tick")
        self.decision = self.last_transition = None
        self._last_observation = self._last_packet = self._current_servo_receipt = None
        self._episode_metadata = {
            "task_schema": TASK_SCHEMA, "reward_schema": REWARD_SCHEMA,
            "observation_schema": OBSERVATION_SCHEMA,
            "action_schema": self.controller.action_schema,
            "controller_schema": self.controller.control_schema,
            "control_loop_schema": self.loop.schema,
            "mode": self.mode, "pure_test_components": self.pure_test_components,
            "episode_index": self._episode_index,
            "measurement_seed": measurement_seed,
            "spawn_position_m": spawn,
            "max_steps": self.max_steps,
            "control_dt_s": self.plant.control_dt,
            "normal_dt_s": float(self.plant.model.opt.timestep)
            if not self.pure_test_components else None,
            "ground_source": "compiled-course-primitive-oracle-not-deployable-sensor",
            "qualified_caps": asdict(self.caps),
            "policy_action_size": 16,
            "observation_size": OBSERVATION_SIZE,
        }
        try:
            observation = self._prepare(0)
        except Exception:
            self._active = False
            raise
        self._active = True
        return observation, {"episode_metadata": self.episode_metadata}

    def _endpoint(self, transition, native_interval: dict[str, Any]
                  ) -> tuple[dict[str, float], str | None]:
        truth = transition.truth
        hit = self.plant.course_ground_hit(*truth.base_position[:2])
        reference_roll, reference_pitch = terrain_normal_to_rpy(
            -math.tan(hit.pitch_rad), math.tan(hit.roll_rad), float(truth.base_rpy[2]),
        )
        relative_roll = float(truth.base_rpy[0] - reference_roll)
        relative_pitch = float(truth.base_rpy[1] - reference_pitch)
        clearance = float(truth.base_position[2] - hit.height_m)
        outside = bool(abs(truth.base_position[0]) > SAFE_X_M
                       or abs(truth.base_position[1]) > SAFE_Y_M)
        undesired = int(self.plant.undesired_ground_contacts)
        native_nonwheel = native_interval["nonwheel_contact_count"]
        reason = (
            "nonwheel_ground_contact" if native_nonwheel > 0 or undesired > 0 else
            "course_map_boundary" if outside else
            "fall_or_low_clearance" if (
                clearance < 0.28 or max(abs(relative_roll), abs(relative_pitch)) > math.radians(20)
            ) else None
        )
        metrics = {
            "control_time_s": float(truth.control_time_s),
            "x_m": float(truth.base_position[0]),
            "y_m": float(truth.base_position[1]),
            "ground_height_m": hit.height_m,
            "ground_geom_id": hit.geom_id,
            "clearance_m": clearance,
            "body_com_vx_mps": float(truth.base_linear_velocity_body[0]),
            "body_com_vy_mps": float(truth.base_linear_velocity_body[1]),
            "body_yaw_rate_rps": float(truth.base_angular_velocity_body[2]),
            "relative_roll_rad": relative_roll,
            "relative_pitch_rad": relative_pitch,
            "absolute_roll_rad": float(truth.base_rpy[0]),
            "absolute_pitch_rad": float(truth.base_rpy[1]),
            "undesired_ground_contacts": undesired,
            "native_interval_nonwheel_contacts": native_nonwheel,
        }
        if not np.isfinite(tuple(value for value in metrics.values())).all():
            raise RuntimeError("post-step metrics are nonfinite")
        if max(abs(truth.base_rpy[0]), abs(truth.base_rpy[1])) > math.radians(45):
            raise RuntimeError("post-step attitude exceeded the hard safety bound")
        return metrics, reason

    def _reward(self, transition, metrics: dict[str, float], terminated: bool) -> dict[str, float]:
        command = transition.decision.motion_command
        traces = self.plant.last_control_interval_actuator_traces
        if len(traces) != self.plant.physics_steps:
            raise RuntimeError("reward requires every actual native actuator trace")
        limits = np.asarray(self.plant.actuator_torque_limit_nm)
        actual_torque_cost = float(np.mean([
            np.mean((trace.applied_nm / limits) ** 2) for trace in traces
        ]))
        qdot = transition.truth.joint_velocity[list(LEG_INDICES)]
        change = (transition.receipt.normalized_action
                  - transition.decision.context.previous_normalized_action)
        height_error = metrics["clearance_m"] - command.clearance_m
        roll, pitch = metrics["relative_roll_rad"], metrics["relative_pitch_rad"]
        terms = {
            "tracking_vx": 2.0 * math.exp(-((metrics["body_com_vx_mps"]
                                             - command.forward_velocity_mps) / 0.25) ** 2),
            "tracking_vy": math.exp(-((metrics["body_com_vy_mps"]
                                       - command.lateral_velocity_mps) / 0.12) ** 2),
            "tracking_yaw": math.exp(-((metrics["body_yaw_rate_rps"]
                                        - command.yaw_rate_rps) / 0.4) ** 2),
            "height": -0.2 * (height_error / 0.08) ** 2,
            "attitude": -0.2 * ((roll / 0.2) ** 2 + (pitch / 0.2) ** 2),
            "actual_torque": -0.02 * actual_torque_cost,
            "action_change": -0.01 * float(np.mean(change**2)),
            "leg_speed": -0.005 * float(np.mean((qdot / 20.0) ** 2)),
            "termination": -10.0 if terminated else 0.0,
        }
        if not np.isfinite(tuple(terms.values())).all():
            raise RuntimeError("new RL reward terms are nonfinite")
        return terms

    def step(self, policy_action):
        if not self._active or self._steps is None or self.decision is None:
            raise RuntimeError("reset must precede each full-drive control step")
        consumed_servo = self._current_servo_receipt
        consumed_packet = self._last_packet
        if consumed_servo is None or consumed_packet is None or self._last_observation is None:
            raise RuntimeError("prepared command and observation are incomplete")
        reader = self._native_interval_reader
        begin = self._native_interval_beginner
        if reader is None or begin is None:
            raise RuntimeError("audited five-substep interval begin/read must be bound before physics")
        try:
            begin(self.plant, start_time_s=self.decision.context.state.control_time_s)
            transition = self.loop.step(policy_action)
            self.last_transition = transition
            self._steps += 1
            interval = reader(self.plant, end_time_s=transition.state.control_time_s)
            interval_values = (
                interval.get("start_time_s"), interval.get("end_time_s"),
                interval.get("max_abs_roll_deg"), interval.get("max_abs_pitch_deg"),
            ) if isinstance(interval, dict) else ()
            if (not isinstance(interval, dict)
                    or type(interval.get("native_returns")) is not int
                    or interval["native_returns"] != self.plant.physics_steps
                    or type(interval.get("nonwheel_contact_count")) is not int
                    or interval["nonwheel_contact_count"] < 0
                    or any(type(value) not in (int, float) or not math.isfinite(value)
                           for value in interval_values)
                    or interval["max_abs_roll_deg"] < 0
                    or interval["max_abs_pitch_deg"] < 0
                    or not math.isclose(float(interval["start_time_s"]),
                                        transition.decision.context.state.control_time_s,
                                        rel_tol=0.0, abs_tol=1e-10)
                    or not math.isclose(float(interval["end_time_s"]),
                                        transition.state.control_time_s,
                                        rel_tol=0.0, abs_tol=1e-10)):
                raise RuntimeError("native interval summary differs from the completed five-substep tick")
            if (float(interval["max_abs_roll_deg"]) > 45.0
                    or float(interval["max_abs_pitch_deg"]) > 45.0):
                raise RuntimeError("a native substep exceeded the hard attitude safety bound")
            metrics, reason = self._endpoint(transition, interval)
            terminated = reason is not None
            truncated = self._steps >= self.max_steps and not terminated
            terms = self._reward(transition, metrics, terminated)
            info = {
                "task_schema": TASK_SCHEMA,
                "consumed_command": asdict(transition.decision.motion_command),
                "raw_operator_command": asdict(transition.decision.raw_operator_command),
                "servo_receipt": asdict(consumed_servo),
                "observation_clip_count": consumed_packet.clipped_count,
                "observation_clip_mask": consumed_packet.clipped_mask,
                "scan_center_geom_id": consumed_packet.scan_center_geom_id,
                "scan_geom_ids": consumed_packet.scan_geom_ids,
                "policy_input_action": transition.policy_input_action,
                "policy_clipped_action": transition.policy_clipped_action,
                "applied_action": transition.receipt.normalized_action,
                "controller_record": self.controller.last_record,
                "native_interval_summary": interval,
                "reward_terms": terms,
                "metrics": metrics,
                "terminal_reason": reason if terminated else "time_limit" if truncated else None,
                "completed_control_intervals": self._steps,
            }
            if terminated or truncated:
                # Gym/SB3 may bootstrap a time limit, so its observation must
                # describe the actual post-step state. Reuse the consumed
                # command with committed I/action; do not request a future
                # command, advance servo, or create an executable preview.
                terminal = self.loop.terminal_observation_decision(
                    transition.decision, transition,
                )
                post_state = terminal.context.state
                terminal_scan = scan_compiled_course_ground(
                    self.ground_map,
                    x_m=float(post_state.base_position[0]),
                    y_m=float(post_state.base_position[1]),
                    yaw_rad=float(post_state.base_rpy[2]),
                )
                observation = encode_full_drive_observation(
                    terminal, terminal_scan, jump_phase=0,
                ).values.copy()
                info["terminal_observation_source"] = (
                    "actual_poststep_state_with_consumed_command_observation_only"
                )
                info["TimeLimit.truncated"] = bool(truncated)
            else:
                observation = self._prepare(self._steps)
        except Exception:
            # A partly completed physical interval is never retried.
            self._active = False
            raise
        self._active = not (terminated or truncated)
        return observation, float(sum(terms.values())), terminated, truncated, info

    def close(self) -> None:
        self._active = False
