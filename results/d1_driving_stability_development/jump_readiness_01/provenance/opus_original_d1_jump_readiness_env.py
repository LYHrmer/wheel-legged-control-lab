"""Thin height-readiness environment for the fixed stationary height profile.

The environment adds no controller, no gain, no thrust and no phase logic.  It
binds the frozen :func:`fixed_height_command` schedule into the already-accepted
raw command callback, records what the composed controller actually received and
produced on every executed tick, and owns its own diagnostic stream so the
parent's terminal close cannot drop the last record.
"""
from __future__ import annotations

import gzip
import json
from contextlib import ExitStack
from typing import Any, Dict

import numpy as np

from scripts.d1_jump_readiness import (
    JUMP_READINESS_CONDITIONS,
    JUMP_READINESS_SCHEMA,
    TERMINAL_PREPARED_TICK,
    fixed_height_command,
)
from scripts.d1_stop_turn_env import StopTurnEnv

READINESS_TASK_SCHEMA = "d1-heading-plane-height-readiness-zero-task-v1"
READINESS_PROBE_SCHEMA = "d1-height-profile-hop-readiness-probe-v1"
CONTROL_DT_S = 0.01
ZERO_ACTION = np.zeros(8, dtype=np.float32)
_TIME_TOL_S = 1.0e-9
_HEIGHT_TOL_M = 1.0e-12


def reject_foreign_checkpoint_metadata(metadata: Any) -> None:
    """Reject any old checkpoint metadata before deserialization is attempted."""
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be a mapping to be screened")
    schema = metadata.get("task_schema")
    if schema != READINESS_TASK_SCHEMA:
        raise ValueError(
            "this diagnostic loads no policy; refusing metadata with task_schema "
            f"{schema!r} before deserialization")
    raise ValueError("no policy or checkpoint load is permitted in this diagnostic")


def require_zero_action(action: Any) -> np.ndarray:
    """Accept only an exactly-zero float32 residual action of eight channels."""
    array = np.asarray(action)
    if array.dtype != np.float32 or array.shape != (8,):
        raise ValueError(
            f"residual action must be float32 with shape (8,), got {array.dtype!r} "
            f"{array.shape}")
    if not np.all(array == 0.0):
        raise ValueError("residual action must be exactly zero on every channel")
    return array


class JumpReadinessEnv(StopTurnEnv):
    """Fixed-schedule height command source with per-tick control diagnostics."""

    task_schema = READINESS_TASK_SCHEMA
    probe_schema = READINESS_PROBE_SCHEMA

    def __init__(self, *, diagnostic_output, condition: str, **kwargs):
        if condition not in JUMP_READINESS_CONDITIONS:
            raise ValueError(
                f"condition must be one of {JUMP_READINESS_CONDITIONS}, got {condition!r}")
        self.condition = condition
        self._prepared: Dict[int, str] = {}
        self.height_callback_count = 0
        self.parent_step_calls = 0
        self.executed_ticks: list[int] = []
        self.last_phase_label: str | None = None
        self._in_step = False
        super().__init__(diagnostic_output=diagnostic_output,
                         command_source=self._fixed_command, **kwargs)
        self._readiness_files = ExitStack()
        self._readiness_stream = self._readiness_files.enter_context(
            gzip.open(diagnostic_output / "height_readiness.jsonl.gz", "xt")  # noqa: SIM115
        )
        self._readiness_closed = False

    # -- fixed command callback ------------------------------------------
    def _fixed_command(self, time_s):
        tick = int(round(float(time_s) / CONTROL_DT_S))
        if abs(float(time_s) - tick * CONTROL_DT_S) > _TIME_TOL_S:
            raise RuntimeError(
                f"command callback time {time_s!r} is not an integer control tick")
        if tick < 0 or tick > TERMINAL_PREPARED_TICK:
            raise RuntimeError(f"prepared tick {tick} is outside 0..{TERMINAL_PREPARED_TICK}")
        if tick in self._prepared:
            raise RuntimeError(f"prepared tick {tick} was already consumed")
        command, label = fixed_height_command(tick, self.condition)
        self._prepared[tick] = label
        self.height_callback_count += 1
        self.last_phase_label = label
        return command

    def _build_heading_task_config(self):
        config = super()._build_heading_task_config()
        config["task_schema"] = self.task_schema
        config["height_readiness"] = {
            "probe_schema": self.probe_schema,
            "schedule_schema": JUMP_READINESS_SCHEMA,
            "condition": self.condition,
            "raw_forward_mps": 0.0,
            "raw_yaw_rps": 0.0,
            "residual_action": "zero8 float32 only",
            "policy_or_checkpoint_load": "rejected before deserialization",
            "display_phase_source": "fixed helper label (hold ticks 0..199 read 'settle')",
            "contract_phase_label": "proposed_cases displays the whole hold as 'hold'",
            "new_controller_or_gain": False,
        }
        return config

    def reset(self, *, seed=None, options=None):
        self._prepared = {}
        self.height_callback_count = 0
        self.parent_step_calls = 0
        self.executed_ticks = []
        self.last_phase_label = None
        return super().reset(seed=seed, options=options)

    # -- executed-tick diagnostics ---------------------------------------
    def _diagnostic_row(self, before, result_record) -> Dict[str, Any]:
        decision = before.decision
        state = decision.context.state
        expected_command, expected_label = fixed_height_command(before.tick, self.condition)
        world_height = float(decision.world_command.base_height_m)
        ground_height = float(decision.ground.height_m)
        raw_clearance = float(before.user_command.clearance_m)
        servo_clearance = float(before.servo_command.clearance_m)
        if abs(raw_clearance - expected_command.clearance_m) > _HEIGHT_TOL_M:
            raise RuntimeError(
                f"tick {before.tick} raw clearance {raw_clearance!r} differs from the fixed "
                f"schedule {expected_command.clearance_m!r}")
        if abs(servo_clearance - expected_command.clearance_m) > _HEIGHT_TOL_M:
            raise RuntimeError(
                f"tick {before.tick} servo clearance {servo_clearance!r} did not carry the "
                "fixed schedule height into the control loop")
        if abs((world_height - ground_height) - expected_command.clearance_m) > _HEIGHT_TOL_M:
            raise RuntimeError(
                f"tick {before.tick} world command height {world_height!r} over ground "
                f"{ground_height!r} differs from the fixed schedule height "
                f"{expected_command.clearance_m!r}")
        if float(before.user_command.forward_velocity_mps) != 0.0 or float(
                before.user_command.yaw_rate_rps) != 0.0:
            raise RuntimeError("raw forward and yaw must be exactly zero on every tick")

        def listed(value):
            return np.asarray(value, dtype=np.float64).tolist()

        row: Dict[str, Any] = {
            "probe_schema": self.probe_schema,
            "condition": self.condition,
            "tick": int(before.tick),
            "endpoint_tick": int(before.tick) + 1,
            "control_time_s": float(before.control_time_s),
            "display_phase": expected_label,
            "commanded_clearance_m": float(expected_command.clearance_m),
            "raw_clearance_m": raw_clearance,
            "servo_clearance_m": servo_clearance,
            "world_command_base_height_m": world_height,
            "ground_height_m": ground_height,
            "raw_forward_mps": float(before.user_command.forward_velocity_mps),
            "raw_yaw_rps": float(before.user_command.yaw_rate_rps),
            "heading_error_rad": float(before.heading_error_rad),
            "stop_latch_active": bool(getattr(
                getattr(self._controller.controller, "last_damping", None), "latched", False)),
            "authority_gate_active": bool(
                float(before.user_command.yaw_rate_rps) != 0.0),
            "joint_position_rad": listed(state.joint_position),
            "joint_velocity_rad_s": listed(state.joint_velocity),
        }
        if result_record is not None:
            row.update({
                "nominal_joint_target_rad": listed(result_record.nominal_joint_target_rad),
                "joint_target_rad": listed(result_record.joint_target_rad),
                "joint_target_rate_limited": listed(result_record.joint_target_rate_limited),
                "leg_extension_target_m": listed(result_record.leg_extension_target_m),
                "requested_extension_m": listed(result_record.requested_extension_m),
                "wheel_speed_target_rad_s": listed(result_record.wheel_speed_target_rad_s),
                "leg_pd_nm": listed(result_record.leg_pd_nm),
                "support_nm": listed(result_record.support_nm),
                "support_force_n": listed(result_record.support_force_n),
                "wheel_nm": listed(result_record.wheel_nm),
                "requested_torque_nm": listed(result_record.requested_torque_nm),
                "torque_nm": listed(result_record.torque_nm),
                "torque_limited": listed(result_record.torque_limited),
                "pi_memory_before": listed(
                    getattr(result_record.memory_before, "wheel_integral", ())),
                "pi_memory_after": listed(
                    getattr(result_record.memory_after, "wheel_integral", ())),
                "effective_yaw_request_rps": float(result_record.effective_yaw_request_rps),
            })
        return row

    def step(self, action):
        require_zero_action(action)
        if self._in_step:
            raise RuntimeError("readiness step must delegate to the parent exactly once")
        before = self.heading_decision
        if before is None:
            raise RuntimeError("no prepared heading decision is available for this tick")
        self._in_step = True
        try:
            result = super().step(action)
            self.parent_step_calls += 1
            self.executed_ticks.append(int(before.tick))
            row = self._diagnostic_row(before, getattr(
                self._controller.controller, "last_result", None))
            self._readiness_stream.write(json.dumps(row, allow_nan=False) + "\n")
            if result[2] or result[3]:
                self._readiness_files.close()
            return result
        finally:
            self._in_step = False

    def close(self):
        if getattr(self, "_readiness_closed", False):
            return
        self._readiness_closed = True
        try:
            self._readiness_files.close()
        finally:
            super().close()
