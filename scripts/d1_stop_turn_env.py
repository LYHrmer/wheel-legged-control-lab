"""Independent native-plane environment for the fixed stop/turn composition."""
from __future__ import annotations

import gzip
import json
from contextlib import ExitStack
from dataclasses import asdict

import numpy as np

from scripts.d1_stop_leg_damping import STOP_LEG_DAMPING_NSPM
from scripts.d1_stop_turn_composition import (
    STOP_TURN_COMPOSITION_SCHEMA,
    StopTurnCompositionController,
)
from scripts.probe_d1_heading_turn_yaw_authority import TurnAuthorityEnv
from wheel_legged_control.d1.control_loop import D1WheelLegControllerAdapter
from wheel_legged_control.d1.model import (
    JOINT_POSITION_HIGH,
    JOINT_POSITION_LOW,
    JOINT_TORQUE_LIMIT,
    JOINT_VELOCITY_LIMIT,
)


class StopTurnEnv(TurnAuthorityEnv):
    """Reuse the authority raw binding and native execution exactly once.

    The controller is replaced before the first reset. Separate stream ownership
    permits writing the stop record after the parent's terminal step closes its
    own streams. Neither mode adds a second controller call or physics step.
    """

    task_schema = "d1-heading-plane-stop-turn-composition-zero-task-v1"

    def __init__(self, *, diagnostic_output, **kwargs):
        super().__init__(diagnostic_output=diagnostic_output, **kwargs)
        self._controller = D1WheelLegControllerAdapter(StopTurnCompositionController(
            enabled=True, **asdict(self.wheel_leg_control)))
        self._composition_files = ExitStack()
        self._stop_stream = self._composition_files.enter_context(
            gzip.open(diagnostic_output / "stop_composition.jsonl.gz", "xt")  # noqa: SIM115
        )
        self._composition_closed = False

    def _build_heading_task_config(self):
        config = super()._build_heading_task_config()
        config["task_schema"] = self.task_schema
        config["turn_yaw_authority"]["leg_damping"] = "fixed post-stop latch only"
        config["stop_turn_composition"] = {
            "schema": STOP_TURN_COMPOSITION_SCHEMA,
            "fixed_damping_nspm": STOP_LEG_DAMPING_NSPM,
            "stop_gate": "executed nonzero forward to exact zero; held until nonzero forward",
            "turn_gate": "raw forward exactly zero and raw yaw nonzero",
            "overlap": "both mechanisms remain enabled when both command gates are active",
            "single_parent_PI_update": True,
            "no_target_shaping_or_new_gains": True,
            "zero_only": True,
        }
        return config

    def step(self, action):
        state = self.decision.context.state
        before = self.heading_decision
        try:
            result = super().step(action)
            record = self._controller.controller.last_damping
            if not (record.forward_command_mps == before.servo_command.forward_velocity_mps
                    == before.user_command.forward_velocity_mps):
                raise RuntimeError("stop latch differs from executed raw forward command")
            base_safe = np.clip(record.base_requested_torque_nm, -JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)
            blocked = (((state.joint_position >= JOINT_POSITION_HIGH) & (base_safe > 0.))
                | ((state.joint_position <= JOINT_POSITION_LOW) & (base_safe < 0.))
                | ((np.abs(state.joint_velocity) >= JOINT_VELOCITY_LIMIT)
                   & (base_safe*state.joint_velocity > 0.)))
            base_safe[blocked] = 0.
            payload = {k: v.tolist() if isinstance(v, np.ndarray) else v
                       for k, v in asdict(record).items()}
            self._stop_stream.write(json.dumps({
                "tick": before.tick, "endpoint_tick": before.tick+1, "record": payload,
                "joint_velocity_before_rad_s": state.joint_velocity.tolist(),
                "base_safe_torque_nm_same_state": base_safe.tolist(),
                "protected_increment_joint_power_w": float((record.safe_torque_nm-base_safe) @ state.joint_velocity),
                "power_phase": "decision-time velocity; not held-interval work",
            }, allow_nan=False)+"\n")
            if result[2] or result[3]:
                self._composition_files.close()
            return result
        except BaseException as error:
            self._error = {"type": type(error).__name__, "message": str(error)}
            raise

    def close(self):
        if self._composition_closed:
            return
        self._composition_closed = True
        try:
            self._composition_files.close()
        finally:
            super().close()
