"""One decision/control/measurement chain for training, evaluation and teleop.

The actor sees D1Decision.context only. Ground-truth state is returned separately
in the transition for rewards and evaluation. Every decision consumes one
provider publication and every step consumes exactly one decision.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Real
from typing import Protocol

import numpy as np

from .control_context import (
    D1AppliedAction,
    D1ControlContext,
    D1ControlProposal,
    D1JointTargetBaseline,
)
from .control_primitives import YawRatePI, terrain_normal_to_rpy
from .controllers import D1Command
from .hierarchical import D1LQRVMCController, D1MPCVMCController
from .model import D1Plant
from .state_estimation import D1MujocoTruthStateSource, D1StateEstimate, _immutable_array
from .state_provider import D1StateProvider
from .training_terrain import TrainingGroundReference
from .wheel_leg_controller import D1WheelLegController


@dataclass(frozen=True, slots=True)
class D1MotionCommand:
    """User command: forward m/s, yaw rad/s, clearance above local ground m."""

    forward_velocity_mps: float = 0.0
    yaw_rate_rps: float = 0.0
    clearance_m: float = 0.455

    def __post_init__(self):
        values = (self.forward_velocity_mps, self.yaw_rate_rps, self.clearance_m)
        if (
            any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                for value in values
            )
            or not np.isfinite(values).all()
        ):
            raise ValueError("motion command must contain finite real values")
        if abs(self.forward_velocity_mps) > 1 or abs(self.yaw_rate_rps) > 1:
            raise ValueError("development command range is +/-1 m/s and +/-1 rad/s")
        if not 0.38 <= self.clearance_m <= 0.53:
            raise ValueError("development clearance range is .38-.53 m")


class D1ControllerAdapter(Protocol):
    action_size: int
    action_schema: str

    def reset(self) -> None: ...

    def preview(
        self, command: D1Command, state: D1StateEstimate, ground: TrainingGroundReference
    ) -> D1ControlProposal: ...

    def compute(
        self,
        command: D1Command,
        state: D1StateEstimate,
        ground: TrainingGroundReference,
        action: np.ndarray,
    ) -> np.ndarray: ...


class D1ForceControllerAdapter:
    """LQR/MPC force residuals and explicit yaw PI memory on the common loop."""

    action_size = 2
    action_schema = "d1-v3-body-force-residual-v1"
    force_scale_n = np.asarray((11.25, 20.0))

    def __init__(
        self,
        controller: D1LQRVMCController | D1MPCVMCController,
        *,
        terrain_velocity_feedforward: bool = False,
    ):
        self.controller = controller
        self.terrain_velocity_feedforward = bool(terrain_velocity_feedforward)
        # The yaw branch is composed once, not on top of the legacy yaw P.
        self.controller.low_level.yaw_rate_gain = 0.0
        self.yaw = YawRatePI(kp=2.0, ki=3.0)

    def reset(self):
        self.controller.reset()
        self.yaw.reset()

    def _vertical_feedforward(self, state, ground):
        origin = state.base_origin_velocity(
            self.controller.low_level.nominal_base_com_offset_body_m
        )
        # Correct the historical VMC's COM damping to match its origin-height
        # error, without changing the frozen controller's default behavior.
        correction = state.base_linear_velocity_world[2] - origin[2]
        if self.terrain_velocity_feedforward:
            correction += (
                -np.tan(ground.pitch_rad) * origin[0] + np.tan(ground.roll_rad) * origin[1]
            )
        return self.controller.low_level.height_kd * float(correction)

    def preview(self, command, state, ground):
        proposal = self.controller.preview_baseline(
            command, state, vertical_feedforward_force_n=self._vertical_feedforward(state, ground)
        )
        return replace(
            proposal, memory=replace(proposal.memory, yaw_integral_nm=self.yaw.integral_nm)
        )

    def compute(self, command, state, ground, action):
        torque = self.controller.compute(
            command,
            state,
            residual_force_n=action * self.force_scale_n,
            vertical_feedforward_force_n=self._vertical_feedforward(state, ground),
        )
        yaw = self.yaw.compute(
            command.yaw_rate_rps - state.base_angular_velocity_body[2], self.controller.control_dt
        )
        torque[[3, 11]] -= yaw
        torque[[7, 15]] += yaw
        # Actual per-domain saturation belongs to the common actuator channel.
        return torque


class D1WheelLegControllerAdapter:
    """Per-leg/per-wheel target residuals, with no fictitious body-force mapping."""

    action_size = 8
    action_schema = D1WheelLegController.action_schema

    def __init__(self, controller: D1WheelLegController):
        self.controller = controller

    def reset(self):
        self.controller.reset()

    def preview(self, command, state, ground):
        nominal = self.controller.nominal_targets(command, state, ground_height_m=ground.height_m)
        return D1ControlProposal(
            state.sequence,
            state.control_time_s,
            D1JointTargetBaseline(
                nominal.nominal_joint_target_rad, nominal.nominal_wheel_speed_rad_s
            ),
            self.controller.control_memory,
        )

    def compute(self, command, state, ground, action):
        return self.controller.compute(command, state, action, ground_height_m=ground.height_m)


@dataclass(frozen=True, slots=True)
class D1Decision:
    motion_command: D1MotionCommand
    world_command: D1Command
    ground: TrainingGroundReference
    context: D1ControlContext


@dataclass(frozen=True, slots=True)
class D1ControlTransition:
    decision: D1Decision
    receipt: D1AppliedAction
    raw_action: np.ndarray
    requested_torque_nm: np.ndarray
    state: D1StateEstimate
    truth: D1StateEstimate

    def __post_init__(self):
        object.__setattr__(
            self,
            "raw_action",
            _immutable_array(self.raw_action, shape=(self.decision.context.action_size,)),
        )
        object.__setattr__(
            self, "requested_torque_nm", _immutable_array(self.requested_torque_nm, shape=(16,))
        )


class D1ControlLoop:
    """Own the step order, not the task schedule or reward.

    reset -> prepare(command) -> infer using decision.context -> step(action).
    prepare/read can repeat; they must not integrate, filter, or consume a
    command twice. The next command is supplied after the completed transition,
    equally by a training schedule, a replay file, or keyboard input.
    """

    schema = "d1-synchronized-control-loop-v1"

    def __init__(self, plant: D1Plant, provider: D1StateProvider, controller: D1ControllerAdapter):
        if plant.sampling_mode != "synchronized":
            raise ValueError("the unified control loop requires synchronized sampling")
        if plant.actuator_channel is None:
            raise ValueError("the unified control loop requires an explicit actuator channel")
        self.plant, self.provider, self.controller = plant, provider, controller
        self._truth_source = D1MujocoTruthStateSource(plant)
        self._decision: D1Decision | None = None
        self._receipt: D1AppliedAction | None = None
        self._initialized = False

    def reset(self, *, seed: int | None = None, base_position=None, base_quaternion=None):
        config = self.provider.config
        if config.kind == "imu_encoder_fusion":
            # The known placement prior must be the actual requested spawn, not
            # a fresh query of hidden pose after randomization.
            prior = np.asarray(config.initial_position_m)
            if base_position is not None and not np.array_equal(base_position, prior):
                raise ValueError("spawn must match the explicit fusion placement prior")
            base_position = prior
            if config.initial_rpy_rad != (0.0, 0.0, 0.0):
                import mujoco

                from .sensor_estimation import _rotation_from_rpy

                expected_quaternion = np.empty(4)
                mujoco.mju_mat2Quat(
                    expected_quaternion,
                    _rotation_from_rpy(np.asarray(config.initial_rpy_rad)).reshape(9),
                )
            else:
                expected_quaternion = np.asarray((1.0, 0.0, 0.0, 0.0))
            if base_quaternion is not None and not np.allclose(
                base_quaternion, expected_quaternion, atol=1e-12, rtol=0
            ):
                raise ValueError("spawn attitude must match the explicit fusion placement prior")
            base_quaternion = expected_quaternion
        self.plant.reset(base_position=base_position, base_quaternion=base_quaternion)
        self.controller.reset()
        self.provider.reset(seed=seed)
        self._truth_source.reset(seed=seed)
        self._decision, self._receipt = None, None
        self._initialized = True
        return self.provider.read()

    def prepare(self, command: D1MotionCommand) -> D1Decision:
        if not self._initialized:
            raise RuntimeError("reset must precede prepare")
        if not isinstance(command, D1MotionCommand):
            raise TypeError("command must be D1MotionCommand")
        if self._decision is not None:
            if command != self._decision.motion_command:
                raise RuntimeError("cannot replace a command after preparing this tick's decision")
            return self._decision
        state, ground = self.provider.read(), self.provider.ground_reference()
        # Ground fields are independent world-axis slope angles, not Euler RPY.
        roll, pitch = terrain_normal_to_rpy(
            -np.tan(ground.pitch_rad), np.tan(ground.roll_rad), float(state.base_rpy[2])
        )
        world = D1Command(
            command.forward_velocity_mps,
            command.yaw_rate_rps,
            ground.height_m + command.clearance_m,
            roll,
            pitch,
        )
        proposal = self.controller.preview(world, state, ground)
        context = D1ControlContext(
            state,
            proposal,
            self.controller.action_schema,
            self.controller.action_size,
            self._receipt,
        )
        self._decision = D1Decision(command, world, ground, context)
        return self._decision

    def step(self, action: np.ndarray, *, push_force_world_n=None) -> D1ControlTransition:
        decision = self._decision
        if decision is None:
            raise RuntimeError("prepare exactly one decision before step")
        raw = np.asarray(action, dtype=np.float64)
        if raw.shape != (self.controller.action_size,) or not np.isfinite(raw).all():
            raise ValueError("action must match the finite action schema")
        if push_force_world_n is not None:
            force = np.asarray(push_force_world_n, dtype=np.float64)
            if force.shape != (3,) or not np.isfinite(force).all():
                raise ValueError("push must be a finite world force vector")
        applied = np.clip(raw, -1, 1)
        try:
            torque = self.controller.compute(
                decision.world_command, decision.context.state, decision.ground, applied
            )
            self.plant.step(torque, push_force_world_n=push_force_world_n)
            state = self.provider.advance(decision.context.proposal.tick + 1)
            truth = self._truth_source.read()
        except Exception:
            # A partially executed control tick is not retryable. Reset owns
            # the controller, actuator queue and estimator histories together.
            self._initialized, self._decision = False, None
            raise
        receipt = D1AppliedAction(
            decision.context.proposal.tick,
            decision.context.proposal.control_time_s,
            state.control_time_s,
            applied,
        )
        result = D1ControlTransition(decision, receipt, raw, torque, state, truth)
        self._receipt, self._decision = receipt, None
        return result
