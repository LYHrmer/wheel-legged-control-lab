"""Eight bounded pure seams for the new course/99D/16D pilot.

The physical loop and Gym base used below are explicit fakes installed before
the actual new loop/env modules are loaded.  No MuJoCo, model or physics
module is imported or executed by these tests.  Root alone runs this file.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from course_ground_08 import CourseGroundMap
from full_drive_command_08 import FullDriveCommand, QualifiedCommandCaps
from full_drive_observation_08 import (
    OBSERVATION_SIZE,
    CourseScan,
    encode_full_drive_observation,
)
from full_drive_schedule_08 import (
    e0_stationary_probe_action,
    e0_stationary_probe_command,
    e1_ladder_schedule,
    heldout_schedule,
    precheck_nominal_path,
    training_schedule,
)
from full_drive_servo_08 import FullDriveCommandServo
from residual16_math_08 import ACTION_SCHEMA, compute_residual16

HERE = Path(__file__).resolve().parent
NOMINAL_Q = np.tile((0.0, 0.8, -1.5, 0.0), 4)


def _rows() -> list[dict]:
    def row(gid, name, kind, xyz, size, quat):
        return {
            "geom_id": gid, "name": name, "type": kind, "body_id": 0,
            "collision": True, "position_m": xyz, "size_m": size,
            "quaternion_wxyz": quat,
        }

    angle = math.radians(8)
    return [
        row(0, "floor", "plane", (0, 0, 0), (1, 1, 1), (1, 0, 0, 0)),
        row(3, "terrain_ramp_up", "box", (0, 0, 0.05), (0.75, 0.5, 0.04),
            (math.cos(angle / 2), 0, -math.sin(angle / 2), 0)),
        row(4, "terrain_ramp_tie", "box", (0, 0, 0.05), (0.75, 0.5, 0.04),
            (math.cos(angle / 2), 0, -math.sin(angle / 2), 0)),
    ]


def _caps() -> QualifiedCommandCaps:
    return QualifiedCommandCaps(1.6, 0.0, 0.0, 0.6, False)


def test_finite_capability_servo_rate_and_raw_stop_are_distinct():
    caps = _caps()
    servo = FullDriveCommandServo(caps)
    moving = servo.advance(0, FullDriveCommand(0.2))
    assert moving.raw_target.forward_velocity_mps == 0.2
    assert moving.applied.forward_velocity_mps == pytest.approx(0.005)
    assert servo.advance(0, FullDriveCommand(0.2)) is moving
    with pytest.raises(RuntimeError):
        servo.advance(0, FullDriveCommand(0.3))
    stopped_raw = servo.advance(1, FullDriveCommand())
    assert stopped_raw.raw_target.forward_velocity_mps == 0.0
    assert stopped_raw.applied.forward_velocity_mps == 0.0
    assert stopped_raw.forward_increment_mps == pytest.approx(-0.005)
    with pytest.raises(ValueError):
        caps.check(FullDriveCommand(-0.1))
    with pytest.raises(TypeError):
        FullDriveCommand(forward_velocity_mps=True)
    with pytest.raises(ValueError):
        FullDriveCommandServo(caps).advance(0, FullDriveCommand(1.6, yaw_rate_rps=0.6))


def test_compiled_ground_batch_equals_scalar_rotated_boundary_and_tie():
    ground = CourseGroundMap(_rows())
    points = np.asarray([
        (0.0, 0.0), (-0.6, 0.0), (0.6, 0.0),
        (0.0, 0.49), (0.0, 0.5), (0.0, 0.5000000001),
        (0.8, 0.0), (2.0, 0.0), (-2.0, 0.0),
    ])
    batched = ground.sample_many(points)
    scalar = tuple(ground.query(float(x), float(y)) for x, y in points)
    assert len(batched) == len(scalar)
    for left, right in zip(batched, scalar, strict=True):
        assert left.geom_id == right.geom_id
        assert left.height_m == pytest.approx(right.height_m, abs=1e-12)
        np.testing.assert_allclose(left.normal_world, right.normal_world, atol=1e-12)
    assert scalar[0].geom_id == 3  # equal-height box tie chooses the lower geom id
    angle = math.radians(8)
    assert scalar[0].height_m == pytest.approx(0.05 + 0.04 / math.cos(angle), abs=1e-12)
    np.testing.assert_allclose(
        scalar[0].normal_world,
        (-math.sin(angle), 0.0, math.cos(angle)), atol=1e-12,
    )
    assert scalar[-1].geom_id == 0


def test_observation99_slices_use_actual_servo_and_previous_effective_action():
    previous = np.zeros(16)
    previous[2] = 0.75
    state = _state(0, (-8.0, -4.7, 0.455), vx=0.2)
    context = SimpleNamespace(
        action_size=16, action_schema=ACTION_SCHEMA, state=state,
        proposal=_proposal(state, np.full(4, 0.5)),
        previous_normalized_action=previous,
    )
    decision = SimpleNamespace(
        motion_command=FullDriveCommand(0.4), context=context,
        world_command=SimpleNamespace(base_height_m=0.455),
    )
    scan = CourseScan(-8.0, -4.7, 0.0, 0.0, 0,
                      np.zeros(15), np.zeros(15), (0,) * 15)
    packet = encode_full_drive_observation(decision, scan)
    assert packet.values.shape == (OBSERVATION_SIZE,)
    assert packet.values.dtype == np.float32
    assert packet.values[0] == pytest.approx(0.1)
    assert packet.values[38] == pytest.approx(0.25)
    np.testing.assert_array_equal(packet.values[59:63], np.full(4, 0.125, dtype=np.float32))
    np.testing.assert_array_equal(packet.values[63:79], previous.astype(np.float32))
    assert packet.clipped_count == 0
    with pytest.raises(ValueError):
        encode_full_drive_observation(
            decision, CourseScan(-8.0, -4.6, 0.0, 0.0, 0,
                                 np.zeros(15), np.zeros(15), (0,) * 15),
        )


def test_preregistered_schedule_boundaries_nominal_path_and_probe():
    e1 = e1_ladder_schedule()
    assert len(e1.raw_commands) == 1600
    for tick, speed in ((174, 0), (175, 0.4), (354, 0.4), (355, 0.6),
                        (815, 1.6), (1094, 1.6), (1095, 0)):
        assert e1.command_source(tick, tick * 0.01).forward_velocity_mps == speed
    assert precheck_nominal_path(e1, _caps()).passed
    yaw = heldout_schedule("flat_1p2_yaw")
    hold_start = 175 + 240
    assert yaw.raw_commands[hold_start].yaw_rate_rps == 0.3
    assert yaw.raw_commands[hold_start + 100].yaw_rate_rps == -0.3
    assert yaw.raw_commands[hold_start + 300].yaw_rate_rps == 0.3
    assert precheck_nominal_path(yaw, _caps()).passed
    assert all(heldout_schedule(name).seed == 88501 + i
               for i, name in enumerate(("flat_0p6", "flat_1p6", "flat_1p2_yaw",
                                         "bumps_0p4", "rough_0p35", "ramp_0p35")))
    assert np.count_nonzero(e0_stationary_probe_action(200)) == 1
    assert e0_stationary_probe_action(200)[0] == 0.15
    assert not e0_stationary_probe_action(210).any()
    assert e0_stationary_probe_command(200, 2.0).forward_velocity_mps == 0.0
    with pytest.raises(ValueError):
        training_schedule(level=0, terrain="flat", target_speed_mps=0.3,
                          yaw_amplitude_rps=0.0, episode_index=3,
                          flat_episode_ordinal=3, schedule_seed=88401)


def _state(tick: int, xyz: tuple[float, float, float], *, vx: float = 0.0):
    return SimpleNamespace(
        sequence=tick, control_time_s=tick * 0.01,
        base_position=np.asarray(xyz, dtype=np.float64), base_rpy=np.zeros(3),
        base_rotation=np.eye(3),
        base_linear_velocity_body=np.asarray((vx, 0.0, 0.0)),
        base_angular_velocity_body=np.zeros(3),
        base_angular_velocity_world=np.zeros(3),
        projected_gravity_body=np.asarray((0.0, 0.0, -1.0)),
        joint_position=NOMINAL_Q.copy(), joint_velocity=np.zeros(16),
        foot_offset_world=np.zeros((4, 3)), foot_jacobian=np.zeros((4, 3, 4)),
        wheel_contact=np.ones(4, dtype=bool), age_s=0.0,
    )


def _proposal(state, wheel_integral):
    return SimpleNamespace(
        tick=state.sequence, control_time_s=state.control_time_s,
        baseline=SimpleNamespace(
            nominal_joint_target_rad=NOMINAL_Q.copy(),
            nominal_wheel_speed_rad_s=np.ones(4),
        ),
        memory=SimpleNamespace(wheel_integral_nm=np.asarray(wheel_integral).copy()),
    )


def _fake_runtime(monkeypatch):
    """Load actual 08 loop/env against explicit pure doubles, never D1's import graph."""
    def fake_package(name):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    fake_package("wheel_legged_control")
    fake_package("wheel_legged_control.d1")

    class Context:
        def __init__(self, state, proposal, action_schema, action_size, previous=None):
            self.state, self.proposal = state, proposal
            self.action_schema, self.action_size = action_schema, action_size
            self.previous_normalized_action = (
                np.zeros(action_size) if previous is None else previous.normalized_action
            )

    @dataclass(frozen=True, slots=True)
    class Transition:
        decision: object
        receipt: object
        raw_action: np.ndarray
        requested_torque_nm: np.ndarray
        state: object
        truth: object

        def __post_init__(self):
            pass

    class BaseLoop:
        def __init__(self, plant, provider, controller):
            self.plant, self.provider, self.controller = plant, provider, controller
            self._initialized, self._decision, self._receipt = False, None, None

        def reset(self, *, seed=None, base_position=None, base_quaternion=None):
            self.plant.reset(base_position=base_position)
            self.controller.reset()
            state = self.provider.reset(seed=seed)
            self._initialized, self._decision, self._receipt = True, None, None
            return state

        def step(self, action, *, push_force_world_n=None):
            decision = self._decision
            assert decision is not None
            applied = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
            torque = self.controller.compute(
                decision.world_command, decision.context.state, decision.ground, applied,
            )
            self.plant.step(torque)
            state = self.provider.advance(decision.context.state.sequence + 1)
            receipt = SimpleNamespace(
                tick=decision.context.state.sequence,
                normalized_action=applied.copy(),
                end_time_s=state.control_time_s,
            )
            result = Transition(decision, receipt, applied.copy(), torque, state, state)
            self._receipt, self._decision = receipt, None
            return result

    context = types.ModuleType("wheel_legged_control.d1.control_context")
    context.D1ControlContext = Context
    control_loop = types.ModuleType("wheel_legged_control.d1.control_loop")
    control_loop.D1ControlLoop, control_loop.D1ControlTransition = BaseLoop, Transition
    primitives = types.ModuleType("wheel_legged_control.d1.control_primitives")
    primitives.terrain_normal_to_rpy = lambda slope_x, slope_y, yaw: (0.0, 0.0)
    terrain = types.ModuleType("wheel_legged_control.d1.training_terrain")

    @dataclass(frozen=True)
    class GroundReference:
        height_m: float = 0.0
        pitch_rad: float = 0.0
        roll_rad: float = 0.0

    terrain.TrainingGroundReference = GroundReference
    for name, value in (
        ("control_context", context), ("control_loop", control_loop),
        ("control_primitives", primitives), ("training_terrain", terrain),
    ):
        monkeypatch.setitem(sys.modules, f"wheel_legged_control.d1.{name}", value)

    gym = types.ModuleType("gymnasium")

    class GymEnv:
        def reset(self, *, seed=None):
            self.np_random = np.random.default_rng(seed)

    class Box:
        def __init__(self, low, high, shape, dtype):
            self.low, self.high, self.shape, self.dtype = low, high, shape, dtype

    gym.Env = GymEnv
    gym.spaces = SimpleNamespace(Box=Box)
    monkeypatch.setitem(sys.modules, "gymnasium", gym)

    def load(name):
        spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    return load("full_drive_loop_08"), load("full_drive_env_08"), GroundReference


class _FakePlant:
    control_dt = 0.01
    physics_steps = 5
    sampling_mode = "synchronized"
    actuator_channel = object()
    actuator_torque_limit_nm = np.full(16, 80.0)

    def __init__(self, ground):
        self.ground = ground
        self.native_steps_total = 0
        self.undesired_ground_contacts = 0
        self.last_control_interval_actuator_traces = ()

    def reset(self, *, base_position):
        self.position = tuple(base_position)
        self.last_control_interval_actuator_traces = ()

    def step(self, torque):
        assert np.asarray(torque).shape == (16,)
        self.native_steps_total += 5
        self.position = (self.position[0] + 0.01, self.position[1], self.position[2])
        self.last_control_interval_actuator_traces = tuple(
            SimpleNamespace(applied_nm=np.zeros(16)) for _ in range(5)
        )

    def course_ground_hit(self, x, y):
        return self.ground.query(float(x), float(y))


class _FakeProvider:
    def __init__(self, plant, ground_reference):
        self.plant, self.reference = plant, ground_reference
        self.state = None

    def reset(self, *, seed=None):
        self.state = _state(0, self.plant.position)
        return self.state

    def read(self):
        return self.state

    def ground_reference(self):
        return self.reference()

    def advance(self, tick):
        assert tick == self.state.sequence + 1
        self.state = _state(tick, self.plant.position, vx=1.0)
        return self.state


class _FakeController:
    action_size = 16
    action_schema = ACTION_SCHEMA
    control_schema = "pure-fake-only"

    def __init__(self):
        self.compute_count = 0
        self.preview_count = 0
        self.terminal_preview_count = 0
        self.integral = np.zeros(4)
        self.last_record = None

    def reset(self):
        self.integral = np.zeros(4)
        self.last_record = None

    def action_gate(self, command):
        return bool(command.raw_forward_velocity_mps or command.raw_yaw_rate_rps)

    def preview(self, command, state, ground):
        self.preview_count += 1
        return _proposal(state, self.integral)

    def terminal_observation_proposal(self, command, state, ground):
        self.terminal_preview_count += 1
        return _proposal(state, self.integral)

    def compute(self, command, state, ground, action):
        self.compute_count += 1
        result = compute_residual16({
            "joint_position_rad": state.joint_position,
            "joint_velocity_rad_s": state.joint_velocity,
            "nominal_joint_target_rad": NOMINAL_Q,
            "nominal_wheel_speed_rad_s": np.ones(4),
            "support_torque_nm": np.zeros(16),
            "base_rotation_world_from_body": np.eye(3),
            "foot_jacobian_world": np.zeros((4, 3, 4)),
            "body_com_forward_mps": 0.0,
            "action_enabled": True,
            "leg_longitudinal_damping_active": False,
            "body_common_p_active": False,
        }, action, self.integral)
        self.integral = result["wheel_integral_after_nm"].copy()
        self.last_record = {"calculation": result}
        return result["safe_torque_nm"].copy()


def _components(loop_module, reference):
    ground = CourseGroundMap(_rows())
    plant = _FakePlant(ground)
    provider = _FakeProvider(plant, reference)
    controller = _FakeController()
    loop = loop_module.FullDriveLoop(plant, provider, controller, caps=_caps())
    return {"plant": plant, "controller": controller, "loop": loop, "ground_map": ground}


def _summary(nonwheel=0):
    def read(plant, *, end_time_s):
        return {
            "start_time_s": end_time_s - 0.01,
            "end_time_s": end_time_s,
            "native_returns": 5,
            "nonwheel_contact_count": nonwheel,
            "max_abs_roll_deg": 0.0,
            "max_abs_pitch_deg": 0.0,
            "terrain_family_positive_wheel_load": {},
        }

    return read


def test_prepared_loop_has_one_control_one_pi_commit_five_native_and_true_gate(monkeypatch):
    runtime_loop, _, reference = _fake_runtime(monkeypatch)
    parts = _components(runtime_loop, reference)
    loop, controller, plant = parts["loop"], parts["controller"], parts["plant"]
    loop.reset(seed=7, base_position=(-8.0, -4.7, 0.455))
    zero = FullDriveCommand()
    decision = loop.prepare(FullDriveCommand(0.005), raw_operator_command=zero)
    assert loop.prepare(FullDriveCommand(0.005), raw_operator_command=zero) is decision
    transition = loop.step(np.ones(16))
    np.testing.assert_array_equal(transition.policy_clipped_action, np.ones(16))
    np.testing.assert_array_equal(transition.receipt.normalized_action, np.zeros(16))
    assert controller.preview_count == controller.compute_count == 1
    assert plant.native_steps_total == 5
    np.testing.assert_allclose(controller.integral, np.full(4, 0.03))
    loop.prepare(FullDriveCommand(0.01), raw_operator_command=FullDriveCommand(0.2))
    second = loop.step(np.zeros(16))
    assert second.action_enabled is True
    assert controller.compute_count == 2 and plant.native_steps_total == 10
    np.testing.assert_allclose(controller.integral, np.full(4, 0.06))
    _assert_adapter_math_source_contract()


def test_terminal_observation_is_poststate_without_future_request_or_preview(monkeypatch):
    runtime_loop, env_module, reference = _fake_runtime(monkeypatch)
    parts = _components(runtime_loop, reference)
    requested = []
    begun = []

    def source(tick, time_s):
        requested.append(tick)
        return FullDriveCommand(0.2)

    env = env_module.FullDriveCourseEnv(
        caps=_caps(), command_source=source, max_steps=1, mode="test_actor",
        pure_components_factory=lambda: parts,
    )
    env.set_native_interval_reader(
        _summary(),
        begin_interval=lambda plant, *, start_time_s: begun.append(start_time_s),
    )
    before, _ = env.reset(seed=123)
    after, _, terminated, truncated, info = env.step(np.ones(16))
    assert not terminated and truncated
    assert requested == [0]
    assert begun == [0.0]
    assert parts["controller"].preview_count == 1
    assert parts["controller"].terminal_preview_count == 1
    assert parts["controller"].compute_count == 1
    assert parts["plant"].native_steps_total == 5
    assert after[0] != before[0]
    np.testing.assert_array_equal(after[63:79], info["applied_action"].astype(np.float32))
    assert info["TimeLimit.truncated"] is True
    assert "actual_poststep" in info["terminal_observation_source"]


def test_fresh_reset_seed_and_action_memory_do_not_refund_native_budget(monkeypatch):
    runtime_loop, env_module, reference = _fake_runtime(monkeypatch)
    parts = _components(runtime_loop, reference)
    env = env_module.FullDriveCourseEnv(
        caps=_caps(), command_source=lambda tick, time: FullDriveCommand(0.2),
        max_steps=2, mode="test_actor", pure_components_factory=lambda: parts,
    )
    assert env._steps is None and env.decision is None
    env.set_native_interval_reader(
        _summary(), begin_interval=lambda plant, *, start_time_s: None,
    )
    first, meta1 = env.reset(seed=77351)
    env.step(np.zeros(16))
    assert parts["plant"].native_steps_total == 5
    second, meta2 = env.reset(seed=77351)
    np.testing.assert_array_equal(first, second)
    assert meta1["episode_metadata"]["measurement_seed"] == meta2["episode_metadata"]["measurement_seed"]
    assert env._steps == 0 and parts["plant"].native_steps_total == 5
    np.testing.assert_array_equal(parts["controller"].integral, np.zeros(4))
    with pytest.raises(ValueError):
        env_module.FullDriveCourseEnv(
            caps=_caps(), command_source=lambda tick, time: FullDriveCommand(),
            mode="train", pure_components_factory=lambda: parts,
        )


def test_any_native_nonwheel_contact_terminates_with_real_penalty(monkeypatch):
    runtime_loop, env_module, reference = _fake_runtime(monkeypatch)
    parts = _components(runtime_loop, reference)
    calls = []

    def source(tick, time_s):
        calls.append(tick)
        return FullDriveCommand(0.2)

    env = env_module.FullDriveCourseEnv(
        caps=_caps(), command_source=source, max_steps=2, mode="test_actor",
        pure_components_factory=lambda: parts,
    )
    with pytest.raises(RuntimeError):
        env.step(np.zeros(16))  # reset and audited interval binding are mandatory
    env.set_native_interval_reader(
        _summary(nonwheel=1), begin_interval=lambda plant, *, start_time_s: None,
    )
    env.reset(seed=1)
    _, reward, terminated, truncated, info = env.step(np.zeros(16))
    assert terminated and not truncated
    assert calls == [0]
    assert info["terminal_reason"] == "nonwheel_ground_contact"
    assert info["metrics"]["native_interval_nonwheel_contacts"] == 1
    assert info["metrics"]["undesired_ground_contacts"] == 0
    assert info["reward_terms"]["termination"] == -10.0
    assert reward == pytest.approx(sum(info["reward_terms"].values()))
    assert parts["plant"].native_steps_total == 5
    with pytest.raises(RuntimeError):
        env.step(np.zeros(16))


def _assert_adapter_math_source_contract():
    """AST pins the engine-dependent adapter without importing its old model graph."""
    tree = ast.parse((HERE / "full_drive_controller_08.py").read_text())
    adapter = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "FullDriveControllerAdapter")
    compute = next(node for node in adapter.body if isinstance(node, ast.FunctionDef)
                   and node.name == "compute")
    names = [node.func.id for node in ast.walk(compute) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)]
    assert names.count("compute_residual16") == 1
    assert names.count("nominal_support") == 1
    assert all(not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "compute" and isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == "_nominal") for node in ast.walk(compute))
    assignments = [node for node in ast.walk(compute) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Attribute) and target.attr == "_wheel_integral_nm"
                           for target in node.targets)]
    assert len(assignments) == 1
