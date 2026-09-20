"""Nonintegrating command, wheel-loop and protection checks for the turn damper."""
import json
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from scripts.d1_flat_plane_env import D1FlatPlaneHeadingEnv
from scripts.d1_heading_tracking_env import D1HeadingTrackingEnv, load_heading_policy
from scripts.d1_stop_leg_damping import STOP_LEG_DAMPING_NSPM
from scripts.d1_turn_leg_damping import TurnLegDampingController
from scripts.probe_d1_heading_plane_stop_damping import PlaneStopDampingEnv
from scripts.probe_d1_heading_turn_center import TurnCenterEnv
from scripts.probe_d1_heading_turn_damping import TurnDampingEnv
from wheel_legged_control.d1.control_loop import D1MotionCommand
from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.locomotion_checkpoint import (
    _environment_contract,
    load_locomotion_policy,
)
from wheel_legged_control.d1.model import JOINT_TORQUE_LIMIT, D1Plant
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource
from wheel_legged_control.d1.wheel_leg_controller import D1WheelLegController


@pytest.fixture(autouse=True, scope="module")
def no_integration():
    def forbidden(*args, **kwargs):
        raise AssertionError("turn damper preflight must not integrate")

    with pytest.MonkeyPatch.context() as patch:
        for name in ("mj_step", "mj_step1", "mj_step2"):
            patch.setattr(mujoco, name, forbidden)
        yield


@pytest.fixture(scope="module")
def state(no_integration):
    plant = D1Plant(sampling_mode="synchronized")
    snapshot = D1MujocoTruthStateSource(plant).reset()
    assert plant.data.time == 0.
    return replace(snapshot, joint_velocity=np.tile([.2, -.4, .1, .3], 4))


def bind(controller, state, forward=0., yaw=.6):
    controller.bind_raw_command(forward_velocity_mps=forward, yaw_rate_rps=yaw,
                                control_time_s=state.control_time_s)


@pytest.mark.parametrize("forward,raw_yaw,servo_yaw,active", [
    (0., .6, .9, True), (-0., -.6, -.9, True), (.25, .6, .9, False),
    (-.25, -.6, -.9, False), (0., 0., 1., False), (-0., -0., -1., False)])
def test_raw_gate_parent_once_wheel_loop_and_preview(state, monkeypatch, forward, raw_yaw, servo_yaw, active):
    candidate = TurnLegDampingController()
    baseline = D1WheelLegController()
    bind(candidate, state, forward, raw_yaw)
    command = D1Command(forward_velocity_mps=forward, yaw_rate_rps=servo_yaw)
    base_nominal = baseline.nominal_targets(command, state)
    for _ in range(2):
        nominal = candidate.nominal_targets(command, state)
        assert nominal.nominal_wheel_speed_rad_s.tobytes() == base_nominal.nominal_wheel_speed_rad_s.tobytes()
        assert nominal.nominal_joint_target_rad.tobytes() == base_nominal.nominal_joint_target_rad.tobytes()
    assert candidate.last_damping is candidate.last_result is None
    assert not np.any(candidate.control_memory.wheel_integral_nm)
    base_torque = baseline.compute(command, state, np.zeros(8))
    parent, calls = D1WheelLegController.compute, []

    def tracked(self, *args, **kwargs):
        torque = parent(self, *args, **kwargs)
        calls.append((torque, self.last_result))
        return torque

    monkeypatch.setattr(D1WheelLegController, "compute", tracked)
    torque = candidate.compute(command, state, np.zeros(8))
    record = candidate.last_damping
    assert len(calls) == 1 and record.active == active
    assert torque[3::4].tobytes() == base_torque[3::4].tobytes()
    assert candidate.control_memory.wheel_integral_nm.tobytes() == baseline.control_memory.wheel_integral_nm.tobytes()
    for field in ("support_nm", "wheel_nm", "joint_target_rad", "wheel_speed_target_rad_s"):
        assert getattr(candidate.last_result, field).tobytes() == getattr(baseline.last_result, field).tobytes()
    if not active:
        assert torque is calls[0][0] and candidate.last_result is calls[0][1]
        assert torque.tobytes() == base_torque.tobytes()
        assert not np.any(record.delta_torque_nm) and record.joint_power_w == 0.
    with pytest.raises(ValueError):
        record.delta_torque_nm.setflags(write=True)
    bind(candidate, state, 0., 0.)
    candidate.nominal_targets(D1Command(yaw_rate_rps=.9), state)
    assert candidate.last_damping is record
    candidate.compute(D1Command(yaw_rate_rps=.9), state, np.zeros(8))
    assert not candidate.last_damping.active
    candidate.reset()
    assert candidate.raw_binding is candidate.last_damping is None
    assert not np.any(candidate.control_memory.wheel_integral_nm)


def test_all_preview_targets_are_exact_parent_objects(state, monkeypatch):
    original, calls = D1WheelLegController.nominal_targets, []

    def tracked(self, *args, **kwargs):
        value = original(self, *args, **kwargs)
        calls.append(value)
        return value

    monkeypatch.setattr(D1WheelLegController, "nominal_targets", tracked)
    candidate = TurnLegDampingController()
    for yaw in (.6, 0.):
        bind(candidate, state, 0., yaw)
        assert candidate.nominal_targets(D1Command(yaw_rate_rps=.8), state) is calls[-1]


def test_body_x_tilt_projection_and_wheel_spin_damping_preserved(state):
    angle = .23
    rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                         [-np.sin(angle), 0., np.cos(angle)]])
    tilted = replace(state, base_rotation=rotation@state.base_rotation,
        base_linear_velocity_world=rotation@state.base_linear_velocity_world,
        base_angular_velocity_world=rotation@state.base_angular_velocity_world)
    candidate = TurnLegDampingController()
    bind(candidate, tilted)
    command = D1Command(yaw_rate_rps=.6)
    candidate.compute(command, tilted, np.zeros(8))
    record = candidate.last_damping
    jx = np.einsum("a,kab->kb", tilted.base_rotation[:, 0], tilted.foot_jacobian[:, :, :3])
    u = np.einsum("ki,ki->k", jx, tilted.joint_velocity.reshape(4, 4)[:, :3])
    np.testing.assert_allclose(record.leg_relative_forward_mps, u, atol=1e-12, rtol=0)
    np.testing.assert_allclose(record.delta_torque_nm.reshape(4, 4)[:, :3],
                               -STOP_LEG_DAMPING_NSPM*jx*u[:, None], atol=1e-12, rtol=0)
    assert not np.any(np.signbit(record.delta_torque_nm[3::4]))
    assert record.joint_power_w == pytest.approx(-STOP_LEG_DAMPING_NSPM*(u@u), abs=1e-12)
    horizontal_u = np.einsum("ki,ki->k", tilted.foot_jacobian[:, 0, :3],
                             tilted.joint_velocity.reshape(4, 4)[:, :3])
    assert np.max(np.abs(u-horizontal_u)) > 1e-4
    qd = tilted.joint_velocity.copy()
    qd[3::4] += np.array([.1, -.1, .1, -.1])
    perturbed = replace(tilted, joint_velocity=qd)
    second = TurnLegDampingController()
    bind(second, perturbed)
    second.compute(command, perturbed, np.zeros(8))
    assert second.last_damping.delta_torque_nm.tobytes() == record.delta_torque_nm.tobytes()
    np.testing.assert_allclose(second.last_result.wheel_nm[3::4]-candidate.last_result.wheel_nm[3::4],
                               -(2.2+3.*.01)*np.array([.1, -.1, .1, -.1]), atol=1e-12, rtol=0)


def test_increment_is_added_before_original_torque_clip(state, monkeypatch):
    jac = np.zeros((4, 3, 4))
    jac[0, 0, 1] = 1.
    velocity = np.zeros(16)
    velocity[1] = 30./STOP_LEG_DAMPING_NSPM
    current = replace(state, foot_jacobian=jac, joint_velocity=velocity)
    original = D1WheelLegController.compute

    def saturated_parent(self, *args, **kwargs):
        original(self, *args, **kwargs)
        base = self.last_result
        request = base.requested_torque_nm.copy()
        request[1] = JOINT_TORQUE_LIMIT[1]+20.
        safe = np.clip(request, -JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)
        self.last_result = replace(base, requested_torque_nm=request, torque_nm=safe,
                                  torque_limited=safe != request)
        return safe.copy()

    monkeypatch.setattr(D1WheelLegController, "compute", saturated_parent)
    candidate = TurnLegDampingController()
    bind(candidate, current)
    torque = candidate.compute(D1Command(yaw_rate_rps=.6), current, np.zeros(8))
    assert torque[1] == pytest.approx(JOINT_TORQUE_LIMIT[1]-10., abs=1e-12)


@pytest.mark.parametrize("action", [np.ones(8)*1e-300, np.zeros(8, dtype=bool),
    np.zeros(8, dtype=complex), np.full(8, np.nan), np.zeros(7), ["0"]*8])
def test_invalid_action_rejected_before_pi_mutation(state, action):
    candidate = TurnLegDampingController()
    bind(candidate, state)
    with pytest.raises((TypeError, ValueError)):
        candidate.compute(D1Command(yaw_rate_rps=.6), state, action)
    assert candidate.last_result is candidate.last_damping is None
    assert not np.any(candidate.control_memory.wheel_integral_nm)


def test_binding_validation_and_boundary_preview_snapshot(state):
    candidate = TurnLegDampingController()
    with pytest.raises(RuntimeError):
        candidate.nominal_targets(D1Command(), state)
    with pytest.raises(TypeError):
        candidate.bind_raw_command(forward_velocity_mps=False, yaw_rate_rps=.6, control_time_s=0.)
    candidate.bind_raw_command(forward_velocity_mps=0., yaw_rate_rps=.6, control_time_s=1.)
    with pytest.raises(ValueError):
        candidate.compute(D1Command(), state, np.zeros(8))
    bind(candidate, state, 0., .6)
    with pytest.raises(ValueError):
        candidate.compute(D1Command(forward_velocity_mps=.1), state, np.zeros(8))
    assert candidate.last_result is None and not np.any(candidate.control_memory.wheel_integral_nm)
    for tick, yaw in ((199, 0.), (200, .6), (249, .6), (250, 0.)):
        current = replace(state, control_time_s=tick*.01+1e-11, measurement_time_s=tick*.01)
        candidate.bind_raw_command(forward_velocity_mps=0., yaw_rate_rps=yaw, control_time_s=tick*.01)
        candidate.compute(D1Command(yaw_rate_rps=.8), current, np.zeros(8))
        record = candidate.last_damping
        assert record.active == (yaw != 0.) and record.control_time_s == tick*.01
        following = replace(current, control_time_s=(tick+1)*.01)
        bind(candidate, following, 0., 0.)
        candidate.nominal_targets(D1Command(yaw_rate_rps=.8), following)
        assert candidate.last_damping is record


def test_disabled_unbound_is_exact_parent(state):
    candidate, baseline = TurnLegDampingController(enabled=False), D1WheelLegController()
    command = D1Command(yaw_rate_rps=.8)
    assert candidate.compute(command, state, np.zeros(8)).tobytes() == baseline.compute(command, state, np.zeros(8)).tobytes()
    assert candidate.control_schema == baseline.control_schema and not candidate.last_damping.active


def test_raw_callback_identity_cache_and_preview_observation(tmp_path):
    calls, raw = [], D1MotionCommand(yaw_rate_rps=.6)

    def source(time_s):
        calls.append(time_s)
        return raw

    env = TurnDampingEnv(command_source=source, diagnostic_output=tmp_path, episode_seconds=.01)
    baseline = D1FlatPlaneHeadingEnv(command_source=lambda t: raw, episode_seconds=.01)
    try:
        observation, info = env.reset(seed=55101)
        base_observation, _ = baseline.reset(seed=55101)
        assert observation.tobytes() == base_observation.tobytes()
        assert env.heading_decision.user_command is raw and calls == [0.]
        assert env.loop.controller is env._controller and env.loop.plant is env.plant
        env._prepare()
        env._prepare()
        assert calls == [0.] and env.raw_callback_count == 1
        assert env._controller.controller.last_damping is None
        config = info["episode_metadata"]["heading_task_config"]
        assert config["task_schema"] == env.task_schema
        assert config["turn_leg_damping"]["fixed_damping_nspm"] == STOP_LEG_DAMPING_NSPM
        assert env.plant.data.time == baseline.plant.data.time == 0.
    finally:
        env.close()
        baseline.close()


@pytest.mark.parametrize("source_type", [D1HeadingTrackingEnv, D1FlatPlaneHeadingEnv,
                                        PlaneStopDampingEnv, TurnCenterEnv])
@pytest.mark.parametrize("loader,match", [(load_heading_policy, "heading_task_config"),
                                         (load_locomotion_policy, "task_schema")])
def test_previous_checkpoint_rejected_before_deserialization(tmp_path, monkeypatch, source_type, loader, match):
    from stable_baselines3 import PPO

    def forbidden(*args, **kwargs):
        pytest.fail("incompatible checkpoint deserialized")

    monkeypatch.setattr(PPO, "load", forbidden)
    kwargs = {"diagnostic_output": None} if source_type is PlaneStopDampingEnv else {}
    if source_type is TurnCenterEnv:
        source_folder = tmp_path/"source"
        source_folder.mkdir()
        kwargs = {"diagnostic_output": source_folder, "command_source": lambda t: D1MotionCommand()}
    source = source_type(episode_seconds=.01, **kwargs)
    target = TurnDampingEnv(command_source=lambda t: D1MotionCommand(),
        diagnostic_output=tmp_path, episode_seconds=.01)
    try:
        source.reset(seed=55101)
        target.reset(seed=55101)
        sidecar = tmp_path/"old_checkpoint.json"
        sidecar.write_text(json.dumps({**_environment_contract(source),
            "recorded_episode": source.episode_metadata, "model_sha256": "not-read"}))
        with pytest.raises(ValueError, match=match):
            loader(tmp_path/"absent.zip", sidecar, target)
        assert source.plant.data.time == target.plant.data.time == 0.
    finally:
        source.close()
        target.close()
