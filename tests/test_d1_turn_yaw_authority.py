"""Nonintegrating checks of original feedback restoration and actuator limits."""
import json
from dataclasses import replace
from io import StringIO
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from scripts.d1_flat_plane_env import D1FlatPlaneHeadingEnv
from scripts.d1_heading_tracking_env import load_heading_policy
from scripts.d1_turn_yaw_authority import TurnYawAuthorityController
from scripts.probe_d1_heading_plane_stop_damping import PlaneStopDampingEnv
from scripts.probe_d1_heading_turn_center import TurnCenterEnv
from scripts.probe_d1_heading_turn_damping import TurnDampingEnv
from scripts.probe_d1_heading_turn_yaw_authority import TurnAuthorityEnv, _CapDiagnosticStream
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
        raise AssertionError("authority preflight must not integrate")

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
def test_raw_gate_parent_once_preview_and_leg_isolation(state, monkeypatch, forward, raw_yaw, servo_yaw, active):
    candidate, baseline = TurnYawAuthorityController(), D1WheelLegController()
    bind(candidate, state, forward, raw_yaw)
    command = D1Command(forward_velocity_mps=forward, yaw_rate_rps=servo_yaw)
    old = baseline.nominal_targets(command, state)
    nominal = candidate.nominal_targets(command, state)
    assert nominal.nominal_joint_target_rad.tobytes() == old.nominal_joint_target_rad.tobytes()
    assert nominal.leg_extension_target_m.tobytes() == old.leg_extension_target_m.tobytes()
    assert candidate.last_authority is candidate.last_result is None
    assert not np.any(candidate.control_memory.wheel_integral_nm)
    assert candidate.yaw_request_limit_rps == .6
    base_torque = baseline.compute(command, state, np.zeros(8))
    original, calls = D1WheelLegController.compute, []

    def parent(self, *args, **kwargs):
        value = original(self, *args, **kwargs)
        calls.append((value, self.last_result))
        return value

    monkeypatch.setattr(D1WheelLegController, "compute", parent)
    torque = candidate.compute(command, state, np.zeros(8))
    record = candidate.last_authority
    assert len(calls) == 1 and torque is calls[0][0] and candidate.last_result is calls[0][1]
    assert record.active == active and record.inner_cap_applied == (not active)
    assert record.raw_yaw_rate_rps == raw_yaw and record.servo_yaw_rate_rps == servo_yaw
    for field in ("leg_pd_nm", "support_nm", "joint_target_rad", "support_force_n"):
        assert getattr(candidate.last_result, field).tobytes() == getattr(baseline.last_result, field).tobytes()
    if active:
        assert abs(record.effective_yaw_request_rps) > .6 and not record.inner_cap_occupied
        assert abs(record.effective_yaw_request_rps) != abs(record.body_yaw_rate_rps)
    else:
        assert torque.tobytes() == base_torque.tobytes()
        assert candidate.control_memory.wheel_integral_nm.tobytes() == baseline.control_memory.wheel_integral_nm.tobytes()
    with pytest.raises(ValueError):
        record.wheel_target_rad_s.setflags(write=True)
    bind(candidate, state, 0., 0.)
    candidate.nominal_targets(D1Command(yaw_rate_rps=.9), state)
    assert candidate.last_authority is record
    candidate.reset()
    assert candidate.raw_binding is candidate.last_authority is None
    assert not np.any(candidate.control_memory.wheel_integral_nm)


def test_inactive_exact_parent_nominal_object(state, monkeypatch):
    original, calls = D1WheelLegController.nominal_targets, []

    def parent(self, *args, **kwargs):
        value = original(self, *args, **kwargs)
        calls.append(value)
        return value

    monkeypatch.setattr(D1WheelLegController, "nominal_targets", parent)
    candidate = TurnYawAuthorityController()
    bind(candidate, state, 0., 0.)
    assert candidate.nominal_targets(D1Command(yaw_rate_rps=.8), state) is calls[-1]


@pytest.mark.parametrize("body_rate,servo_yaw,expected_target_clip", [(0., .84, False), (-20., 1., True)])
def test_original_pi_antiwindup_and_distinct_target_torque_limits(state, body_rate, servo_yaw, expected_target_clip):
    angular = np.array([0., 0., body_rate])
    current = replace(state, base_angular_velocity_body=angular,
                      base_angular_velocity_world=state.base_rotation@angular)
    candidate = TurnYawAuthorityController()
    bind(candidate, current)
    before = np.array([.4, -.4, 3.9, -3.9])
    candidate._wheel_integral_nm[:] = before
    candidate.compute(D1Command(yaw_rate_rps=servo_yaw), current, np.zeros(8))
    record = candidate.last_authority
    expected_yaw = servo_yaw+4.*(servo_yaw-body_rate)
    lateral = current.foot_offset_world[:, 1]
    target = np.clip(-expected_yaw*lateral/.087, -30., 30.)
    error = target-current.joint_velocity[3::4]
    trial = np.clip(before+3.*.01*error, -4., 4.)
    request = 2.2*error+trial
    accept = (np.abs(request) <= 12.) | (request*error < 0.)
    after = np.where(accept, trial, before)
    np.testing.assert_array_equal(record.wheel_target_rad_s, target)
    np.testing.assert_array_equal(record.wheel_integral_after_nm, after)
    np.testing.assert_array_equal(record.wheel_request_nm, 2.2*error+after)
    np.testing.assert_array_equal(record.wheel_torque_nm, np.clip(2.2*error+after, -12., 12.))
    assert bool(np.any(record.wheel_target_clipped)) == expected_target_clip
    assert np.max(np.abs(record.wheel_request_nm)) > 12.
    assert np.max(np.abs(record.wheel_torque_nm)) == JOINT_TORQUE_LIMIT[3] == 12.


def test_preserves_wheel_spin_feedback_and_restores_body_rate_slope(state):
    command = D1Command(yaw_rate_rps=.05)
    first = TurnYawAuthorityController()
    bind(first, state)
    first.compute(command, state, np.zeros(8))
    qd = state.joint_velocity.copy()
    increment = np.array([.1, -.1, .1, -.1])
    qd[3::4] += increment
    spin = replace(state, joint_velocity=qd)
    second = TurnYawAuthorityController()
    bind(second, spin)
    second.compute(command, spin, np.zeros(8))
    assert second.last_authority.wheel_target_rad_s.tobytes() == first.last_authority.wheel_target_rad_s.tobytes()
    np.testing.assert_allclose(second.last_authority.wheel_request_nm-first.last_authority.wheel_request_nm,
                               -(2.2+3.*.01)*increment, rtol=0, atol=1e-12)
    angular = state.base_angular_velocity_body+np.array([0., 0., 1e-6])
    moved = replace(state, base_angular_velocity_body=angular,
                    base_angular_velocity_world=state.base_rotation@angular)
    third = TurnYawAuthorityController()
    bind(third, moved)
    third.compute(command, moved, np.zeros(8))
    assert third.last_authority.effective_yaw_request_rps-first.last_authority.effective_yaw_request_rps == pytest.approx(-4e-6, abs=1e-14)


@pytest.mark.parametrize("action", [np.ones(8)*1e-300, np.zeros(8, dtype=bool),
    np.zeros(8, dtype=complex), np.full(8, np.nan), np.zeros(7), ["0"]*8])
def test_invalid_action_before_memory(state, action):
    candidate = TurnYawAuthorityController()
    bind(candidate, state)
    with pytest.raises((TypeError, ValueError)):
        candidate.compute(D1Command(yaw_rate_rps=.6), state, action)
    assert candidate.last_result is candidate.last_authority is None
    assert not np.any(candidate.control_memory.wheel_integral_nm)


def test_binding_time_and_next_preview_execution_snapshot(state):
    candidate = TurnYawAuthorityController()
    with pytest.raises(RuntimeError):
        candidate.nominal_targets(D1Command(), state)
    candidate.bind_raw_command(forward_velocity_mps=0., yaw_rate_rps=.6, control_time_s=1.)
    with pytest.raises(ValueError):
        candidate.compute(D1Command(), state, np.zeros(8))
    bind(candidate, state)
    with pytest.raises(ValueError):
        candidate.compute(D1Command(forward_velocity_mps=.1), state, np.zeros(8))
    assert candidate.last_result is None
    for tick, yaw in ((199, 0.), (200, .6), (249, .6), (250, 0.)):
        current = replace(state, control_time_s=tick*.01+1e-11, measurement_time_s=tick*.01)
        candidate.bind_raw_command(forward_velocity_mps=0., yaw_rate_rps=yaw, control_time_s=tick*.01)
        candidate.compute(D1Command(yaw_rate_rps=.8), current, np.zeros(8))
        record, memory = candidate.last_authority, candidate.control_memory.wheel_integral_nm.copy()
        assert record.active == (yaw != 0.) and record.control_time_s == tick*.01
        following = replace(current, control_time_s=(tick+1)*.01)
        bind(candidate, following, 0., 0.)
        candidate.nominal_targets(D1Command(yaw_rate_rps=.8), following)
        assert candidate.last_authority is record
        np.testing.assert_array_equal(candidate.control_memory.wheel_integral_nm, memory)


def test_disabled_unbound_parent_equivalence(state):
    candidate, baseline = TurnYawAuthorityController(enabled=False), D1WheelLegController()
    command = D1Command(yaw_rate_rps=.8)
    assert candidate.compute(command, state, np.zeros(8)).tobytes() == baseline.compute(command, state, np.zeros(8)).tobytes()
    assert candidate.control_schema == baseline.control_schema and not candidate.last_authority.active


def test_cap_diagnostic_stream_separates_actual_and_legacy_semantics(state):
    candidate = TurnYawAuthorityController()
    stream = StringIO()
    wrapper = _CapDiagnosticStream(stream, SimpleNamespace(_controller=SimpleNamespace(controller=candidate)))
    original = '{"inner_yaw_limit_occupied": true, "raw_zero": -0.0}\n'
    bind(candidate, state)
    candidate.compute(D1Command(yaw_rate_rps=.84), state, np.zeros(8))
    wrapper.write(original)
    active = json.loads(stream.getvalue())
    assert active['inner_yaw_limit_occupied'] is active['inner_cap_applied'] is False
    assert active['legacy_abs_effective_yaw_ge_0p6'] and active['original_inner_cap_would_be_occupied']
    stream.seek(0)
    stream.truncate(0)
    bind(candidate, state, 0., 0.)
    candidate.compute(D1Command(yaw_rate_rps=.12), state, np.zeros(8))
    assert candidate.last_authority.inner_cap_occupied
    wrapper.write(original)
    assert stream.getvalue() == original


@pytest.mark.parametrize("raw_yaw", [0., .6])
def test_raw_callback_identity_cache_and_preview_difference(tmp_path, raw_yaw):
    calls, raw = [], D1MotionCommand(yaw_rate_rps=raw_yaw)

    def source(time_s):
        calls.append(time_s)
        return raw

    env = TurnAuthorityEnv(command_source=source, diagnostic_output=tmp_path, episode_seconds=.01)
    baseline = D1FlatPlaneHeadingEnv(command_source=lambda t: raw, episode_seconds=.01)
    try:
        observation, info = env.reset(seed=55101)
        old, _ = baseline.reset(seed=55101)
        assert (observation.tobytes() == old.tobytes()) == (raw_yaw == 0.)
        assert env.heading_decision.user_command is raw and calls == [0.]
        env._prepare()
        env._prepare()
        assert calls == [0.] and env.raw_callback_count == 1
        assert env._controller.controller.last_authority is None
        assert info['episode_metadata']['heading_task_config']['task_schema'] == env.task_schema
        assert env.loop.controller is env._controller and env.loop.plant is env.plant
        assert env.plant.data.time == baseline.plant.data.time == 0.
    finally:
        env.close()
        baseline.close()


@pytest.mark.parametrize("source_type", [D1FlatPlaneHeadingEnv, PlaneStopDampingEnv, TurnCenterEnv, TurnDampingEnv])
@pytest.mark.parametrize("loader,match", [(load_heading_policy, "heading_task_config"),
                                         (load_locomotion_policy, "task_schema")])
def test_previous_checkpoint_rejected_before_deserialization(tmp_path, monkeypatch, source_type, loader, match):
    from stable_baselines3 import PPO

    def forbidden(*args, **kwargs):
        pytest.fail("incompatible checkpoint deserialized")

    monkeypatch.setattr(PPO, "load", forbidden)
    kwargs = {'diagnostic_output': None} if source_type is PlaneStopDampingEnv else {}
    if source_type in (TurnCenterEnv, TurnDampingEnv):
        source_folder = tmp_path/'source'
        source_folder.mkdir()
        kwargs = {'diagnostic_output': source_folder, 'command_source': lambda t: D1MotionCommand()}
    source = source_type(episode_seconds=.01, **kwargs)
    target = TurnAuthorityEnv(command_source=lambda t: D1MotionCommand(), diagnostic_output=tmp_path, episode_seconds=.01)
    try:
        source.reset(seed=55101)
        target.reset(seed=55101)
        sidecar = tmp_path/'old_checkpoint.json'
        sidecar.write_text(json.dumps({**_environment_contract(source),
            'recorded_episode': source.episode_metadata, 'model_sha256': 'not-read'}))
        with pytest.raises(ValueError, match=match):
            loader(tmp_path/'absent.zip', sidecar, target)
        assert source.plant.data.time == target.plant.data.time == 0.
    finally:
        source.close()
        target.close()
