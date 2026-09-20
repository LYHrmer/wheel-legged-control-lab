"""Nonintegrating mechanism composition and command-boundary checks."""
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from scripts import d1_stop_turn_composition as module
from scripts.d1_stop_leg_damping import StopLegDampingController
from scripts.d1_stop_turn_env import StopTurnEnv
from scripts.d1_turn_yaw_authority import TurnYawAuthorityController
from scripts.probe_d1_heading_turn_yaw_authority import TurnAuthorityEnv
from wheel_legged_control.d1.control_loop import D1MotionCommand
from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.model import D1Plant
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource
from wheel_legged_control.d1.wheel_leg_controller import D1WheelLegController


@pytest.fixture(autouse=True, scope="module")
def no_integration():
    def forbidden(*args, **kwargs):
        raise AssertionError("composition unit checks must not integrate")

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


def bind(controller, state, forward, yaw):
    controller.bind_raw_command(forward_velocity_mps=forward, yaw_rate_rps=yaw,
                                control_time_s=state.control_time_s)


@pytest.mark.parametrize("sign", [1., -1.])
def test_composition_matches_existing_components_all_modes(state, monkeypatch, sign):
    candidate = module.StopTurnCompositionController()
    stop, turn = StopLegDampingController(), TurnYawAuthorityController()
    sequence = [(0., 0., False), (0., sign*.6, False), (0., 0., False),
                (sign*.25, 0., False), (-0., 0., True), (0., sign*.6, True),
                (0., 0., True), (-sign*.25, 0., False), (0., -sign*.6, True)]
    original, calls = D1WheelLegController.compute, []

    def parent(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if self is candidate:
            calls.append(result)
        return result

    monkeypatch.setattr(D1WheelLegController, "compute", parent)
    seen = set()
    for k, (forward, yaw, stop_active) in enumerate(sequence):
        sample = replace(state, sequence=k, control_time_s=k*.01, measurement_time_s=k*.01)
        command = D1Command(forward_velocity_mps=forward, yaw_rate_rps=1.4*yaw+.01)
        for controller in (candidate, turn):
            bind(controller, sample, forward, yaw)
        previous = candidate.previous_forward_mps
        memory = candidate.control_memory.wheel_integral_nm.copy()
        old_record = candidate.last_damping
        for _ in range(3):
            preview = candidate.nominal_targets(command, sample)
            expected = turn.nominal_targets(command, sample)
            assert preview.nominal_wheel_speed_rad_s.tobytes() == expected.nominal_wheel_speed_rad_s.tobytes()
        assert candidate.previous_forward_mps == previous and candidate.last_damping is old_record
        assert candidate.control_memory.wheel_integral_nm.tobytes() == memory.tobytes()
        actual = candidate.compute(command, sample, np.zeros(8))
        stopped = stop.compute(command, sample, np.zeros(8))
        turned = turn.compute(command, sample, np.zeros(8))
        assert len(calls) == k+1
        assert actual.reshape(4, 4)[:, :3].tobytes() == stopped.reshape(4, 4)[:, :3].tobytes()
        assert actual[3::4].tobytes() == turned[3::4].tobytes()
        assert candidate.control_memory.wheel_integral_nm.tobytes() == turn.control_memory.wheel_integral_nm.tobytes()
        assert candidate.last_authority.wheel_torque_nm.tobytes() == actual[3::4].tobytes()
        assert candidate.last_damping.active == candidate.stop_damping_active == stop_active
        assert candidate.last_damping.delta_torque_nm.tobytes() == stop.last_damping.delta_torque_nm.tobytes()
        assert candidate.last_damping.joint_power_w == stop.last_damping.joint_power_w
        assert candidate.last_damping.joint_power_w <= 0.
        assert candidate.previous_forward_mps == forward
        seen.add((stop_active, candidate.last_authority.active))
        if not stop_active:
            assert actual is calls[-1]
            assert actual.tobytes() == turned.tobytes()
    assert seen == {(False, False), (False, True), (True, False), (True, True)}
    candidate.reset()
    assert candidate.last_authority is candidate.raw_binding is candidate.last_damping is None
    assert candidate.previous_forward_mps is None and not candidate.stop_damping_active
    assert not np.any(candidate.control_memory.wheel_integral_nm)


@pytest.mark.parametrize("bad_action", [np.ones(8), np.zeros(7), np.full(8, np.nan),
                                         np.zeros(8, dtype=bool), ["0"]*8, np.zeros(8, dtype=complex)])
def test_invalid_action_does_not_advance_either_mechanism(state, bad_action):
    candidate = module.StopTurnCompositionController()
    bind(candidate, state, .25, 0.)
    candidate.compute(D1Command(forward_velocity_mps=.25), state, np.zeros(8))
    before = (candidate.last_result, candidate.last_authority, candidate.last_damping)
    memory = candidate.control_memory.wheel_integral_nm.copy()
    bind(candidate, state, 0., .6)
    with pytest.raises((ValueError, TypeError)):
        candidate.compute(D1Command(yaw_rate_rps=.84), state, bad_action)
    assert all(a is b for a, b in zip(before, (candidate.last_result, candidate.last_authority, candidate.last_damping)))
    assert candidate.previous_forward_mps == .25 and not candidate.stop_damping_active
    assert candidate.control_memory.wheel_integral_nm.tobytes() == memory.tobytes()


@pytest.mark.parametrize("kind", ["unbound", "stale", "forward_mismatch"])
def test_binding_failure_precedes_latch_and_PI(state, kind):
    candidate = module.StopTurnCompositionController()
    if kind != "unbound":
        candidate.bind_raw_command(forward_velocity_mps=.25 if kind == "forward_mismatch" else 0.,
                                    yaw_rate_rps=.6, control_time_s=1. if kind == "stale" else 0.)
    with pytest.raises((ValueError, RuntimeError)):
        candidate.compute(D1Command(yaw_rate_rps=.84), state, np.zeros(8))
    assert candidate.previous_forward_mps is None and candidate.last_damping is None
    assert candidate.last_authority is candidate.last_result is None
    assert not np.any(candidate.control_memory.wheel_integral_nm)


def test_disabled_schema_and_exact_original_dynamics(state):
    candidate, baseline = module.StopTurnCompositionController(enabled=False), D1WheelLegController()
    assert candidate.control_schema == baseline.control_schema
    for forward, yaw in ((.25, 0.), (0., .84), (0., 0.), (-.25, -.84)):
        command = D1Command(forward_velocity_mps=forward, yaw_rate_rps=yaw)
        actual = candidate.compute(command, state, np.zeros(8))
        expected = baseline.compute(command, state, np.zeros(8))
        assert actual.tobytes() == expected.tobytes()
        assert not candidate.last_damping.active and not candidate.last_authority.active


def test_post_parent_failure_is_not_claimed_atomic(state, monkeypatch):
    candidate = module.StopTurnCompositionController()
    bind(candidate, state, .25, 0.)
    candidate.compute(D1Command(forward_velocity_mps=.25), state, np.zeros(8))
    old_authority, old_stop = candidate.last_authority, candidate.last_damping
    bind(candidate, state, 0., .6)

    def fail_record(*args, **kwargs):
        raise RuntimeError("injected stop record failure after inherited compute")

    monkeypatch.setattr(module, "StopLegDampingRecord", fail_record)
    with pytest.raises(RuntimeError, match="injected stop record"):
        candidate.compute(D1Command(yaw_rate_rps=.84), state, np.zeros(8))
    assert candidate.last_authority is not old_authority
    assert candidate.last_damping is old_stop and candidate.previous_forward_mps == .25
    assert not candidate.stop_damping_active
    # No retry: an outer runner must terminate and archive this partially advanced call.


@pytest.mark.parametrize("yaw", [0., .6])
def test_environment_initial_preview_matches_authority_with_one_callback(tmp_path, yaw):
    command = D1MotionCommand(yaw_rate_rps=yaw)
    candidate_path, baseline_path = tmp_path/"candidate", tmp_path/"authority"
    candidate_path.mkdir()
    baseline_path.mkdir()
    callbacks = []

    def source(time_s):
        callbacks.append(time_s)
        return command

    candidate = StopTurnEnv(command_source=source, diagnostic_output=candidate_path)
    baseline = TurnAuthorityEnv(command_source=lambda _: command, diagnostic_output=baseline_path)
    try:
        observation, _ = candidate.reset(seed=103)
        other, _ = baseline.reset(seed=103)
        assert observation.tobytes() == other.tobytes()
        assert callbacks == [0.] and candidate.raw_callback_count == 1
        assert candidate._controller.controller.previous_forward_mps is None
        assert candidate._controller.controller.last_authority is None
        assert candidate.heading_decision.user_command is command
        assert candidate.plant.data.time == baseline.plant.data.time == 0.
        assert candidate.heading_task_config != baseline.heading_task_config
    finally:
        candidate.close()
        baseline.close()
    candidate.close()
