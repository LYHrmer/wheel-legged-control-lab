import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wheel_legged_control.d1.policy_symmetry43 import (
    SymmetricActor,
    mirror_action,
    mirror_observation,
)

ROOT = Path(__file__).resolve().parents[1]


def observation():
    result = np.arange(99, dtype=np.float32) / 100
    return result


def test_reflection_is_involution_preserves_norm_and_does_not_mutate():
    rng = np.random.default_rng(431)
    for _ in range(10):
        obs = rng.uniform(-1, 1, 99).astype(np.float32)
        action = rng.uniform(-1, 1, 16).astype(np.float32)
        for x, reflect in ((obs, mirror_observation), (action, mirror_action)):
            before = x.copy()
            transformed = reflect(x)
            assert np.array_equal(reflect(transformed), x)
            assert np.array_equal(np.sort(abs(transformed)), np.sort(abs(x)))
            assert np.array_equal(x, before) and not np.shares_memory(x, transformed)


def test_semantic_fields_polar_axial_commands_scan_and_action_layout():
    obs = observation()
    actual = mirror_observation(obs)
    assert np.array_equal(actual[:10], obs[:10] * [1, -1, 1, -1, 1, -1, 1, -1, 1, 1])
    assert np.array_equal(actual[38:43], obs[38:43] * [1, -1, -1, 1, 1])
    assert np.array_equal(actual[22:26], obs[26:30] * [-1, 1, 1, 1])
    assert np.array_equal(actual[55:59], obs[[56, 55, 58, 57]])
    assert np.array_equal(actual[59:63], obs[[60, 59, 62, 61]])
    assert np.array_equal(actual[79:83], obs[[80, 79, 82, 81]])
    assert np.array_equal(actual[83:86], obs[[85, 84, 83]])
    assert actual[98] == obs[98]
    action = np.arange(16, dtype=np.float32)/16
    reflected = mirror_action(action)
    assert np.array_equal(reflected[:6], action[[3, 4, 5, 0, 1, 2]] * [-1, 1, 1, -1, 1, 1])
    assert np.array_equal(reflected[12:], action[[13, 12, 15, 14]])


def test_projection_is_exactly_equivariant_for_asymmetric_nonlinear_actor():
    rng = np.random.default_rng(432)
    weights = rng.normal(0, .1, (16, 99)).astype(np.float32)
    bias = rng.normal(0, .1, 16).astype(np.float32)
    actor = SymmetricActor(lambda x: np.tanh(weights @ x + bias).astype(np.float32))
    obs = observation()
    assert np.array_equal(actor(mirror_observation(obs)), mirror_action(actor(obs)))
    assert actor.predict_calls == 2 and actor.actor_attempts == actor.actor_returns == 4


def test_symmetric_state_cancels_differential_bias_and_retains_common_drive():
    biased = np.zeros(16, np.float32)
    biased[12:] = [.2, -.4, .3, -.1]
    result = SymmetricActor(lambda _: biased)(np.zeros(99, np.float32))
    assert result[12] == result[13] and result[14] == result[15]
    assert float(result[12:].mean()) == pytest.approx(float(biased[12:].mean()), abs=1e-8)


@pytest.mark.parametrize('bad', [np.zeros(98, np.float32), np.zeros(99, np.float64),
                               np.full(99, np.nan, np.float32), np.full(99, 6, np.float32)])
def test_invalid_observation_rejected_before_actor(bad):
    calls = []
    actor = SymmetricActor(lambda x: calls.append(x))
    with pytest.raises(ValueError):
        actor(bad)
    assert not calls and actor.actor_attempts == 0


def test_failed_second_actor_does_not_emit_action_or_hide_attempt():
    calls = []
    def predict(x):
        calls.append(x)
        if len(calls) == 2:
            raise RuntimeError('synthetic actor failure')
        return np.zeros(16, np.float32)
    actor = SymmetricActor(predict)
    with pytest.raises(RuntimeError):
        actor(observation())
    assert actor.actor_attempts == 2 and actor.actor_returns == 1


def test_mapping_matches_original_encoder_on_independently_reflected_state(monkeypatch):
    # Import only pure course modules, no environment/engine construction.
    folder = ROOT/'runtime/d1_b22/research/course_impl08'
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location('symmetry43_encoder', folder/'full_drive_observation_08.py')
    enc = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, enc)
    spec.loader.exec_module(enc)
    swap = [1, 0, 3, 2]
    signs = np.array([-1, 1, 1, 1])
    rng = np.random.default_rng(433)
    q = np.tile([0., .8, -1.5, 0.], 4) + rng.uniform(-.2, .2, 16)
    qd, nominal = rng.uniform(-1, 1, 16), q + .1
    previous = rng.uniform(-1, 1, 16).astype(np.float32)
    def joint(x):
        return (x.reshape(4, 4)[swap]*signs).reshape(16)
    def decision(mirror):
        state = SimpleNamespace(sequence=1, control_time_s=.01, age_s=.002,
            base_position=np.array([0., 0., .46]), base_rpy=np.zeros(3),
            joint_position=joint(q) if mirror else q,
            joint_velocity=joint(qd) if mirror else qd,
            wheel_contact=np.array([True, False, False, True])[swap if mirror else [0, 1, 2, 3]],
            base_linear_velocity_body=np.array([.2, -.03 if mirror else .03, .01]),
            base_angular_velocity_body=np.array([-.04 if mirror else .04, .05, -.1 if mirror else .1]),
            projected_gravity_body=np.array([.02, -.03 if mirror else .03, -.99]))
        baseline = SimpleNamespace(nominal_joint_target_rad=joint(nominal) if mirror else nominal,
            nominal_wheel_speed_rad_s=np.array([1., 2., 3., 4.])[swap if mirror else [0, 1, 2, 3]])
        proposal = SimpleNamespace(tick=1, control_time_s=.01, baseline=baseline,
            memory=SimpleNamespace(wheel_integral_nm=np.array([.1, .2, .3, .4])[swap if mirror else [0, 1, 2, 3]]))
        context = SimpleNamespace(action_size=16, action_schema=enc.ACTION_SCHEMA, state=state,
            proposal=proposal, previous_normalized_action=mirror_action(previous) if mirror else previous)
        return SimpleNamespace(context=context, world_command=SimpleNamespace(base_height_m=.455),
            motion_command=SimpleNamespace(lateral_velocity_mps=0., jump_requested=False,
                forward_velocity_mps=.6, yaw_rate_rps=-.2 if mirror else .2, clearance_m=.455))
    heights = np.arange(15, dtype=np.float64)*.001
    def scan(mirror):
        values = heights.reshape(5, 3)[:, ::-1].reshape(15) if mirror else heights
        return enc.CourseScan(0., 0., 0., 0., 0, values, values, tuple(range(15)))
    a = enc.encode_full_drive_observation(decision(False), scan(False)).values
    b = enc.encode_full_drive_observation(decision(True), scan(True)).values
    assert np.array_equal(mirror_observation(a), b)
