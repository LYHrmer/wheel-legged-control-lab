"""Policy dimension changes must leave the physical D1 task unchanged."""

import json
from dataclasses import fields

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from wheel_legged_control.d1.actuator_channel import ActuatorChannelConfig
from wheel_legged_control.d1.locomotion_checkpoint import (
    load_locomotion_policy,
    write_checkpoint_metadata,
)
from wheel_legged_control.d1.locomotion_env import D1LocomotionEnv
from wheel_legged_control.d1.model import JOINT_TORQUE_LIMIT
from wheel_legged_control.d1.observation_history import D1ObservationHistory
from wheel_legged_control.d1.state_provider import D1StateProviderConfig

SHARED_SCHEMA = "d1-shared-wheel-leg-extension-speed-v1"
PHYSICAL_SCHEMA = "d1-wheel-leg-extension-speed-v1"
ACTION_FIELDS = (
    "action_mode",
    "policy_action_size",
    "physical_action_size",
    "physical_action_schema",
    "policy_to_physical_indices",
)


def make_env(mode=None, *, delayed=False, baseline="wheel_leg"):
    options = {}
    if delayed:
        options = {
            "provider_config": D1StateProviderConfig(
                "imu_encoder_fusion",
                sensor_delay_steps=2,
                initial_position_m=(-3.8, 0.0, 0.455),
                initial_rpy_rad=(0.0, 0.0, 0.0),
            ),
            "actuator_config": ActuatorChannelConfig(
                torque_limit_nm=tuple(JOINT_TORQUE_LIMIT),
                delay_steps=2,
                time_constant_s=0.004,
            ),
        }
    return D1LocomotionEnv(
        baseline=baseline,
        action_mode=mode,
        episode_seconds=0.12,
        command_mode="development",
        **options,
    )


def assert_memory_equal(first, second):
    for field in fields(first):
        left, right = getattr(first, field.name), getattr(second, field.name)
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right


def assert_physical_equal(first, second, left, right):
    np.testing.assert_array_equal(left[0], right[0])
    assert left[0].shape == (82,)
    assert left[1:4] == right[1:4]
    for key in ("reward_terms", "metrics", "command", "terrain_exposure", "terminal_reason"):
        assert left[4][key] == right[4][key]
    for key in ("raw_action", "applied_action"):
        np.testing.assert_array_equal(left[4][key], right[4][key])
    np.testing.assert_array_equal(first.plant.data.qpos, second.plant.data.qpos)
    np.testing.assert_array_equal(first.plant.data.qvel, second.plant.data.qvel)
    np.testing.assert_array_equal(
        first.last_transition.requested_torque_nm, second.last_transition.requested_torque_nm
    )
    np.testing.assert_array_equal(
        first.decision.context.previous_normalized_action,
        second.decision.context.previous_normalized_action,
    )
    assert_memory_equal(
        first.decision.context.proposal.memory, second.decision.context.proposal.memory
    )
    for ltrace, rtrace in zip(
        first.plant.last_control_interval_actuator_traces,
        second.plant.last_control_interval_actuator_traces,
        strict=True,
    ):
        for field in fields(ltrace):
            np.testing.assert_array_equal(getattr(ltrace, field.name), getattr(rtrace, field.name))


@pytest.mark.parametrize("delayed", [False, True])
@pytest.mark.parametrize("zero", [False, True])
def test_shared_broadcast_matches_independent_physics_and_reset(delayed, zero):
    shared, independent = (
        make_env("shared2", delayed=delayed),
        make_env("independent8", delayed=delayed),
    )
    try:
        first, _ = shared.reset(seed=19)
        other, _ = independent.reset(seed=19)
        np.testing.assert_array_equal(first, other)
        actions = (
            [(0.0, 0.0)] * 5
            if zero
            else [(0.1, -0.2), (-0.3, 0.4), (2.0, -3.0), (0.0, 0.1), (0.2, 0.0)]
        )
        for values in actions:
            raw = np.asarray(values)
            broadcast = np.array([values[0]] * 4 + [values[1]] * 4)
            left, right = shared.step(raw), independent.step(broadcast)
            assert_physical_equal(shared, independent, left, right)
            np.testing.assert_array_equal(left[4]["raw_action"], broadcast)
            np.testing.assert_array_equal(
                left[0][-8:], np.clip(broadcast, -1, 1).astype(np.float32)
            )
        if delayed:
            assert left[4]["metrics"]["measurement_age_s"] == pytest.approx(0.02)
        reset_shared, _ = shared.reset(seed=19)
        reset_independent, _ = independent.reset(seed=19)
        np.testing.assert_array_equal(reset_shared, first)
        np.testing.assert_array_equal(reset_independent, other)
        assert_physical_equal(
            shared, independent, shared.step([0, 0]), independent.step(np.zeros(8))
        )
    finally:
        shared.close()
        independent.close()


def test_default_and_explicit_independent_actions_are_exactly_equal():
    default, explicit = make_env(), make_env("independent8")
    try:
        np.testing.assert_array_equal(default.reset(seed=23)[0], explicit.reset(seed=23)[0])
        assert default.episode_metadata == explicit.episode_metadata
        for i in range(5):
            action = np.linspace(-0.4, 0.5, 8) * (-1) ** i
            assert_physical_equal(default, explicit, default.step(action), explicit.step(action))
    finally:
        default.close()
        explicit.close()


@pytest.mark.parametrize(
    "baseline,mode,size,resolved",
    [
        ("wheel_leg", "shared2", 2, "shared2"),
        ("wheel_leg", "independent8", 8, "independent8"),
        ("wheel_leg", None, 8, "independent8"),
        ("lqr", None, 2, "legacy_force2"),
        ("mpc", None, 2, "legacy_force2"),
    ],
)
def test_public_mapping_and_episode_metadata(baseline, mode, size, resolved):
    env = make_env(mode, baseline=baseline)
    try:
        _, info = env.reset(seed=7)
        physical = 8 if baseline == "wheel_leg" else 2
        indices = (0, 0, 0, 0, 1, 1, 1, 1) if resolved == "shared2" else tuple(range(physical))
        schema = PHYSICAL_SCHEMA if physical == 8 else "d1-v3-body-force-residual-v1"
        assert env.action_mode == resolved
        assert env.policy_action_size == size
        assert env.physical_action_size == physical
        assert env.physical_action_schema == schema
        assert env.policy_to_physical_indices == indices
        assert isinstance(env.policy_to_physical_indices, tuple)
        assert env.action_space.shape == (size,)
        assert env.action_schema == (SHARED_SCHEMA if resolved == "shared2" else schema)
        for name in ACTION_FIELDS:
            expected = list(indices) if name == "policy_to_physical_indices" else getattr(env, name)
            assert info["episode_metadata"][name] == expected
        assert env.decision.context.action_schema == schema
        assert env.decision.context.action_size == physical
    finally:
        env.close()


@pytest.mark.parametrize(
    "baseline,mode",
    [
        ("lqr", "shared2"),
        ("lqr", "independent8"),
        ("mpc", "shared2"),
        ("mpc", "independent8"),
        ("wheel_leg", "legacy_force2"),
        ("wheel_leg", ""),
        ("wheel_leg", True),
        ("wheel_leg", 2),
    ],
)
def test_unsupported_mapping_rejected(baseline, mode):
    with pytest.raises(ValueError, match="action_mode"):
        make_env(mode, baseline=baseline)


@pytest.mark.parametrize("mode", [None, "independent8", "shared2"])
@pytest.mark.parametrize("bad_kind", ["shape", "nan", "inf"])
def test_invalid_policy_keeps_physics_decision_and_memory_but_requires_reset(mode, bad_kind):
    env = make_env(mode)
    try:
        original, _ = env.reset(seed=7)
        decision = env.decision
        qpos, qvel = env.plant.data.qpos.copy(), env.plant.data.qvel.copy()
        bad = np.zeros(env.policy_action_size + (bad_kind == "shape"))
        if bad_kind != "shape":
            bad[0] = float(bad_kind)
        with pytest.raises(ValueError, match="finite action schema"):
            env.step(bad)
        assert env.plant.data.time == 0
        assert env.decision is decision
        assert env.loop.prepare(decision.motion_command) is decision
        assert env.last_transition is None
        np.testing.assert_array_equal(qpos, env.plant.data.qpos)
        np.testing.assert_array_equal(qvel, env.plant.data.qvel)
        with pytest.raises(RuntimeError, match="reset"):
            env.step(np.zeros(env.policy_action_size))
        np.testing.assert_array_equal(env.reset(seed=7)[0], original)
        env.step(np.zeros(env.policy_action_size))
    finally:
        env.close()


def test_policy_info_is_independent_of_input_and_physical_info_arrays():
    env = make_env("shared2")
    try:
        env.reset(seed=5)
        action = np.array([2.0, -0.3], dtype=np.float64)
        _, _, _, _, info = env.step(action)
        action[:] = 0
        np.testing.assert_array_equal(info["policy_action"], [2.0, -0.3])
        np.testing.assert_array_equal(info["raw_action"], [2.0] * 4 + [-0.3] * 4)
        np.testing.assert_array_equal(info["applied_action"], [1.0] * 4 + [-0.3] * 4)
        info["policy_action"][:] = -7
        info["raw_action"][:] = 8
        info["applied_action"][:] = 9
        np.testing.assert_array_equal(env.last_transition.raw_action, [2.0] * 4 + [-0.3] * 4)
        np.testing.assert_array_equal(
            env.decision.context.previous_normalized_action, [1.0] * 4 + [-0.3] * 4
        )
    finally:
        env.close()


@pytest.fixture(scope="module")
def real_models(tmp_path_factory):
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("shared-real-ppo")
    paths = {}
    for label, mode, baseline in [
        ("shared2", "shared2", "wheel_leg"),
        ("independent8", None, "wheel_leg"),
        ("legacy_force2", None, "lqr"),
    ]:
        env = make_env(mode, baseline=baseline)
        try:
            env.reset(seed=11)
            model = PPO(
                "MlpPolicy", env, n_steps=8, batch_size=4, n_epochs=1, seed=13, device="cpu"
            )
            if label == "shared2":
                model.learn(total_timesteps=8)
            path = root / f"{label}.zip"
            model.save(path)
            paths[label] = path
        finally:
            env.close()
    return paths


@pytest.mark.parametrize(
    "mode,baseline,label",
    [
        ("shared2", "wheel_leg", "shared2"),
        (None, "wheel_leg", "independent8"),
        (None, "lqr", "legacy_force2"),
    ],
)
def test_real_checkpoint_roundtrip_and_legacy_contract_omission(
    tmp_path, real_models, mode, baseline, label
):
    env = make_env(mode, baseline=baseline)
    try:
        obs, _ = env.reset(seed=11)
        path, sidecar = real_models[label], tmp_path / "model.json"
        record = write_checkpoint_metadata(path, env, sidecar)
        for name in ACTION_FIELDS:
            assert record[name] == env.episode_metadata[name]
        loaded = load_locomotion_policy(path, sidecar, env)
        prediction, _ = loaded.predict(obs, deterministic=True)
        assert prediction.shape == env.action_space.shape
        if mode != "shared2":
            for name in ACTION_FIELDS:
                record.pop(name)
                record["recorded_episode"].pop(name)
            sidecar.write_text(json.dumps(record))
            legacy = load_locomotion_policy(path, sidecar, env)
            np.testing.assert_array_equal(legacy.predict(obs, deterministic=True)[0], prediction)
    finally:
        env.close()


def test_real_shared_ppo_has_two_gaussians_and_two_dimensional_log_probability(real_models):
    model = PPO.load(real_models["shared2"], device="cpu")
    observations = torch.zeros((3, 82))
    actions = torch.tensor([[0.1, -0.4], [0.2, 0.3], [-0.6, 0.5]])
    distribution = model.policy.get_distribution(observations)
    assert distribution.distribution.mean.shape == (3, 2)
    per_dimension = distribution.distribution.log_prob(actions)
    torch.testing.assert_close(
        distribution.log_prob(actions), per_dimension[:, 0] + per_dimension[:, 1]
    )
    assert model.policy.action_net.out_features == 2
    assert model.num_timesteps == 8


@pytest.mark.parametrize(
    "source,target",
    [
        ("shared2", "independent8"),
        ("independent8", "shared2"),
        ("shared2", "legacy_force2"),
        ("legacy_force2", "shared2"),
        ("independent8", "legacy_force2"),
        ("legacy_force2", "independent8"),
    ],
)
def test_cross_mode_sidecar_rejected_before_deserializing(tmp_path, source, target):
    first = make_env(
        None if source == "legacy_force2" else source,
        baseline="lqr" if source == "legacy_force2" else "wheel_leg",
    )
    second = make_env(
        None if target == "legacy_force2" else target,
        baseline="lqr" if target == "legacy_force2" else "wheel_leg",
    )
    try:
        first.reset(seed=1)
        second.reset(seed=1)
        path, sidecar = tmp_path / "invalid.zip", tmp_path / "model.json"
        path.write_bytes(b"invalid model must not reach deserialization")
        write_checkpoint_metadata(path, first, sidecar)
        with pytest.raises(ValueError, match="checkpoint"):
            load_locomotion_policy(path, sidecar, second)
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("mode", ["shared2", "independent8"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("action_mode", "wrong"),
        ("policy_action_size", True),
        ("physical_action_size", 8.0),
        ("physical_action_schema", "wrong"),
        ("policy_to_physical_indices", [False] * 8),
        ("policy_to_physical_indices", [0.0] * 8),
        ("policy_to_physical_indices", [1] * 8),
        ("policy_to_physical_indices", [0, 1]),
    ],
)
def test_corrupted_new_mapping_fields_fail_before_model_load(tmp_path, mode, field, value):
    env = make_env(mode)
    try:
        env.reset(seed=1)
        path, sidecar = tmp_path / "invalid.zip", tmp_path / "model.json"
        path.write_bytes(b"invalid model must not reach deserialization")
        record = write_checkpoint_metadata(path, env, sidecar)
        record[field] = value
        sidecar.write_text(json.dumps(record))
        with pytest.raises(ValueError, match="checkpoint"):
            load_locomotion_policy(path, sidecar, env)
    finally:
        env.close()


@pytest.mark.parametrize("remove", [*ACTION_FIELDS, "all"])
def test_shared_checkpoint_requires_complete_explicit_mapping(tmp_path, remove):
    env = make_env("shared2")
    try:
        env.reset(seed=1)
        path, sidecar = tmp_path / "invalid.zip", tmp_path / "model.json"
        path.write_bytes(b"invalid model must not reach deserialization")
        record = write_checkpoint_metadata(path, env, sidecar)
        for field in ACTION_FIELDS if remove == "all" else [remove]:
            record.pop(field)
        sidecar.write_text(json.dumps(record))
        with pytest.raises(ValueError, match="checkpoint"):
            load_locomotion_policy(path, sidecar, env)
    finally:
        env.close()


def test_real_model_eight_action_space_cannot_hide_behind_shared_sidecar(tmp_path, real_models):
    env = make_env("shared2")
    try:
        env.reset(seed=3)
        sidecar = tmp_path / "model.json"
        write_checkpoint_metadata(real_models["independent8"], env, sidecar)
        with pytest.raises(ValueError, match="saved model action_space"):
            load_locomotion_policy(real_models["independent8"], sidecar, env)
    finally:
        env.close()


@pytest.mark.parametrize("missing_field", ACTION_FIELDS)
def test_legacy_sidecar_with_partial_new_contract_is_rejected(tmp_path, missing_field):
    env = make_env("independent8")
    try:
        env.reset(seed=3)
        path, sidecar = tmp_path / "invalid.zip", tmp_path / "model.json"
        path.write_bytes(b"invalid model must not reach deserialization")
        metadata = write_checkpoint_metadata(path, env, sidecar)
        metadata.pop(missing_field)
        sidecar.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="partial action contract"):
            load_locomotion_policy(path, sidecar, env)
    finally:
        env.close()


def test_shared_history_wrapper_keeps_physical_history_and_roundtrips(tmp_path):
    torch.set_num_threads(1)
    env = D1ObservationHistory(make_env("shared2"), 2)
    try:
        env.reset(seed=4)
        observation = env.step([0.2, -0.1])[0]
        np.testing.assert_array_equal(
            observation[-8:], np.array([0.2] * 4 + [-0.1] * 4, dtype=np.float32)
        )
        model = PPO("MlpPolicy", env, n_steps=2, batch_size=2, device="cpu")
        path, sidecar = tmp_path / "model.zip", tmp_path / "model.json"
        model.save(path)
        metadata = write_checkpoint_metadata(path, env, sidecar)
        assert metadata["history_length"] == 2
        assert metadata["observation_dim"] == 164
        assert load_locomotion_policy(path, sidecar, env).predict(observation)[0].shape == (2,)
    finally:
        env.close()
