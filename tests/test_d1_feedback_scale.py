"""Configurable leg/attitude feedback bandwidth is versioned, not retroactive.

These tests use the real plant, the real locomotion environment and a fake SB3
loader, so nothing here depends on stable-baselines3 being installed. Scaling a
feedback pair is a diagnostic knob; none of the numbers below are a tuning or
sim2real claim.
"""

import json
import math
import sys
import types
from dataclasses import asdict, replace

import numpy as np
import pytest

pytest.importorskip("mujoco")
from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.locomotion_checkpoint import (
    load_locomotion_policy,
    write_checkpoint_metadata,
)
from wheel_legged_control.d1.locomotion_env import D1LocomotionEnv
from wheel_legged_control.d1.model import NOMINAL_JOINT_POSITION, D1Plant
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource
from wheel_legged_control.d1.wheel_leg_controller import (
    LEG_INDICES,
    NOMINAL_LEG_KD,
    NOMINAL_LEG_KP,
    WHEEL_INDICES,
    D1WheelLegControlConfig,
    D1WheelLegController,
)

SCALES = ("leg_feedback_scale", "attitude_feedback_scale")
SCHEMA_V2 = "d1-wheel-leg-joint-pd-wheel-pi-yaw-feedback-v2"
SCHEMA_V3 = "d1-wheel-leg-joint-pd-wheel-pi-yaw-feedback-configured-v3"
SCHEMA_V4 = "d1-wheel-leg-joint-pd-wheel-pi-yaw-feedback-configured-v4"
OLD_GAINS = ("wheel_kp", "wheel_ki", "yaw_feedback_gain")
DEFAULT_RECORD = {
    "wheel_kp": 2.2,
    "wheel_ki": 3.0,
    "yaw_feedback_gain": 4.0,
    "leg_feedback_scale": 1.0,
    "attitude_feedback_scale": 1.0,
}


@pytest.fixture(scope="module")
def perturbed_state():
    """A real tilted, moving plant state so every feedback term is nonzero."""
    plant = D1Plant(sampling_mode="synchronized")
    joints = NOMINAL_JOINT_POSITION.copy()
    joints[LEG_INDICES] += 0.03
    velocity = np.zeros(16)
    velocity[LEG_INDICES] = 0.2
    velocity[WHEEL_INDICES] = 1.5
    plant.reset(
        base_quaternion=(math.cos(0.03), 0.0, math.sin(0.03), 0.0),
        joint_position=joints,
        joint_velocity=velocity,
    )
    plant.data.qvel[3:6] = (0.2, -0.3, 0.1)
    plant.refresh_measurements()
    state = D1MujocoTruthStateSource(plant).reset()
    assert abs(state.base_rpy[1]) > 0.05
    assert np.abs(state.base_angular_velocity_world[:2]).min() > 0.1
    return state


def _torques(state, **gains):
    controller = D1WheelLegController(control_dt=0.01, **gains)
    controller.compute(D1Command(forward_velocity_mps=0.25), state, np.zeros(8))
    return controller, controller.last_result


@pytest.mark.parametrize("field", SCALES)
@pytest.mark.parametrize(
    "value",
    (True, np.bool_(True), np.array([0.25]), np.zeros(2), float("nan"), float("inf"), 0, -0.25),
)
def test_invalid_feedback_scale_is_rejected(field, value):
    with pytest.raises((TypeError, ValueError)):
        D1WheelLegControlConfig(**{field: value})
    with pytest.raises((TypeError, ValueError)):
        D1WheelLegController(**{field: value})


@pytest.mark.parametrize("field", SCALES)
def test_positive_real_scales_are_accepted_as_floats(field):
    config = D1WheelLegControlConfig(**{field: np.float32(0.25)})
    value = getattr(config, field)
    assert type(value) is float and value == 0.25
    assert asdict(config)[field] == 0.25


def test_defaults_keep_historical_gains_schema_and_five_field_record():
    config = D1WheelLegControlConfig()
    assert config.leg_feedback_scale == config.attitude_feedback_scale == 1.0
    assert not config.feedback_scaled
    assert list(asdict(config)) == [*OLD_GAINS, *SCALES]
    controller = D1WheelLegController()
    assert controller.control_schema == SCHEMA_V2
    np.testing.assert_array_equal(controller.leg_kp, NOMINAL_LEG_KP)
    np.testing.assert_array_equal(controller.leg_kd, NOMINAL_LEG_KD)
    assert (controller.roll_pitch_kp, controller.roll_pitch_kd) == (180.0, 24.0)


@pytest.mark.parametrize(
    "gains,schema",
    (
        ({}, SCHEMA_V2),
        ({"wheel_kp": 0.55, "wheel_ki": 1.5}, SCHEMA_V3),
        ({"yaw_feedback_gain": 0.0}, SCHEMA_V3),
        ({"leg_feedback_scale": 0.25}, SCHEMA_V4),
        ({"attitude_feedback_scale": 0.25}, SCHEMA_V4),
        ({"wheel_kp": 0.55, "attitude_feedback_scale": 2.0}, SCHEMA_V4),
        ({"leg_feedback_scale": 1.0, "attitude_feedback_scale": 1.0}, SCHEMA_V2),
    ),
)
def test_schema_records_whether_feedback_was_reconfigured(gains, schema):
    assert D1WheelLegController(**gains).control_schema == schema


def test_leg_scale_moves_the_pd_pair_only(perturbed_state):
    reference, base = _torques(perturbed_state)
    scaled, result = _torques(perturbed_state, leg_feedback_scale=0.25)
    np.testing.assert_allclose(scaled.leg_kp, 0.25 * NOMINAL_LEG_KP, rtol=0, atol=0)
    np.testing.assert_allclose(scaled.leg_kd, 0.25 * NOMINAL_LEG_KD, rtol=0, atol=0)
    assert (scaled.roll_pitch_kp, scaled.roll_pitch_kd) == (180.0, 24.0)
    assert (scaled.wheel_kp, scaled.wheel_ki, scaled.yaw_feedback_gain) == (
        reference.wheel_kp,
        reference.wheel_ki,
        reference.yaw_feedback_gain,
    )
    np.testing.assert_allclose(result.leg_pd_nm, 0.25 * base.leg_pd_nm, rtol=0, atol=1e-12)
    assert np.abs(base.leg_pd_nm[LEG_INDICES]).min() > 0.1
    # Gravity support, the wheel PI law and the IK targets are untouched.
    np.testing.assert_array_equal(result.support_nm, base.support_nm)
    np.testing.assert_array_equal(result.support_force_n, base.support_force_n)
    np.testing.assert_array_equal(result.wheel_nm, base.wheel_nm)
    np.testing.assert_array_equal(result.joint_target_rad, base.joint_target_rad)
    np.testing.assert_array_equal(result.wheel_speed_target_rad_s, base.wheel_speed_target_rad_s)
    assert not np.allclose(result.torque_nm, base.torque_nm)


def test_attitude_scale_moves_the_attitude_pair_only(perturbed_state):
    reference, base = _torques(perturbed_state)
    scaled, result = _torques(perturbed_state, attitude_feedback_scale=0.25)
    assert (scaled.roll_pitch_kp, scaled.roll_pitch_kd) == (45.0, 6.0)
    np.testing.assert_array_equal(scaled.leg_kp, reference.leg_kp)
    np.testing.assert_array_equal(scaled.leg_kd, reference.leg_kd)
    np.testing.assert_array_equal(result.leg_pd_nm, base.leg_pd_nm)
    np.testing.assert_array_equal(result.wheel_nm, base.wheel_nm)
    np.testing.assert_array_equal(result.joint_target_rad, base.joint_target_rad)
    assert not np.allclose(result.support_force_n, base.support_force_n)
    assert not np.allclose(result.torque_nm, base.torque_nm)
    # Support is affine in the attitude scale: both Kp and Kd move together and
    # the gravity part of the moment stays where it was.
    half = _torques(perturbed_state, attitude_feedback_scale=0.5)[1].support_force_n
    one_and_a_half = _torques(perturbed_state, attitude_feedback_scale=1.5)[1].support_force_n
    np.testing.assert_allclose(
        one_and_a_half - base.support_force_n, base.support_force_n - half, rtol=0, atol=1e-9
    )
    assert np.abs(one_and_a_half - base.support_force_n).max() > 1.0


def test_gravity_support_is_identical_when_attitude_error_is_zero():
    plant = D1Plant(sampling_mode="synchronized")
    plant.reset()
    state = D1MujocoTruthStateSource(plant).reset()
    upright = replace(state, base_angular_velocity_world=np.zeros(3))
    base = _torques(upright)[1]
    scaled = _torques(upright, attitude_feedback_scale=0.25)[1]
    assert abs(np.sum(base.support_force_n) - 9.81 * plant.nominal_total_mass_kg) < 1e-6
    np.testing.assert_allclose(scaled.support_force_n, base.support_force_n, rtol=0, atol=1e-9)
    np.testing.assert_allclose(scaled.support_nm, base.support_nm, rtol=0, atol=1e-9)


def test_environment_reset_and_step_keep_the_scaled_feedback_law():
    config = D1WheelLegControlConfig(leg_feedback_scale=0.25)
    scaled = D1LocomotionEnv(episode_seconds=0.03, wheel_leg_control=config)
    reference = D1LocomotionEnv(episode_seconds=0.03)
    try:
        for seed in (17, 29):
            scaled.reset(seed=seed)
            reference.reset(seed=seed)
            assert scaled.episode_metadata["controller_parameters"] == asdict(config)
            assert scaled.episode_metadata["controller_schema"] == SCHEMA_V4
            assert reference.episode_metadata["controller_schema"] == SCHEMA_V2
            controller = scaled.loop.controller.controller
            np.testing.assert_array_equal(controller.leg_kp, 0.25 * NOMINAL_LEG_KP)
            np.testing.assert_array_equal(controller.leg_kd, 0.25 * NOMINAL_LEG_KD)
            assert controller.roll_pitch_kp == 180.0
            for _ in range(2):
                scaled.step(np.zeros(8, np.float32))
                reference.step(np.zeros(8, np.float32))
            np.testing.assert_array_equal(controller.leg_kp, 0.25 * NOMINAL_LEG_KP)
            assert np.isfinite(controller.last_result.torque_nm).all()
        # Same seed, same first measured state: the leg PD is exactly quartered.
        first = reference.reset(seed=17)[0]
        np.testing.assert_array_equal(scaled.reset(seed=17)[0], first)
        scaled.step(np.zeros(8, np.float32))
        reference.step(np.zeros(8, np.float32))
        np.testing.assert_allclose(
            scaled.loop.controller.controller.last_result.leg_pd_nm,
            0.25 * reference.loop.controller.controller.last_result.leg_pd_nm,
            rtol=0,
            atol=1e-12,
        )
    finally:
        scaled.close()
        reference.close()


def test_rollout_cli_forwards_both_scales(tmp_path, monkeypatch):
    from scripts import run_d1_locomotion as entry

    captured = []

    def fake_run(env, output, **kwargs):
        captured.append(env.unwrapped.wheel_leg_control)
        env.close()
        return {"completed": True, "source_unchanged": True}

    monkeypatch.setattr(entry, "run", fake_run)
    arguments = [
        "--output",
        str(tmp_path / "new"),
        "--source",
        "oracle",
        "--leg-feedback-scale",
        "0.25",
        "--attitude-feedback-scale",
        "2",
    ]
    assert entry.main(arguments) == 0
    assert captured == [
        D1WheelLegControlConfig(leg_feedback_scale=0.25, attitude_feedback_scale=2.0)
    ]


@pytest.mark.parametrize("baseline", ("lqr", "mpc"))
@pytest.mark.parametrize("flag", ("--leg-feedback-scale", "--attitude-feedback-scale"))
def test_rollout_cli_rejects_force_baselines_before_creating_output(
    tmp_path, monkeypatch, baseline, flag
):
    from scripts import run_d1_locomotion as entry

    def fail(*args, **kwargs):
        raise AssertionError("the rollout must not start")

    monkeypatch.setattr(entry, "run", fail)
    output = tmp_path / "new"
    with pytest.raises(SystemExit) as error:
        entry.main(["--output", str(output), "--baseline", baseline, flag, "0.25"])
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("flag", ("--leg-feedback-scale", "--attitude-feedback-scale"))
def test_rollout_cli_rejects_malformed_scales(tmp_path, flag):
    from scripts import run_d1_locomotion as entry

    output = tmp_path / "new"
    with pytest.raises(SystemExit):
        entry.main(["--output", str(output), "--source", "oracle", flag, "0"])
    assert not output.exists()


def test_experiment_cli_forwards_scales_and_rejects_force_baselines(tmp_path):
    pytest.importorskip("stable_baselines3")
    from scripts.run_d1_locomotion_experiment import make_env, parser, wheel_control_config

    output = tmp_path / "run"
    common = ["benchmark", "--output", str(output)]
    scales = ["--leg-feedback-scale", "0.25", "--attitude-feedback-scale", "2"]
    config = wheel_control_config(parser().parse_args(common + scales))
    assert config == D1WheelLegControlConfig(leg_feedback_scale=0.25, attitude_feedback_scale=2.0)
    with pytest.raises(ValueError, match="wheel_leg"):
        wheel_control_config(parser().parse_args(common + ["--baseline", "mpc"] + scales))
    assert not output.exists()
    env = make_env("wheel_leg", 0, 0.01, "oracle", 1, wheel_control=config)
    try:
        env.reset(seed=17)
        metadata = env.unwrapped.episode_metadata
        assert metadata["controller_parameters"] == asdict(config)
        assert metadata["controller_schema"] == SCHEMA_V4
    finally:
        env.close()


@pytest.fixture
def fake_loader(monkeypatch):
    """Reach-the-loader sentinel; the real SB3 pickle path is never needed."""
    calls = []

    class PPO:
        @staticmethod
        def load(path, device=None, **kwargs):
            calls.append((str(path), device))
            raise RuntimeError("gain contract accepted; loader reached")

    module = types.ModuleType("stable_baselines3")
    module.PPO = PPO
    monkeypatch.setitem(sys.modules, "stable_baselines3", module)
    return calls


def _sidecar(tmp_path, env, tag):
    model, sidecar = tmp_path / f"{tag}.zip", tmp_path / f"{tag}.json"
    model.write_bytes(f"{tag} fixture: the gain contract must be checked first".encode())
    return model, sidecar, write_checkpoint_metadata(model, env, sidecar)


def _rewrite(sidecar, metadata, **changes):
    sidecar.write_text(json.dumps({**metadata, **changes}))


@pytest.fixture
def default_env():
    env = D1LocomotionEnv(episode_seconds=0.01)
    env.reset(seed=17)
    yield env
    env.close()


def _scaled_env(**gains):
    env = D1LocomotionEnv(episode_seconds=0.01, wheel_leg_control=D1WheelLegControlConfig(**gains))
    env.reset(seed=17)
    return env


def test_scaled_checkpoint_records_five_fields_and_reloads_against_itself(tmp_path, fake_loader):
    env = _scaled_env(attitude_feedback_scale=0.25)
    try:
        model, sidecar, metadata = _sidecar(tmp_path, env, "scaled")
        assert metadata["controller_parameters"] == asdict(env.wheel_leg_control)
        assert metadata["controller_parameters"]["attitude_feedback_scale"] == 0.25
        assert metadata["controller_schema"] == SCHEMA_V4
        with pytest.raises(RuntimeError, match="loader reached"):
            load_locomotion_policy(model, sidecar, env)
        assert len(fake_loader) == 1
    finally:
        env.close()


def test_three_key_sidecar_still_loads_for_unscaled_default_controller(
    tmp_path, fake_loader, default_env
):
    model, sidecar, metadata = _sidecar(tmp_path, default_env, "legacy")
    assert metadata["controller_schema"] == SCHEMA_V2
    _rewrite(
        sidecar,
        metadata,
        controller_parameters={name: metadata["controller_parameters"][name] for name in OLD_GAINS},
    )
    with pytest.raises(RuntimeError, match="loader reached"):
        load_locomotion_policy(model, sidecar, default_env)
    assert len(fake_loader) == 1


def test_three_key_sidecar_still_loads_for_unscaled_configured_controller(tmp_path, fake_loader):
    env = _scaled_env(wheel_kp=0.55, wheel_ki=1.5)
    try:
        model, sidecar, metadata = _sidecar(tmp_path, env, "configured")
        assert metadata["controller_schema"] == SCHEMA_V3
        _rewrite(
            sidecar,
            metadata,
            controller_parameters={"wheel_kp": 0.55, "wheel_ki": 1.5, "yaw_feedback_gain": 4.0},
        )
        with pytest.raises(RuntimeError, match="loader reached"):
            load_locomotion_policy(model, sidecar, env)
        assert len(fake_loader) == 1
    finally:
        env.close()


@pytest.mark.parametrize("schema", (SCHEMA_V2, SCHEMA_V3, SCHEMA_V4))
def test_three_key_sidecar_never_loads_against_scaled_feedback(tmp_path, fake_loader, schema):
    env = _scaled_env(leg_feedback_scale=0.25)
    try:
        model, sidecar, metadata = _sidecar(tmp_path, env, "old-against-scaled")
        _rewrite(
            sidecar,
            metadata,
            controller_schema=schema,
            controller_parameters={"wheel_kp": 2.2, "wheel_ki": 3.0, "yaw_feedback_gain": 4.0},
        )
        with pytest.raises(ValueError, match="controller_parameters"):
            load_locomotion_policy(model, sidecar, env)
        assert fake_loader == []
    finally:
        env.close()


def test_three_key_sidecar_with_new_schema_is_rejected(tmp_path, fake_loader, default_env):
    model, sidecar, metadata = _sidecar(tmp_path, default_env, "wrong-schema")
    _rewrite(
        sidecar,
        metadata,
        controller_schema=SCHEMA_V4,
        controller_parameters={name: metadata["controller_parameters"][name] for name in OLD_GAINS},
    )
    with pytest.raises(ValueError, match="controller_parameters"):
        load_locomotion_policy(model, sidecar, default_env)
    assert fake_loader == []


def test_absent_gain_record_stays_compatible_only_with_the_fixed_default(tmp_path, fake_loader):
    default, scaled = D1LocomotionEnv(episode_seconds=0.01), _scaled_env(leg_feedback_scale=0.25)
    try:
        default.reset(seed=17)
        model, sidecar, metadata = _sidecar(tmp_path, default, "sidecarless")
        legacy = {key: value for key, value in metadata.items() if key != "controller_parameters"}
        legacy["recorded_episode"] = {
            key: value
            for key, value in metadata["recorded_episode"].items()
            if key != "controller_parameters"
        }
        sidecar.write_text(json.dumps(legacy))
        with pytest.raises(RuntimeError, match="loader reached"):
            load_locomotion_policy(model, sidecar, default)
        assert len(fake_loader) == 1
        with pytest.raises(ValueError, match="controller_parameters"):
            load_locomotion_policy(model, sidecar, scaled)
        assert len(fake_loader) == 1
    finally:
        default.close()
        scaled.close()


def test_changed_scale_is_rejected_despite_identical_observation_contract(tmp_path, fake_loader):
    quarter, half = _scaled_env(leg_feedback_scale=0.25), _scaled_env(leg_feedback_scale=0.5)
    try:
        assert quarter.observation_space.shape == half.observation_space.shape == (82,)
        assert quarter.action_space.shape == half.action_space.shape == (8,)
        assert (
            quarter.episode_metadata["controller_schema"]
            == half.episode_metadata["controller_schema"]
            == SCHEMA_V4
        )
        model, sidecar, _ = _sidecar(tmp_path, quarter, "quarter")
        with pytest.raises(ValueError, match="controller_parameters"):
            load_locomotion_policy(model, sidecar, half)
        assert fake_loader == []
    finally:
        quarter.close()
        half.close()


@pytest.mark.parametrize(
    "parameters",
    (
        {"wheel_kp": 2.2, "wheel_ki": 3.0},
        {key: value for key, value in DEFAULT_RECORD.items() if key != "attitude_feedback_scale"},
        {**DEFAULT_RECORD, "extra": 1.0},
        {**DEFAULT_RECORD, "leg_feedback_scale": "1.0"},
        {**DEFAULT_RECORD, "leg_feedback_scale": True},
        {**DEFAULT_RECORD, "leg_feedback_scale": None},
        {**DEFAULT_RECORD, "leg_feedback_scale": float("nan")},
        {**DEFAULT_RECORD, "leg_feedback_scale": float("inf")},
        {**DEFAULT_RECORD, "leg_feedback_scale": 0},
        {**DEFAULT_RECORD, "attitude_feedback_scale": -1.0},
        {**DEFAULT_RECORD, "attitude_feedback_scale": [1.0]},
        {**DEFAULT_RECORD, "attitude_feedback_scale": 0.25},
        {**DEFAULT_RECORD, "wheel_ki": 1.0},
        [2.2, 3.0, 4.0, 1.0, 1.0],
        "wheel_kp=2.2",
        None,
    ),
)
def test_malformed_or_differing_gain_records_fail_before_deserialization(
    tmp_path, fake_loader, default_env, parameters
):
    model, sidecar, metadata = _sidecar(tmp_path, default_env, "malformed")
    assert metadata["controller_parameters"] == DEFAULT_RECORD
    _rewrite(sidecar, metadata, controller_parameters=parameters)
    with pytest.raises(ValueError, match="controller_parameters"):
        load_locomotion_policy(model, sidecar, default_env)
    assert fake_loader == []
