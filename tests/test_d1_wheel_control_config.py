"""Runtime gains are part of the policy contract, not a hidden tuning knob."""

import json
from dataclasses import asdict

import numpy as np
import pytest

from wheel_legged_control.d1.locomotion_checkpoint import (
    load_locomotion_policy,
    write_checkpoint_metadata,
)
from wheel_legged_control.d1.locomotion_env import D1LocomotionEnv
from wheel_legged_control.d1.wheel_leg_controller import (
    D1WheelLegControlConfig,
    D1WheelLegController,
)


@pytest.mark.parametrize("field", ("wheel_kp", "wheel_ki", "yaw_feedback_gain"))
@pytest.mark.parametrize("value", (True, np.array([1.0]), float("nan"), float("inf"), -1.0))
def test_invalid_gain_is_rejected(field, value):
    with pytest.raises((TypeError, ValueError)):
        D1WheelLegControlConfig(**{field: value})


def test_zero_integral_and_yaw_are_allowed_but_positive_wheel_p_is_required():
    assert D1WheelLegControlConfig(wheel_ki=0, yaw_feedback_gain=0).wheel_ki == 0
    with pytest.raises(ValueError):
        D1WheelLegControlConfig(wheel_kp=0)


def test_default_controller_retains_historical_schema_and_gains():
    controller = D1WheelLegController()
    assert controller.control_schema == "d1-wheel-leg-joint-pd-wheel-pi-yaw-feedback-v2"
    assert controller.wheel_kp == 2.2 and controller.wheel_ki == 3.0
    assert controller.yaw_feedback_gain == 4.0


def test_custom_gain_configuration_survives_reset_and_is_recorded():
    config = D1WheelLegControlConfig(wheel_kp=0.55, wheel_ki=0.75)
    env = D1LocomotionEnv(episode_seconds=0.02, wheel_leg_control=config)
    try:
        for seed in (17, 29):
            env.reset(seed=seed)
            assert env.episode_metadata["controller_parameters"] == asdict(config)
            assert env.episode_metadata["controller_schema"].endswith("configured-v3")
            assert env.loop.controller.controller.wheel_kp == 0.55
            env.step(np.zeros(8, np.float32))
    finally:
        env.close()


def test_force_controller_does_not_silently_ignore_wheel_configuration():
    with pytest.raises(ValueError, match="wheel_leg"):
        D1LocomotionEnv(baseline="lqr", wheel_leg_control=D1WheelLegControlConfig())


def test_same_custom_schema_different_gains_rejected_before_deserialization(tmp_path):
    first = D1LocomotionEnv(
        episode_seconds=0.01,
        wheel_leg_control=D1WheelLegControlConfig(wheel_kp=0.55, wheel_ki=0.75),
    )
    second = D1LocomotionEnv(
        episode_seconds=0.01, wheel_leg_control=D1WheelLegControlConfig(wheel_kp=0.55, wheel_ki=1.5)
    )
    model, sidecar = tmp_path / "model.zip", tmp_path / "checkpoint.json"
    model.write_bytes(b"invalid model: the parameter contract must reject first")
    try:
        first.reset(seed=17)
        second.reset(seed=17)
        metadata = write_checkpoint_metadata(model, first, sidecar)
        assert metadata["controller_parameters"] == asdict(first.wheel_leg_control)
        with pytest.raises(ValueError, match="controller_parameters"):
            load_locomotion_policy(model, sidecar, second)
        # Dropping the field must not make a custom controller look historical.
        metadata.pop("controller_parameters")
        sidecar.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="controller_parameters"):
            load_locomotion_policy(model, sidecar, first)
    finally:
        first.close()
        second.close()


def test_legacy_default_sidecar_still_reaches_the_model_loader(tmp_path, monkeypatch):
    env = D1LocomotionEnv(episode_seconds=0.01)
    model, sidecar = tmp_path / "model.zip", tmp_path / "checkpoint.json"
    model.write_bytes(b"legacy metadata fixture, not a trained policy")
    try:
        env.reset(seed=17)
        metadata = write_checkpoint_metadata(model, env, sidecar)
        metadata.pop("controller_parameters")
        metadata["recorded_episode"].pop("controller_parameters")
        sidecar.write_text(json.dumps(metadata))

        def stop_at_loader(*args, **kwargs):
            raise RuntimeError("contract accepted; loader reached")

        monkeypatch.setattr("stable_baselines3.PPO.load", stop_at_loader)
        with pytest.raises(RuntimeError, match="loader reached"):
            load_locomotion_policy(model, sidecar, env)
    finally:
        env.close()


@pytest.mark.parametrize("direction", (-1, 1))
def test_lower_bandwidth_turn_with_twenty_ms_packet_delay(direction):
    from scripts.probe_d1_wheel_leg_actions import run_case

    rows, result = run_case(
        "turn_left" if direction == 1 else "turn_right",
        state_source="sensor",
        sensor_delay_steps=2,
        wheel_kp=0.55,
        wheel_ki=1.5,
        yaw_gain=4,
        seed=17,
    )
    assert result["completed"] and len(rows) == 800
    assert result["turn_progress_fraction"] >= 0.70
    assert result["tail_yaw_rmse_rps"] <= 0.05
    assert result["max_velocity_limit_fraction"] <= 1
    assert result["max_joint_position_violation_rad"] <= 1e-10


def test_shared_command_entry_passes_explicit_gains_to_environment(tmp_path, monkeypatch):
    from scripts import run_d1_locomotion as entry

    captured = []

    def fake_run(env, output, **kwargs):
        captured.append(env.unwrapped.wheel_leg_control)
        env.close()
        return {"completed": True, "source_unchanged": True}

    monkeypatch.setattr(entry, "run", fake_run)
    assert (
        entry.main(
            [
                "--output",
                str(tmp_path / "new"),
                "--source",
                "oracle",
                "--wheel-kp",
                "0.55",
                "--wheel-ki",
                "1.5",
            ]
        )
        == 0
    )
    assert captured == [D1WheelLegControlConfig(wheel_kp=0.55, wheel_ki=1.5)]


def test_training_worker_receives_same_gain_configuration():
    from scripts.run_d1_locomotion_experiment import make_env

    config = D1WheelLegControlConfig(wheel_kp=0.55, wheel_ki=1.5)
    env = make_env("wheel_leg", 0, 0.01, "oracle", 1, wheel_control=config)
    try:
        env.reset(seed=17)
        assert env.unwrapped.episode_metadata["controller_parameters"] == asdict(config)
    finally:
        env.close()
