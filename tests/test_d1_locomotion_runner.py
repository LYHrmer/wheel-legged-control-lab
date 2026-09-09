import csv
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.run_d1_locomotion import KeyboardCommands, RecordedCommands, build_parser, run
from wheel_legged_control.d1.control_loop import D1MotionCommand
from wheel_legged_control.d1.locomotion_env import D1LocomotionEnv
from wheel_legged_control.d1.observation_history import D1ObservationHistory


def test_keyboard_timeout_height_keys_cannot_revive_expired_motion():
    now = [0.0]
    commands = KeyboardCommands(clock=lambda: now[0])
    assert commands(0) == D1MotionCommand()
    commands.key_callback(ord("W"))
    commands.key_callback(ord("A"))
    assert commands(0.01) == D1MotionCommand(0.05, 0.05, 0.455)
    now[0] = 0.81
    commands.key_callback(ord("R"))
    assert commands(0.01) == D1MotionCommand(0, 0, 0.46)
    commands.key_callback(9999)
    assert commands(0.02) == D1MotionCommand(0, 0, 0.46)
    commands.key_callback(ord("W"))
    assert commands(0.02) == D1MotionCommand(0.05, 0, 0.46)


def test_headless_records_executed_command_not_next_prepared_command(tmp_path):
    def source(time_s):
        return D1MotionCommand(forward_velocity_mps=0.05 if time_s == 0 else -0.05)

    env = D1ObservationHistory(D1LocomotionEnv(episode_seconds=0.03, command_source=source), 2)
    viewer_before = "mujoco.viewer" in sys.modules
    output = tmp_path / "rollout"
    summary = run(env, output, seed=17)
    assert ("mujoco.viewer" in sys.modules) == viewer_before
    assert summary["steps"] == 3
    assert summary["stop_reason"] == "time_limit"
    assert summary["completed"] is True
    with (output / "telemetry.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row["command_vx_mps"]) for row in rows] == [0.05, -0.05, -0.05]
    assert [float(row["decision_time_s"]) for row in rows] == pytest.approx([0, 0.01, 0.02])
    assert [float(row["time_s"]) for row in rows] == pytest.approx([0.01, 0.02, 0.03])
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["episode"]["schedule"] is None
    assert protocol["history"]["observation_dim"] == 164
    import hashlib

    import mujoco

    model_path = output / "model.mjb"
    assert protocol["compiled_model_sha256"] == hashlib.sha256(model_path.read_bytes()).hexdigest()
    archived = mujoco.MjModel.from_binary_path(str(model_path))
    assert (archived.nq, archived.nv, archived.nu) == (23, 22, 16)
    np.testing.assert_array_equal(archived.hfield_data, env.unwrapped.plant.model.hfield_data)
    with np.load(output / "states.npz", allow_pickle=False) as states:
        assert states["qpos"].shape == (4, 23)
        assert states["observation"].shape == (4, 164)
        np.testing.assert_array_equal(states["raw_action"], np.zeros((3, 8)))
        np.testing.assert_allclose(states["time_s"], [0, 0.01, 0.02, 0.03])


def test_recorded_commands_replay_same_physics_without_next_command_shift(tmp_path):
    original = tmp_path / "original"
    replay = tmp_path / "replay"
    run(
        D1LocomotionEnv(
            episode_seconds=0.03,
            command_source=lambda t: D1MotionCommand(0.05 if t < 0.015 else -0.05),
        ),
        original,
    )
    commands = RecordedCommands(original / "telemetry.csv")
    assert commands.duration_s == 0.03
    assert commands(0).forward_velocity_mps == 0.05
    assert commands(0.02).forward_velocity_mps == -0.05
    assert commands(0.03) == commands(0.02)  # terminal observation only
    run(D1LocomotionEnv(episode_seconds=0.03, command_source=commands), replay)
    with np.load(original / "states.npz") as before, np.load(replay / "states.npz") as after:
        for key in ("qpos", "qvel", "raw_action", "actuator_applied_nm", "observation"):
            # Last terminal-only command is intentionally held in CSV replay.
            comparison = slice(None, -1) if key == "observation" else slice(None)
            np.testing.assert_array_equal(before[key][comparison], after[key][comparison])


def test_cli_is_headless_by_default_and_input_modes_are_exclusive(tmp_path):
    parser = build_parser()
    args = parser.parse_args(["--output", str(tmp_path / "new")])
    assert args.keyboard is False
    assert args.policy is None
    assert args.baseline == "wheel_leg"
    with pytest.raises(SystemExit):
        parser.parse_args(["--output", str(tmp_path), "--keyboard", "--command-csv", "old.csv"])


def test_viewer_cannot_change_control_physics_or_state(tmp_path, monkeypatch):
    import mujoco.viewer

    opened = []

    def launch(model, data, key_callback):
        opened.append((model, data))

        class Viewer:
            cam = SimpleNamespace(distance=0, lookat=np.zeros(3))

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def lock(self):
                return nullcontext()

            def is_running(self):
                return True

            def sync(self):
                # A GUI edit/drag must affect only the viewer copy.
                model.opt.gravity[:] = 1000
                data.qpos[:] = 20

        return Viewer()

    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    neutral = lambda _: D1MotionCommand()
    run(D1LocomotionEnv(episode_seconds=0.03, command_source=neutral), tmp_path / "headless")
    keyboard = KeyboardCommands()
    task = D1LocomotionEnv(episode_seconds=0.03, command_source=keyboard)
    run(task, tmp_path / "viewer", keyboard=keyboard)
    assert len(opened) == 1
    assert opened[0][0] is not task.plant.model
    assert opened[0][1] is not task.plant.data
    with (
        np.load(tmp_path / "headless/states.npz") as a,
        np.load(tmp_path / "viewer/states.npz") as b,
    ):
        np.testing.assert_array_equal(a["qpos"], b["qpos"])
        np.testing.assert_array_equal(a["qvel"], b["qvel"])


def test_keyboard_bounds_space_escape_and_unknown_key_timeout():
    now = [0.0]
    commands = KeyboardCommands(clock=lambda: now[0])
    for _ in range(20):
        commands.key_callback(ord("W"))
        commands.key_callback(ord("A"))
        commands.key_callback(ord("R"))
    assert commands(0) == D1MotionCommand(0.30, 0.25, 0.48)
    now[0] = 0.7
    commands.key_callback(ord("X"))
    now[0] = 0.8
    assert commands(0) == D1MotionCommand(0, 0, 0.48)
    for _ in range(20):
        commands.key_callback(ord("S"))
        commands.key_callback(ord("D"))
        commands.key_callback(ord("F"))
    assert commands(0) == D1MotionCommand(-0.20, -0.25, 0.43)
    commands.key_callback(32)
    assert commands(0) == D1MotionCommand(0, 0, 0.43)
    commands.key_callback(ord("W"))
    commands.key_callback(256)
    commands.key_callback(ord("W"))
    assert commands.stopped
    assert commands(0) == D1MotionCommand(0, 0, 0.43)


@pytest.mark.parametrize("timeout", [True, 0, -1, float("nan"), float("inf")])
def test_invalid_keyboard_timeout_is_rejected(timeout):
    with pytest.raises(ValueError):
        KeyboardCommands(timeout_s=timeout)
