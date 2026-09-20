"""Plane archive/reset and prefix boundaries without any native integration."""
import json

import mujoco
import numpy as np
import pytest

from scripts.probe_d1_heading_flat_plane import PlaneDiagnosticEnv, initial_check, prefix_check


def test_instrumented_reset_and_close_archive_have_no_physics(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("no physical steps in reset/archive test")

    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    env = PlaneDiagnosticEnv(diagnostic_output=tmp_path, episode_seconds=.01)
    try:
        observation, info = env.reset(seed=55101)
        assert observation.shape == (85,)
        assert info["episode_metadata"]["terrain"]["collision_geometry"] == "native_mujoco_plane"
        assert env.plant.data.time == 0.
    finally:
        env.close()
        env.close()
    receipt = json.loads((tmp_path / "plane_execution_receipt.json").read_text())
    assert receipt["actual_physics_substeps_from_clock"] == 0
    with np.load(tmp_path / "execution_states.npz") as saved:
        assert saved["tag"].tolist() == ["reset", "close"]
        assert np.all(saved["time_s"] == 0.)


def test_prefix_respects_next_command_observation_and_requires_full_length():
    a = {"qpos": np.zeros((5, 2)), "observations": np.zeros((5, 3)),
         "requested_torque_nm": np.zeros((4, 16))}
    b = {k: v.copy() for k, v in a.items()}
    b["observations"][4] = 1.
    assert prefix_check("release", a, b, 4, next_observation_equal=False)["passed"]
    assert not prefix_check("impulse", a, b, 4, next_observation_equal=True)["passed"]
    b["qpos"] = b["qpos"][:4]
    assert not prefix_check("short", a, b, 4, next_observation_equal=False)["passed"]


def test_initial_observation_only_tolerates_exact_zero_sign_and_keeps_raw_evidence():
    a = {"qpos": np.zeros((1, 2)), "qvel": np.zeros((1, 2)),
         "observations": np.zeros((1, 85), dtype=np.float32)}
    b = {k: v.copy() for k, v in a.items()}
    b["observations"][0, [38, 56, 58]] = -0.
    check = initial_check("reverse", a, b)
    assert check["passed"] and not check["raw_observations_bitwise_equal"]
    assert check["raw_observation_signbit_differences"] == [38, 56, 58]
    assert np.signbit(b["observations"][0, 38])
    b["observations"][0, 38] = np.nextafter(np.float32(0), np.float32(1))
    assert not initial_check("nonzero", a, b)["passed"]
    b["observations"] = a["observations"].copy()
    b["qpos"][0, 0] = -0.
    assert not initial_check("physical_sign", a, b)["passed"]


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_initial_observation_rejects_identical_nonfinite_values(invalid):
    a = {"qpos": np.zeros((1, 2)), "qvel": np.zeros((1, 2)),
         "observations": np.zeros((1, 85), dtype=np.float32)}
    a["observations"][0, 3] = invalid
    assert not initial_check("invalid", a, {k: v.copy() for k, v in a.items()})["passed"]
