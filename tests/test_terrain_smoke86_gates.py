from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np

from scripts import run_terrain_smoke86 as runner
from scripts.audit_terrain_smoke86 import audit
from scripts.terrain_smoke86_env import policy_observation, reward


def test_incomplete_or_uncorroborated_sensing_cannot_release_training():
    design = runner.read(runner.PLAN)["development"]
    cases = [{"terrain": t, "seed": s, "passed": True, "full_horizon": True, "task_steps": 600,
              "zero_window_complete": True, "sensor_passed": True, "first_sensor_window_corroborated": True,
              "torque_bound_violations": 0} for t in design["terrains"] for s in design["seeds"]]
    assert runner.permitted_zero_cases(cases, design)
    assert not runner.permitted_zero_cases(cases[:-1], design)
    assert not runner.permitted_zero_cases([cases[0]] * 6, design)
    for field, value in (("full_horizon", False), ("task_steps", 599), ("zero_window_complete", False),
                         ("first_sensor_window_corroborated", False), ("torque_bound_violations", 1)):
        assert not runner.permitted_zero_cases([*cases[:-1], {**cases[-1], field: value}], design)


def test_actor_inputs_ignore_simulator_truth_and_pitch_reference_uses_uphill_sign():
    state = SimpleNamespace(base_rpy=(0., -0.04, 0.), base_angular_velocity_body=(0., 0., 0.),
                            control_time_s=1., base_linear_velocity_body=(0.35, 0., 0.))
    cmd = SimpleNamespace(forward_velocity_mps=0.35, pitch_rad=0.)
    reading = {"reason": "valid", "slope_rad": 0.04, "captured_s": 0.95}
    obs = policy_observation(state, cmd, reading)
    assert obs.shape == (9,)
    other_truth = {**reading, "corroborated_slope_rad": -0.04, "ground_slope_rad": 1.0}
    np.testing.assert_array_equal(obs, policy_observation(state, cmd, other_truth))
    limits, torques = np.ones(16), np.zeros(16)
    assert reward(state, cmd, reading, 0., torques, limits, False) == 0
    assert reward(state, cmd, other_truth, 0., torques, limits, True) == -10
    stale = {"reason": "expired", "slope_rad": None, "captured_s": 0.6}
    assert not policy_observation(state, cmd, stale)[7]
    assert reward(state, cmd, stale, 0., torques, limits, False) < 0


def test_published_physics_and_optimizer_gates_reconstruct_without_integration(tmp_path):
    assets = runner.ROOT / "docs/assets/project86"
    with ZipFile(assets / "recordings.zip") as records:
        for member in records.namelist():
            if member.startswith(("development/", "fallback/", "fixture/")):
                records.extract(member, tmp_path)
    report = {}
    audit(tmp_path, report)
    published = runner.read(assets / "independent_audit.json")["stages"]
    for name, windows in (("development", 3), ("fallback", 5)):
        stage = report["stages"][name]
        assert stage["controls"] == published[name]["controls"] == 3600
        assert sum(c["zero_window_complete"] for c in stage["cases"]) == windows
        assert stage["passed"] == published[name]["passed"] is False
    assert report["stages"]["fixture"] == published["fixture"]
    assert report["stages"]["fresh"]["status"] == "NOT_RUN"
