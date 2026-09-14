"""Check command consumption and real evaluation-record consistency."""

import json
from dataclasses import asdict

import numpy as np
import pytest

from scripts.run_d1_course_curriculum import forward_command
from scripts.run_d1_wheel_common_mean_study import case_command, evaluate_one
from wheel_legged_control.d1.locomotion_terrain import D1LocomotionTerrainConfig


def command_spec(vx=.25, yaw=0.):
    return {"target_forward_mps": vx, "target_yaw_rps": yaw,
            "settle_seconds": .5, "ramp_seconds": .5, "height_m": .455}


def test_development_command_is_exactly_the_previous_training_command():
    command = case_command(command_spec())
    for time in np.arange(3201) * .01:
        assert command(time) == forward_command(time)


@pytest.mark.parametrize("vx,yaw", [(.22, 0.), (.27, .01)])
def test_final_cases_really_use_their_own_velocity_and_yaw(vx, yaw):
    command = case_command(command_spec(vx, yaw))
    for time, fraction in [(0., 0.), (.5, 0.), (.75, .5), (1., 1.), (32., 1.)]:
        actual = command(time)
        assert actual.forward_velocity_mps == pytest.approx(vx * fraction)
        assert actual.yaw_rate_rps == pytest.approx(yaw * fraction)
        assert actual.clearance_m == .455


def test_real_evaluation_records_include_initial_state_and_all_physical_steps(tmp_path):
    case = {"name": "short", "split": "dev", "seed": 55101, "episode_seconds": .04,
            "terrain": asdict(D1LocomotionTerrainConfig()), "command": command_spec()}
    output = tmp_path / "zero"
    summary, initial = evaluate_one(output, case=case, label="zero")
    assert initial is not None
    assert summary["actual_transitions"] == 4
    assert summary["terminal_reason"] == "time_limit"
    assert summary["initial_state_observation_metadata_same"]
    assert summary["mean_deterministic_box_clipped_fraction"] == 0
    rows = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert summary["cumulative_return"] == sum(row["reward"] for row in rows)
    with np.load(output / "states.npz") as states:
        assert states["observation"].shape == (5, 82)
        np.testing.assert_allclose(states["time_s"], np.arange(5) * .01, rtol=0, atol=1e-12)
    assert all(row["applied_action"] == [0.] * 8 for row in rows)
