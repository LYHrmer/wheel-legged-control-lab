import csv

import numpy as np
import pytest

from wheel_legged_control.d1.experiments import (
    D1Rollout,
    _paired_ci,
    build_parser,
    compute_d1_metrics,
    write_d1_randomized_audit,
)
from wheel_legged_control.d1.model import JOINT_TORQUE_LIMIT


def _rollout() -> D1Rollout:
    torques = np.vstack((np.zeros(16), JOINT_TORQUE_LIMIT))
    joint_velocities = np.ones_like(torques)
    return D1Rollout(
        controller="D1 LQR+VMC",
        scenario="randomized",
        time_s=np.asarray((0.0, 0.01)),
        states=np.zeros((2, 6)),
        forward_velocities_mps=np.zeros(2),
        torques=torques,
        commands=np.zeros((2, 2)),
        pushes=np.zeros(2),
        rewards=np.asarray((1.0, 3.0)),
        wheel_contacts=np.asarray((4, 4)),
        undesired_contacts=np.zeros(2),
        solve_times_ms=np.asarray((0.1, 0.2)),
        terminated=False,
        evaluation_seed=123,
        state_estimation_mode="estimated",
        state_ages_ms=np.asarray((1.0, 3.0)),
        domain={
            "base_mass_scale": 1.1,
            "damping_scale": 1.2,
            "friction_scale": 0.8,
            "actuator_strength_scale": 0.9,
        },
        action_delay_steps=1,
        state_delay_steps=3,
        sensor_noise_scale=0.75,
        state_estimator_seed=987,
        absolute_mechanical_power_w=np.sum(np.abs(torques * joint_velocities), axis=1),
        torque_saturation_fractions=np.mean(np.abs(torques) >= 0.98 * JOINT_TORQUE_LIMIT, axis=1),
        residual_actions=np.asarray(((0.2, -0.1), (0.4, 0.3))),
    )


def test_d1_metrics_record_reproducibility_and_effort_evidence() -> None:
    rollout = _rollout()

    metrics = compute_d1_metrics(rollout)

    assert metrics["evaluation_seed"] == 123
    assert metrics["state_estimation_mode"] == "estimated"
    assert metrics["state_age_p95_ms"] == 2.9
    assert metrics["action_delay_steps"] == 1
    assert metrics["state_delay_steps"] == 3
    assert metrics["sensor_noise_scale"] == 0.75
    assert metrics["state_estimator_seed"] == 987
    assert metrics["base_mass_scale"] == 1.1
    assert metrics["damping_scale"] == 1.2
    assert metrics["friction_scale"] == 0.8
    assert metrics["actuator_strength_scale"] == 0.9
    assert metrics["mean_abs_mechanical_power_w"] == np.sum(JOINT_TORQUE_LIMIT) / 2.0
    assert metrics["torque_saturation_ratio"] == 0.5
    assert metrics["residual_action_rms"] == pytest.approx(
        np.sqrt(np.mean(np.asarray(((0.2, -0.1), (0.4, 0.3))) ** 2))
    )
    assert metrics["mean_residual_longitudinal_force_n"] == pytest.approx(13.5)
    assert metrics["mean_residual_vertical_force_n"] == pytest.approx(8.0)


def test_d1_rollout_keeps_legacy_constructor_defaults() -> None:
    rollout = D1Rollout(
        controller="D1 LQR+VMC",
        scenario="nominal",
        time_s=np.asarray((0.0,)),
        states=np.zeros((1, 6)),
        forward_velocities_mps=np.zeros(1),
        torques=np.zeros((1, 16)),
        commands=np.zeros((1, 2)),
        pushes=np.zeros(1),
        rewards=np.zeros(1),
        wheel_contacts=np.asarray((4,)),
        undesired_contacts=np.zeros(1),
        solve_times_ms=np.zeros(1),
        terminated=False,
    )

    metrics = compute_d1_metrics(rollout)

    assert metrics["evaluation_seed"] == -1
    assert metrics["state_estimation_mode"] == "oracle"
    assert metrics["state_age_p95_ms"] == 0.0


def test_randomized_audit_pairs_by_evaluation_seed(tmp_path) -> None:
    def record(controller: str, seed: int, value: float, reward: float) -> dict[str, object]:
        return {
            "controller": controller,
            "scenario": "randomized",
            "evaluation_seed": seed,
            "success": 1,
            "velocity_rmse_mps": value,
            "pitch_rmse_deg": value,
            "height_rmse_mm": value,
            "mean_reward": reward,
        }

    baseline = "D1 LQR+VMC"
    residual = "D1 LQR+VMC+PPO"
    records = [
        record(baseline, 2, 20.0, 2.0),
        record(baseline, 1, 10.0, 1.0),
        record(baseline, 3, 40.0, 4.0),
        record(residual, 3, 50.0, 14.0),
        record(residual, 1, 13.0, 4.0),
        record(residual, 2, 25.0, 7.0),
    ]

    write_d1_randomized_audit(records, tmp_path)

    markdown = (tmp_path / "randomized_audit.md").read_text(encoding="utf-8")
    expected = _paired_ci(np.asarray((3.0, 5.0, 10.0)))
    assert (
        f"Velocity RMSE [m/s]: {expected[0]:+.3f} [{expected[1]:+.3f}, {expected[2]:+.3f}]"
    ) in markdown
    reward_ci = _paired_ci(np.asarray((3.0, 5.0, 10.0)))
    assert (
        f"Mean reward: {reward_ci[0]:+.3f} [{reward_ci[1]:+.3f}, {reward_ci[2]:+.3f}]"
    ) in markdown
    assert markdown.count("| Controller | Episodes |") == 1

    with (tmp_path / "randomized_audit.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(records)
    assert {int(row["evaluation_seed"]) for row in rows} == {1, 2, 3}


def test_d1_cli_defaults_to_oracle_state_estimation() -> None:
    args = build_parser().parse_args([])

    assert args.state_mode == "oracle"
    assert not args.no_policy


def test_d1_cli_can_disable_default_policy() -> None:
    args = build_parser().parse_args(["--state-mode", "estimated", "--no-policy"])

    assert args.state_mode == "estimated"
    assert args.no_policy


def test_randomized_audit_rejects_unmatched_seed_sets(tmp_path) -> None:
    records = [
        {
            "controller": "D1 LQR+VMC",
            "evaluation_seed": 1,
            "success": 1,
            "mean_reward": 1.0,
            "velocity_rmse_mps": 1.0,
            "pitch_rmse_deg": 1.0,
            "height_rmse_mm": 1.0,
        },
        {
            "controller": "D1 LQR+VMC+PPO",
            "evaluation_seed": 2,
            "success": 1,
            "mean_reward": 1.0,
            "velocity_rmse_mps": 1.0,
            "pitch_rmse_deg": 1.0,
            "height_rmse_mm": 1.0,
        },
    ]

    with pytest.raises(ValueError, match="evaluation seeds differ"):
        write_d1_randomized_audit(records, tmp_path)
