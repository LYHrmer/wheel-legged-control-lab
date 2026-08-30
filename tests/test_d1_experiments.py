import csv
import hashlib
import json

import numpy as np
import pytest

from wheel_legged_control.d1.experiments import (
    D1_STATE_DELAY_SWEEP_STEPS,
    D1_SWEEP_METRICS,
    D1Rollout,
    _paired_bootstrap_ci,
    _paired_ci,
    _wilson_interval,
    build_parser,
    compute_d1_metrics,
    main,
    run_d1_state_delay_sweep,
    write_d1_metrics,
    write_d1_randomized_audit,
)
from wheel_legged_control.d1.model import JOINT_TORQUE_LIMIT


def _rollout() -> D1Rollout:
    torques = np.vstack((np.zeros(16), JOINT_TORQUE_LIMIT))
    joint_velocities = np.ones_like(torques)
    raw_estimated_states = np.zeros((2, 6))
    raw_estimated_states[:, (0, 2)] = 1.0
    raw_estimated_states[:, 1] = 0.1
    raw_estimated_states[:, (3, 5)] = 0.5
    control_states = raw_estimated_states / 2.0
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
        compensation_applied_flags=np.asarray((True, False)),
        compensation_rejected_flags=np.asarray((False, True)),
        raw_estimated_states=raw_estimated_states,
        control_states=control_states,
        initial_state_fingerprint="initial-state",
        initial_command_fingerprint="initial-command",
        push_schedule_fingerprint="push-schedule",
    )


def test_d1_metrics_record_reproducibility_and_effort_evidence() -> None:
    rollout = _rollout()

    metrics = compute_d1_metrics(rollout)

    assert metrics["evaluation_seed"] == 123
    assert metrics["state_estimation_mode"] == "estimated"
    assert metrics["latency_compensation"] == "none"
    assert metrics["state_age_mean_ms"] == 2.0
    assert metrics["state_age_p95_ms"] == 2.9
    assert metrics["state_age_max_ms"] == 3.0
    assert metrics["compensation_applied_ratio"] == 0.5
    assert metrics["compensation_rejected_ratio"] == 0.5
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
    assert metrics["raw_position_estimation_rmse_m"] == 1.0
    assert metrics["control_position_estimation_rmse_m"] == 0.5
    assert metrics["raw_pitch_estimation_rmse_deg"] == pytest.approx(np.rad2deg(0.1))
    assert metrics["control_pitch_estimation_rmse_deg"] == pytest.approx(np.rad2deg(0.05))
    assert metrics["raw_velocity_estimation_rmse_mps"] == 0.5
    assert metrics["control_velocity_estimation_rmse_mps"] == 0.25


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
    assert np.isnan(metrics["raw_position_estimation_rmse_m"])


def _sweep_rollout(
    baseline: str,
    seed: int,
    delay_steps: int,
    *,
    latency_compensation: str,
    policy: object | None = None,
) -> D1Rollout:
    steps = 6
    states = np.zeros((steps, 6), dtype=np.float64)
    states[:, 2] = 0.45
    states[:, 3] = 0.1 * delay_steps
    raw_states = states.copy()
    raw_states[:, (0, 2, 3, 5)] += 0.001 * delay_steps
    control_states = states.copy()
    control_states[:, (0, 2, 3, 5)] += 0.0005 * delay_steps
    state_ages_ms = np.minimum(np.arange(1, steps + 1), delay_steps) * 10.0
    controller = (
        "D1 LQR+VMC+PPO"
        if policy is not None
        else ("D1 LQR+VMC" if baseline == "lqr" else "D1 MPC+VMC")
    )
    return D1Rollout(
        controller=controller,
        scenario="randomized",
        time_s=np.arange(steps) * 0.01,
        states=states,
        forward_velocities_mps=states[:, 3],
        torques=np.zeros((steps, 16)),
        commands=np.zeros((steps, 2)),
        pushes=np.zeros(steps),
        rewards=np.full(steps, 1.0 - 0.01 * delay_steps),
        wheel_contacts=np.full(steps, 4),
        undesired_contacts=np.zeros(steps),
        solve_times_ms=np.full(steps, 0.2),
        terminated=bool(delay_steps == 5 and seed % 2),
        evaluation_seed=seed,
        state_estimation_mode="estimated",
        state_ages_ms=state_ages_ms,
        domain={
            "base_mass_scale": 1.0 + seed * 1e-5,
            "damping_scale": 1.0,
            "friction_scale": 1.0,
            "actuator_strength_scale": 1.0,
        },
        action_delay_steps=0,
        state_delay_steps=delay_steps,
        sensor_noise_scale=0.0,
        state_estimator_seed=seed + 1000,
        latency_compensation=latency_compensation,
        raw_estimated_states=raw_states,
        control_states=control_states,
        initial_state_fingerprint=f"state-{seed}",
        initial_command_fingerprint=f"command-{seed}",
        push_schedule_fingerprint=f"push-{seed}",
    )


def _patch_sweep_rollout(monkeypatch) -> None:
    def fake_run(
        baseline,
        scenario,
        seed,
        policy=None,
        *,
        state_mode,
        latency_compensation,
        episode_options,
        **kwargs,
    ):
        del scenario, state_mode, kwargs
        return _sweep_rollout(
            baseline,
            seed,
            int(episode_options["state_delay_steps"]),
            latency_compensation=latency_compensation,
            policy=policy,
        )

    monkeypatch.setattr(
        "wheel_legged_control.d1.experiments.run_d1_rollout",
        fake_run,
    )


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
    assert args.latency_compensation == "none"
    assert not args.no_policy
    assert args.output is None


def test_d1_cli_can_disable_default_policy() -> None:
    args = build_parser().parse_args(["--state-mode", "estimated", "--no-policy"])

    assert args.state_mode == "estimated"
    assert args.no_policy


def test_d1_cli_accepts_latency_compensation() -> None:
    args = build_parser().parse_args(
        [
            "--state-mode",
            "estimated",
            "--latency-compensation",
            "constant_velocity",
            "--no-policy",
        ]
    )

    assert args.latency_compensation == "constant_velocity"


def test_wilson_interval_bounds_success_rate() -> None:
    estimate, lower, upper = _wilson_interval(8, 10)

    assert estimate == 0.8
    assert 0.0 < lower < estimate < upper < 1.0


def test_paired_intervals_need_two_pairs_and_bootstrap_is_deterministic() -> None:
    mean, lower, upper = _paired_ci(np.asarray((0.25,)))

    assert mean == 0.25
    assert np.isnan(lower)
    assert np.isnan(upper)

    values = np.asarray((-1.0, 0.0, 1.0, 1.0))
    first = _paired_bootstrap_ci(values, resamples=2_000)
    second = _paired_bootstrap_ci(values, resamples=2_000)
    assert first == second
    assert first[1] <= first[0] <= first[2]


def test_state_delay_sweep_writes_complete_paired_evidence(tmp_path, monkeypatch) -> None:
    _patch_sweep_rollout(monkeypatch)
    run_metadata = {
        "source": {"git_commit": "abc", "git_dirty": False},
        "checkpoint": {"included": False, "path": None, "sha256": None},
    }

    records = run_d1_state_delay_sweep(
        output=tmp_path,
        seed=10,
        episodes=2,
        policy=None,
        latency_compensation="constant_velocity",
        run_metadata=run_metadata,
    )

    assert len(records) == 2 * len(D1_STATE_DELAY_SWEEP_STEPS) * 2
    manifest = json.loads((tmp_path / "delay_sweep_manifest.json").read_text())
    config = json.loads((tmp_path / "evaluation_config.json").read_text())
    assert manifest["completed"] is True
    assert manifest["status"] == "complete"
    assert manifest["run_id"] == config["run_id"]
    assert manifest["provenance"] == config["provenance"]
    assert manifest["validation"]["paired_input_fingerprints"] is True
    assert manifest["validation"]["measured_state_age_traces"] is True
    assert set(manifest["artifacts"]) == {
        "delay_sweep_episodes.csv",
        "delay_sweep_summary.csv",
        "state_delay_sensitivity.md",
        "state_delay_sensitivity.png",
        "evaluation_config.json",
    }
    for filename, expected_sha256 in manifest["artifacts"].items():
        actual_sha256 = hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()
        assert actual_sha256 == expected_sha256
    assert not (tmp_path / ".delay_sweep_manifest.json.tmp").exists()

    with (tmp_path / "delay_sweep_summary.csv").open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    assert len(summary) == 2 * len(D1_STATE_DELAY_SWEEP_STEPS) * len(D1_SWEEP_METRICS)
    success_rows = [row for row in summary if row["metric"] == "success"]
    assert {row["ci_method"] for row in success_rows} == {"Wilson score"}
    assert {row["delta_ci_method"] for row in success_rows} == {"paired bootstrap"}

    markdown = (tmp_path / "state_delay_sensitivity.md").read_text(encoding="utf-8")
    assert "Episode duration" in markdown
    assert "trajectory prefix" in markdown
    assert "Raw position-estimation RMSE" in markdown


def test_one_pair_sweep_marks_difference_intervals_unavailable(tmp_path, monkeypatch) -> None:
    _patch_sweep_rollout(monkeypatch)

    run_d1_state_delay_sweep(
        output=tmp_path,
        seed=10,
        episodes=1,
        policy=None,
        latency_compensation="none",
    )

    with (tmp_path / "delay_sweep_summary.csv").open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    assert all(np.isnan(float(row["delta_ci95_low"])) for row in summary)
    markdown = (tmp_path / "state_delay_sensitivity.md").read_text(encoding="utf-8")
    assert "[unavailable]" in markdown


def test_state_delay_sweep_rejects_wrong_measured_age_and_clears_completion(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "delay_sweep_manifest.json").write_text('{"status":"complete"}')
    (tmp_path / "evaluation_config.json").write_text("stale")

    def wrong_age_rollout(
        baseline,
        scenario,
        seed,
        policy=None,
        *,
        latency_compensation,
        episode_options,
        **kwargs,
    ):
        del scenario, kwargs
        rollout = _sweep_rollout(
            baseline,
            seed,
            int(episode_options["state_delay_steps"]),
            latency_compensation=latency_compensation,
            policy=policy,
        )
        rollout.state_ages_ms = np.ones_like(rollout.state_ages_ms)
        return rollout

    monkeypatch.setattr(
        "wheel_legged_control.d1.experiments.run_d1_rollout",
        wrong_age_rollout,
    )

    with pytest.raises(ValueError, match="state-age trace"):
        run_d1_state_delay_sweep(
            output=tmp_path,
            seed=10,
            episodes=1,
            policy=None,
            latency_compensation="none",
        )
    assert not (tmp_path / "delay_sweep_manifest.json").exists()
    assert not (tmp_path / "evaluation_config.json").exists()


def test_policy_flags_conflict_and_explicit_missing_checkpoint_fails_before_output(
    tmp_path,
) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--policy", "model.zip", "--no-policy"])

    output = tmp_path / "results"
    with pytest.raises(SystemExit, match="policy checkpoint not found"):
        main(["--policy", str(tmp_path / "missing.zip"), "--output", str(output)])
    assert not output.exists()


def _patch_fast_benchmark(monkeypatch) -> None:
    def fake_rollout(baseline, scenario, seed, policy=None, **kwargs):
        rollout = _rollout()
        suffix = "+PPO" if policy is not None else ""
        rollout.controller = f"D1 {baseline.upper()}+VMC{suffix}"
        rollout.scenario = scenario
        rollout.evaluation_seed = seed
        rollout.state_estimation_mode = kwargs.get("state_mode", "oracle")
        rollout.latency_compensation = kwargs.get("latency_compensation", "none")
        return rollout

    def fake_plot(rollouts, output):
        (output / f"{rollouts[0].scenario}.png").write_bytes(b"plot")

    def fake_gif(rollouts, output):
        del rollouts
        (output / "d1_push_comparison.gif").write_bytes(b"gif")

    monkeypatch.setattr(
        "wheel_legged_control.d1.experiments.run_d1_rollout",
        fake_rollout,
    )
    monkeypatch.setattr(
        "wheel_legged_control.d1.experiments.plot_d1_scenario",
        fake_plot,
    )
    monkeypatch.setattr(
        "wheel_legged_control.d1.experiments.create_d1_comparison_gif",
        fake_gif,
    )
    monkeypatch.setattr(
        "wheel_legged_control.d1.experiments.capture_git_provenance",
        lambda path: {"git_commit": "abc", "git_dirty": False},
    )


def test_main_writes_complete_benchmark_manifest_last(tmp_path, monkeypatch) -> None:
    _patch_fast_benchmark(monkeypatch)

    main(["--no-policy", "--audit-episodes", "1", "--gif", "--output", str(tmp_path)])

    manifest = json.loads((tmp_path / "benchmark_manifest.json").read_text())
    config = json.loads((tmp_path / "evaluation_config.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["completed"] is True
    assert manifest["schema"] == "d1-benchmark-v1"
    assert config["schema"] == "d1-benchmark-evaluation-config-v1"
    assert manifest["run_id"] == config["run_id"]
    assert manifest["provenance"] == config["provenance"]
    assert manifest["provenance"]["source"] == {"git_commit": "abc", "git_dirty": False}
    assert manifest["provenance"]["checkpoint"]["included"] is False
    assert manifest["validation"]["matched_input_fingerprints"] is True
    assert manifest["validation"]["row_counts"] == {
        "metrics.csv": {"expected": 6, "actual": 6},
        "randomized_audit.csv": {"expected": 2, "actual": 2},
    }
    assert set(manifest["artifacts"]) == {
        "evaluation_config.json",
        "metrics.csv",
        "metrics.md",
        "nominal.png",
        "push.png",
        "mismatch_delay.png",
        "randomized_audit.csv",
        "randomized_audit.md",
        "d1_push_comparison.gif",
    }
    for filename, expected_sha256 in manifest["artifacts"].items():
        actual_sha256 = hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()
        assert actual_sha256 == expected_sha256
    assert not (tmp_path / ".benchmark_manifest.json.tmp").exists()


def test_main_removes_stale_completion_manifest_before_running(tmp_path, monkeypatch) -> None:
    (tmp_path / "benchmark_manifest.json").write_text('{"status":"complete"}')
    monkeypatch.setattr(
        "wheel_legged_control.d1.experiments.capture_git_provenance",
        lambda path: {"git_commit": "abc", "git_dirty": False},
    )
    monkeypatch.setattr(
        "wheel_legged_control.d1.experiments.run_d1_rollout",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("interrupted")),
    )

    with pytest.raises(RuntimeError, match="interrupted"):
        main(["--no-policy", "--output", str(tmp_path)])

    assert not (tmp_path / "benchmark_manifest.json").exists()


def test_main_rejects_truncated_metrics_before_completion(tmp_path, monkeypatch) -> None:
    _patch_fast_benchmark(monkeypatch)

    def write_truncated_metrics(records, output):
        write_d1_metrics(records, output)
        path = output / "metrics.csv"
        header = path.read_text(encoding="utf-8").splitlines()[0]
        path.write_text(header + "\n", encoding="utf-8")

    monkeypatch.setattr(
        "wheel_legged_control.d1.experiments.write_d1_metrics",
        write_truncated_metrics,
    )

    with pytest.raises(ValueError, match="record count"):
        main(["--no-policy", "--output", str(tmp_path)])

    assert not (tmp_path / "benchmark_manifest.json").exists()


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
