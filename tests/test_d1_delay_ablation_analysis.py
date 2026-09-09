"""Synthetic matrix plumbing; CSV/NPZ arithmetic has its own independent tests."""

import ast
import hashlib
import json
from pathlib import Path

import pytest

from scripts import analyze_d1_delay_ablation as analysis

GAINS = {
    "wheel_kp": 0.55,
    "wheel_ki": 1.5,
    "yaw_feedback_gain": 4.0,
    "leg_feedback_scale": 1.0,
    "attitude_feedback_scale": 0.25,
}
RANGES = {
    "measurement_delay_steps": [0, 2],
    "actuator_delay_steps": [0, 3],
    "actuator_time_constant_s": [0.0, 0.006],
    "actuator_gain": [0.95, 1.05],
}
DOMAINS = {
    key: 1.0
    for key in ("base_mass_scale", "damping_scale", "friction_scale", "actuator_strength_scale")
}
PREFIX = "/original/workspace/results/d1_v3_delay_ablation/"
NOISE = {"gyro_std_rad_s": 0.002}
QUALITY = {
    "velocity_rmse_mps": 0.08,
    "yaw_rmse_rps": 0.08,
    "height_rmse_m": 0.025,
    "attitude_rmse_rad": 0.15,
}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def mutate(path, action):
    value = json.loads(path.read_text())
    action(value)
    write(path, value)


def episode(arguments, *, road=0, episode_seed=17):
    randomized = arguments["delay_randomization"]
    return {
        "baseline": "wheel_leg",
        "duration_s": 60.0,
        "control_dt_s": 0.01,
        "physics_dt_s": 0.002,
        "terminal_reference_handling": "finite-envelope-exit-cached-terminal-observation-v1",
        "source_schema": "d1-synchronized-imu-encoder-fusion-v1",
        "controller_schema": "d1-wheel-leg-joint-pd-wheel-pi-yaw-feedback-configured-v4",
        "controller_parameters": GAINS,
        "domain": DOMAINS,
        "randomization": {
            **{key: [1.0, 1.0] for key in DOMAINS},
            **(RANGES if randomized else dict.fromkeys(RANGES)),
        },
        "provider": {
            "kind": "imu_encoder_fusion",
            "sensor_delay_steps": 1 if randomized else arguments["measurement_delay"],
            "sensor_noise": NOISE,
            "initial_position_m": [-3.8, 0.0, 0.455],
            "initial_rpy_rad": [0.0, 0.0, 0.0],
            "impairments": None,
        },
        "actuator": {
            "n_axes": 16,
            "physics_dt_s": 0.002,
            "torque_limit_nm": [80.0, 80.0, 80.0, 12.0] * 4,
            "delay_steps": 1 if randomized else 0,
            "gain": 1.0,
            "time_constant_s": 0.002 if randomized else 0.0,
        },
        "measurement_seed": episode_seed * 1000,
        "terrain": {"road": road},
    }


def make_run(study, label, mode, *, seed=24000, history=1, randomized=False, delay=0, policy=None):
    directory = study / label
    arguments = {
        "mode": mode,
        "baseline": "wheel_leg",
        "source": "imu_encoder_fusion",
        "duration": 60.0,
        "history": history,
        "delay_randomization": randomized,
        "measurement_delay": delay,
        "seed": seed,
        "steps": 32768,
        "workers": 4,
        "split": "development",
        "output": PREFIX + label,
        "policy": PREFIX + policy + "/checkpoint.zip" if policy else None,
        "metadata": PREFIX + policy + "/checkpoint.json" if policy else None,
        **GAINS,
    }
    protocol = {
        "schema": "d1-command-locomotion-experiment-v1",
        "arguments": arguments,
        "ppo_settings": {
            "learning_rate": 0.0003,
            "n_steps": 128,
            "batch_size": 128,
            "n_epochs": 4,
            "gamma": 0.9950124791926823,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "policy_kwargs": {"net_arch": [64, 64], "log_std_init": -2.0},
        },
        "quality_thresholds": QUALITY,
        "evaluation_seeds": {"development": [17, 29], "holdout": [617, 629]},
        "terrain_splits": {
            "train": [{"road": i} for i in range(4)],
            "development": [{"road": 0}, {"road": 1}],
        },
        "sensor_noise": NOISE,
        "versions": {"synthetic": "fixture"},
        "reward_scale": 1.0,
        "discount_horizon_s": 2.0,
        "selection_rule": "fixed final budget; no holdout checkpoint selection",
    }
    write(directory / "protocol.json", protocol)
    write(directory / "source.json", {"sha256": {"src/frozen.py": "a" * 64}})
    write(directory / "source_consistency.json", {"unchanged": True, "changed": []})
    if mode == "train":
        payload = ("Not a real model: " + label).encode()
        (directory / "checkpoint.zip").write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        write(
            directory / "training.json",
            {"num_timesteps": 32768, "updates": 256, "checkpoint_sha256": sha},
        )
        observation_schema = "d1-proprio-current-control82-v1"
        if history == 4:
            observation_schema += (
                "+history/v1;length=4;order=oldest_first;padding=repeat_initial_observation"
            )
        write(
            directory / "checkpoint.json",
            {
                "schema": "d1-locomotion-checkpoint-v1",
                "baseline": "wheel_leg",
                "model_sha256": sha,
                "extra": {"training_seed": seed, "num_timesteps": 32768},
                "history_length": history,
                "observation_dim": 82 * history,
                "action_dim": 8,
                "observation_schema": observation_schema,
                "source_schema": "d1-synchronized-imu-encoder-fusion-v1",
                "controller_schema": "d1-wheel-leg-joint-pd-wheel-pi-yaw-feedback-configured-v4",
                "controller_parameters": GAINS,
                "recorded_episode": episode(arguments),
            },
        )
    else:
        cases = []
        for road in (0, 1):
            for episode_seed in (17, 29):
                name = f"road{road}_seed{episode_seed}"
                cases.append(
                    {
                        "case": name,
                        "seed": episode_seed,
                        "terrain": {"road": road},
                        "schedule": {"synthetic": "same commands"},
                        "controller_parameters": GAINS,
                        "duration_s": 60.0,
                        "completed": True,
                        "quality_pass": True,
                        "terminal_reason": "time_limit",
                        "velocity_rmse_mps": 0.03,
                        "yaw_rmse_rps": 0.04,
                        "height_rmse_m": 0.01,
                        "attitude_rmse_rad": 0.02,
                        "mean_mechanical_power_w": 20.0,
                        "action_rms": 0.01,
                    }
                )
                write(
                    directory / f"{name}_episode.json",
                    episode(arguments, road=road, episode_seed=episode_seed),
                )
        write(directory / "evaluation.json", cases)
    command = ["python3", "scripts/run_d1_locomotion_experiment.py", mode]
    for key, value in arguments.items():
        if key == "mode" or value is None or value is False:
            continue
        command.append("--" + key.replace("_", "-"))
        if value is not True:
            command.append(str(value))
    return command


@pytest.fixture
def study(tmp_path, monkeypatch):
    root = tmp_path / "restored/results/d1_v3_delay_ablation"
    commands = []
    for arm, history, randomized in (("h1", 1, False), ("h4", 4, False), ("h4_dr", 4, True)):
        for seed in (27000, 28000, 29000):
            label = f"{arm}_seed{seed}"
            commands.append(
                make_run(root, label, "train", seed=seed, history=history, randomized=randomized)
            )
            for delay in (0, 2, 3):
                commands.append(
                    make_run(
                        root,
                        f"{label}_delay{delay}",
                        "evaluate",
                        history=history,
                        delay=delay,
                        policy=label,
                    )
                )
    for delay in (0, 2, 3):
        commands.append(make_run(root, f"zero_delay{delay}", "evaluate", delay=delay))
    write(
        root / "protocol.json",
        {
            "schema": "d1-delay-ablation-fixed-budget-v1",
            "training_seeds": [27000, 28000, 29000],
            "timesteps_per_model": 32768,
            "models": 9,
            "evaluation_measurement_delay_steps": [0, 2, 3],
            "control_dt_s": 0.01,
            "selection_rule": "fixed final checkpoint; retain all arms and seeds",
            "training_randomization": {
                "h1_and_h4": "fixed zero delay and ideal actuator",
                "h4_dr": RANGES,
                "actuator_dt_s": 0.002,
            },
            "commands": commands,
        },
    )

    def mock_source(directory):
        state = analysis.audit.read_json(directory / "source_consistency.json")
        analysis.require(state == {"unchanged": True, "changed": []}, "source changed during run")
        return {"file_count": 1, "archive_sha256": "b" * 64}

    def mock_evaluation(directory):
        return {
            "source": mock_source(directory),
            "cases": analysis.audit.read_json(directory / "evaluation.json"),
        }

    monkeypatch.setattr(analysis.audit, "audit_source", mock_source)
    monkeypatch.setattr(analysis.audit, "audit_evaluation", mock_evaluation)
    return root


def test_complete_matrix_is_read_only_and_accepts_relocated_workspace(study):
    before = {
        str(path.relative_to(study)): path.read_bytes()
        for path in study.rglob("*")
        if path.is_file()
    }
    result = analysis.analyze(study)
    assert len(result["training"]) == 9
    assert len(result["sources"]) == 39
    assert result["total_training_timesteps"] == 294912
    assert result["bootstrap_ci"] is None
    for arm in ("h1", "h4", "h4_dr"):
        for delay in ("0", "2", "3"):
            summary = result["by_arm"][arm][delay]
            assert summary["independent_training_seeds"] == 3
            assert (
                summary["evaluated_cases"]
                == summary["completed_cases"]
                == summary["quality_passes"]
                == 12
            )
            assert [item["training_seed"] for item in summary["per_training_seed"]] == [
                27000,
                28000,
                29000,
            ]
    assert [item["case_count"] for item in result["zero_by_delay"].values()] == [4, 4, 4]
    assert {
        str(path.relative_to(study)): path.read_bytes()
        for path in study.rglob("*")
        if path.is_file()
    } == before


def test_pair_labels_distinguish_episode_seed_from_sampled_measurement_seed(study):
    result = analysis.analyze(study)
    for contrast in result["paired_contrasts"].values():
        for pairs in contrast.values():
            for pair in pairs:
                expected_episode_seed = int(pair["case"].split("_seed")[1])
                assert pair["episode_seed"] == expected_episode_seed
                assert pair["measurement_seed"] == expected_episode_seed * 1000
                assert "sensor_seed" not in pair


def test_failures_are_not_averaged_into_full_horizon_metrics_or_paired_gains(study):
    path = study / "h4_seed27000_delay2/evaluation.json"

    def failure(cases):
        cases[0].update(
            completed=False,
            quality_pass=False,
            duration_s=3.0,
            terminal_reason="control_reference_out_of_envelope",
            velocity_rmse_mps=0.00001,
        )
        cases[1].update(quality_pass=False, velocity_rmse_mps=0.12)

    mutate(path, failure)
    result = analysis.analyze(study)
    item = result["by_arm"]["h4"]["2"]["per_training_seed"][0]
    assert item["completed_cases"] == 3 and item["quality_passes"] == 2
    assert item["full_horizon_metrics"]["velocity_rmse_mps"] == pytest.approx(
        (0.12 + 0.03 + 0.03) / 3
    )
    assert item["failure_duration_mean_s"] == 3.0
    assert item["terminal_reason_counts"] == {
        "control_reference_out_of_envelope": 1,
        "time_limit": 3,
    }
    pairs = result["paired_contrasts"]["h4_minus_h1"]["2"]
    assert pairs[0]["delta_right_minus_left"] is None
    assert (pairs[0]["left_duration_s"], pairs[0]["right_duration_s"]) == (60.0, 3.0)
    assert pairs[1]["delta_right_minus_left"]["velocity_rmse_mps"] == pytest.approx(0.09)
    # Two imperfect but complete policies still yield valid continuous deltas;
    # completion and quality are separate endpoints.
    assert not pairs[1]["right_quality_pass"]


def test_no_complete_cases_yields_null_full_horizon_metrics(study):
    def fail_all(cases):
        for case in cases:
            case.update(
                completed=False,
                quality_pass=False,
                duration_s=2.0,
                terminal_reason="fall_or_body_contact",
            )

    mutate(study / "zero_delay3/evaluation.json", fail_all)
    result = analysis.analyze(study)["zero_by_delay"]["3"]
    assert result["full_horizon_metrics"] is None
    assert result["failure_duration_mean_s"] == 2.0
    assert result["completed_cases"] == result["quality_passes"] == 0


@pytest.mark.parametrize(
    "fault",
    (
        "missing_run",
        "extra_run",
        "missing_case",
        "duplicate_case",
        "missing_command",
        "duplicate_command",
        "duplicate_flag",
    ),
)
def test_matrix_faults_are_rejected(study, fault):
    if fault == "missing_run":
        (study / "zero_delay3").rename(study / "unplanned_saved_copy")
    elif fault == "extra_run":
        write(study / "h4_seed30000/protocol.json", {})
    elif fault in ("missing_case", "duplicate_case"):
        mutate(
            study / "h1_seed27000_delay0/evaluation.json",
            lambda c: c.pop() if fault == "missing_case" else c.__setitem__(1, c[0]),
        )
    else:

        def change(document):
            if fault == "missing_command":
                document["commands"].pop()
            elif fault == "duplicate_command":
                document["commands"][1] = document["commands"][0]
            else:
                document["commands"][0] += ["--seed", "27000"]

        mutate(study / "protocol.json", change)
    with pytest.raises(ValueError):
        analysis.analyze(study)


@pytest.mark.parametrize(
    "field,value",
    (
        ("steps", 16384),
        ("workers", 3),
        ("seed", 27001),
        ("source", "oracle"),
        ("history", True),
        ("delay_randomization", True),
        ("attitude_feedback_scale", 1.0),
        ("wheel_kp", float("nan")),
        ("unrecorded_setting", 1),
    ),
)
def test_training_argument_corruption_is_rejected(study, field, value):
    mutate(study / "h1_seed27000/protocol.json", lambda p: p["arguments"].update({field: value}))
    with pytest.raises(ValueError):
        analysis.analyze(study)


@pytest.mark.parametrize(
    "fault",
    (
        "budget",
        "checkpoint_budget",
        "checkpoint_seed",
        "digest",
        "checkpoint_bytes",
        "history",
        "five_gains",
        "dr_ranges",
        "sampled_delay",
    ),
)
def test_training_checkpoint_and_domain_corruption_is_rejected(study, fault):
    directory = study / (
        "h4_dr_seed27000" if fault in ("dr_ranges", "sampled_delay") else "h1_seed27000"
    )
    if fault == "budget":
        mutate(directory / "training.json", lambda d: d.update(num_timesteps=100))
    elif fault == "checkpoint_bytes":
        (directory / "checkpoint.zip").write_bytes(b"replaced model")
    else:

        def corrupt(metadata):
            if fault == "checkpoint_budget":
                metadata["extra"]["num_timesteps"] = 100
            elif fault == "checkpoint_seed":
                metadata["extra"]["training_seed"] = 27001
            elif fault == "digest":
                metadata["model_sha256"] = "0" * 64
            elif fault == "history":
                metadata["history_length"] = 4
            elif fault == "five_gains":
                metadata["controller_parameters"].pop("leg_feedback_scale")
            elif fault == "dr_ranges":
                metadata["recorded_episode"]["randomization"]["measurement_delay_steps"] = [0, 3]
            else:
                metadata["recorded_episode"]["provider"]["sensor_delay_steps"] = 3

        mutate(directory / "checkpoint.json", corrupt)
    with pytest.raises(ValueError):
        analysis.analyze(study)


@pytest.mark.parametrize(
    "path",
    (
        "h1_seed27000/checkpoint.zip",
        "/other/study/h1_seed27000/checkpoint.zip",
        PREFIX + "h1_seed28000/checkpoint.zip",
        PREFIX + "../d1_v3_delay_ablation/h1_seed27000/checkpoint.zip",
        PREFIX + "h1_seed27000/checkpoint.json",
    ),
)
def test_policy_path_requires_full_study_namespace_and_correct_model(study, path):
    mutate(
        study / "h1_seed27000_delay0/protocol.json", lambda p: p["arguments"].update(policy=path)
    )
    with pytest.raises(ValueError, match="path"):
        analysis.analyze(study)


@pytest.mark.parametrize(
    "fault",
    (
        "delay",
        "actuator",
        "randomization",
        "measurement_seed",
        "schedule",
        "terminal_semantics",
        "source_changed",
        "cross_run_source",
    ),
)
def test_evaluation_domain_pairing_and_source_corruption_is_rejected(study, fault):
    directory = study / "h4_seed27000_delay2"
    if fault == "source_changed":
        write(
            directory / "source_consistency.json",
            {"unchanged": False, "changed": ["src/frozen.py"]},
        )
    elif fault == "cross_run_source":
        write(directory / "source.json", {"sha256": {"src/frozen.py": "c" * 64}})
    elif fault == "schedule":
        mutate(
            directory / "evaluation.json",
            lambda cases: cases[0].update(schedule={"synthetic": "different commands"}),
        )
    else:

        def corrupt(record):
            if fault == "delay":
                record["provider"]["sensor_delay_steps"] = 3
            elif fault == "actuator":
                record["actuator"]["delay_steps"] = 1
            elif fault == "randomization":
                record["randomization"]["measurement_delay_steps"] = [0, 2]
            elif fault == "measurement_seed":
                record["measurement_seed"] = 999
            else:
                record.pop("terminal_reference_handling")

        mutate(directory / "road0_seed17_episode.json", corrupt)
    with pytest.raises(ValueError):
        analysis.analyze(study)


def test_source_audit_failure_is_not_hidden(study, monkeypatch):
    def reject_source(directory):
        raise ValueError("archive digest mismatch")

    monkeypatch.setattr(analysis.audit, "audit_source", reject_source)
    with pytest.raises(ValueError, match="archive digest"):
        analysis.analyze(study)


@pytest.mark.parametrize("fault", ("measurement_seed", "schedule"))
def test_shared_zero_must_match_policy_scenario_and_noise(study, fault):
    directory = study / "zero_delay2"
    if fault == "measurement_seed":
        mutate(directory / "road0_seed17_episode.json", lambda e: e.update(measurement_seed=999))
    else:
        mutate(
            directory / "evaluation.json", lambda cases: cases[0].update(schedule={"changed": True})
        )
    with pytest.raises(ValueError, match="paired"):
        analysis.analyze(study)


@pytest.mark.parametrize("fault", ("model_count", "delay_range", "training_seed", "dr_range"))
def test_declared_study_protocol_cannot_silently_change(study, fault):
    def change(document):
        if fault == "model_count":
            document["models"] = 6
        elif fault == "delay_range":
            document["evaluation_measurement_delay_steps"] = [0, 2, 4]
        elif fault == "training_seed":
            document["training_seeds"][-1] = 30000
        else:
            document["training_randomization"]["h4_dr"]["actuator_gain"] = [0.9, 1.1]

    mutate(study / "protocol.json", change)
    with pytest.raises(ValueError, match="study"):
        analysis.analyze(study)


@pytest.mark.parametrize("field", ("sensor_noise", "terrain_splits", "ppo_settings"))
def test_runtime_protocol_cannot_change_between_runs(study, field):
    def change(protocol):
        if field == "sensor_noise":
            protocol[field]["gyro_std_rad_s"] = 0.0
        elif field == "terrain_splits":
            protocol[field]["development"][1]["road"] = 3
        else:
            protocol[field]["n_epochs"] = 8

    mutate(study / "h4_seed27000/protocol.json", change)
    with pytest.raises(ValueError):
        analysis.analyze(study)


def test_cli_refuses_existing_output_before_analysis(tmp_path, monkeypatch):
    output = tmp_path / "summary.json"
    output.write_text("keep existing report\n")
    monkeypatch.setattr(analysis, "analyze", lambda _: pytest.fail("analysis must not run"))
    with pytest.raises(ValueError, match="output must be new"):
        analysis.main([str(tmp_path), "--output", str(output)])
    assert output.read_text() == "keep existing report\n"


def test_cli_emits_strict_json(study, tmp_path):
    output = tmp_path / "summary.json"
    analysis.main([str(study), "--output", str(output)])
    assert json.loads(output.read_text())["schema"] == "d1-delay-ablation-analysis-v1"


def test_analysis_imports_no_policy_or_simulator():
    tree = ast.parse(Path(analysis.__file__).read_text())
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imports <= {"argparse", "json", "math", "numpy", "audit_d1_locomotion"}
    assert not any(
        name in Path(analysis.__file__).read_text()
        for name in ("PPO.load", "torch.load", "pickle.load")
    )
