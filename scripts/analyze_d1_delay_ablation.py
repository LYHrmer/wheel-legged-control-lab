"""Audit the frozen nine-model delay study without loading policies or simulators.

The arithmetic audit owns CSV/NPZ validation. This layer checks the experiment
matrix and keeps complete-episode comparisons separate from early failures.
Checkpoint hashes establish archive consistency, not historical authenticity.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

if __package__:
    from . import audit_d1_locomotion as audit
else:
    import audit_d1_locomotion as audit

SEEDS = (27000, 28000, 29000)
DELAYS = (0, 2, 3)
ARMS = {"h1": (1, False), "h4": (4, False), "h4_dr": (4, True)}
STUDY_NAME = "d1_v3_delay_ablation"
GAINS = {
    "wheel_kp": 0.55,
    "wheel_ki": 1.5,
    "yaw_feedback_gain": 4.0,
    "leg_feedback_scale": 1.0,
    "attitude_feedback_scale": 0.25,
}
DR_RANGES = {
    "measurement_delay_steps": [0, 2],
    "actuator_delay_steps": [0, 3],
    "actuator_time_constant_s": [0.0, 0.006],
    "actuator_gain": [0.95, 1.05],
}
DOMAINS = ("base_mass_scale", "damping_scale", "friction_scale", "actuator_strength_scale")
CASES = tuple(f"road{road}_seed{seed}" for road in (0, 1) for seed in (17, 29))
ARGUMENT_DEFAULTS = {
    "mode": None,
    "baseline": "wheel_leg",
    "source": "oracle",
    "duration": 60.0,
    "history": 1,
    "delay_randomization": False,
    "measurement_delay": 0,
    "seed": 24000,
    "steps": 32768,
    "workers": 4,
    "split": "development",
    "policy": None,
    "metadata": None,
    "output": None,
    **dict.fromkeys(GAINS),
}
INTEGER_ARGUMENTS = {"history", "measurement_delay", "seed", "steps", "workers"}
PPO_SETTINGS = {
    "learning_rate": 0.0003,
    "n_steps": 128,
    "batch_size": 128,
    "n_epochs": 4,
    "gamma": math.exp(-0.01 / 2.0),
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "policy_kwargs": {"net_arch": [64, 64], "log_std_init": -2.0},
}
TERMINAL_HANDLING = "finite-envelope-exit-cached-terminal-observation-v1"
require = audit.require


def same(actual, expected, label):
    """JSON equality with booleans excluded from numeric fields."""
    if isinstance(expected, dict):
        require(
            type(actual) is dict and actual.keys() == expected.keys(), f"{label}: fields differ"
        )
        for key, value in expected.items():
            same(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        require(type(actual) is list and len(actual) == len(expected), f"{label}: list differs")
        for index, value in enumerate(expected):
            same(actual[index], value, f"{label}[{index}]")
    elif type(expected) is float:
        require(
            type(actual) in (float, int) and math.isfinite(actual) and actual == expected,
            f"{label}: numeric value differs",
        )
    else:
        require(type(actual) is type(expected) and actual == expected, f"{label}: value differs")


def check_recorded_path(value, relative):
    require(type(value) is str, "recorded path must be a string")
    audit.safe_relative(value.removeprefix("/"))
    tail = ("results", STUDY_NAME, *Path(relative).parts)
    require(
        Path(value).parts[-len(tail) :] == tail, f"recorded path outside expected study: {value}"
    )


def expected_runs():
    runs = {}
    for arm, (history, randomized) in ARMS.items():
        for seed in SEEDS:
            label = f"{arm}_seed{seed}"
            base = {
                **ARGUMENT_DEFAULTS,
                **GAINS,
                "source": "imu_encoder_fusion",
                "history": history,
            }
            runs[label] = {
                **base,
                "mode": "train",
                "seed": seed,
                "delay_randomization": randomized,
                "output": label,
            }
            for delay in DELAYS:
                name = f"{label}_delay{delay}"
                runs[name] = {
                    **base,
                    "mode": "evaluate",
                    "measurement_delay": delay,
                    "output": name,
                    "policy": f"{label}/checkpoint.zip",
                    "metadata": f"{label}/checkpoint.json",
                }
    for delay in DELAYS:
        name = f"zero_delay{delay}"
        runs[name] = {
            **ARGUMENT_DEFAULTS,
            **GAINS,
            "source": "imu_encoder_fusion",
            "mode": "evaluate",
            "measurement_delay": delay,
            "output": name,
        }
    return runs


def check_arguments(arguments, expected):
    require(
        type(arguments) is dict and arguments.keys() == ARGUMENT_DEFAULTS.keys(),
        "run argument fields differ",
    )
    same(audit.wheel_cli_parameters(arguments), GAINS, "controller parameters")
    for key, value in expected.items():
        if key in GAINS:
            continue
        if key in ("output", "policy", "metadata") and value is not None:
            check_recorded_path(arguments[key], value)
        else:
            same(arguments[key], value, f"arguments.{key}")


def command_arguments(command):
    require(
        type(command) is list and len(command) >= 3 and all(type(v) is str for v in command),
        "invalid predeclared command",
    )
    same(command[1], "scripts/run_d1_locomotion_experiment.py", "command script")
    require(command[2] in ("train", "evaluate"), "unexpected command mode")
    result, seen, index = {**ARGUMENT_DEFAULTS, "mode": command[2]}, set(), 3
    while index < len(command):
        flag = command[index]
        require(flag.startswith("--"), "expected command option")
        key = flag[2:].replace("-", "_")
        require(
            key in ARGUMENT_DEFAULTS and key != "mode" and key not in seen,
            "unknown or duplicate command option",
        )
        seen.add(key)
        if key == "delay_randomization":
            result[key], index = True, index + 1
            continue
        require(index + 1 < len(command), "command option lacks value")
        value = command[index + 1]
        result[key] = (
            int(value)
            if key in INTEGER_ARGUMENTS
            else float(value)
            if key in GAINS or key == "duration"
            else value
        )
        index += 2
    return result


def check_study_protocol(document, runs):
    expected = {
        "schema": "d1-delay-ablation-fixed-budget-v1",
        "training_seeds": list(SEEDS),
        "timesteps_per_model": 32768,
        "models": 9,
        "evaluation_measurement_delay_steps": list(DELAYS),
        "control_dt_s": 0.01,
        "selection_rule": "fixed final checkpoint; retain all arms and seeds",
        "training_randomization": {
            "h1_and_h4": "fixed zero delay and ideal actuator",
            "h4_dr": DR_RANGES,
            "actuator_dt_s": 0.002,
        },
    }
    for key, value in expected.items():
        same(document.get(key), value, f"study.{key}")
    commands = document.get("commands")
    require(type(commands) is list and len(commands) == len(runs), "incomplete command matrix")
    seen = set()
    for command in commands:
        arguments = command_arguments(command)
        require(type(arguments["output"]) is str, "command must name output")
        name = Path(arguments["output"]).name
        require(name in runs and name not in seen, "unexpected or duplicate study run")
        check_arguments(arguments, runs[name])
        seen.add(name)


def check_episode(episode, arguments, noise, *, randomized=False):
    same(audit.audit_wheel_parameters(episode, arguments), GAINS, "episode gains")
    for key, value in {
        "baseline": "wheel_leg",
        "duration_s": 60.0,
        "control_dt_s": 0.01,
        "physics_dt_s": 0.002,
        "terminal_reference_handling": TERMINAL_HANDLING,
        "source_schema": "d1-synchronized-imu-encoder-fusion-v1",
        "domain": dict.fromkeys(DOMAINS, 1.0),
        "randomization": {
            **{key: [1.0, 1.0] for key in DOMAINS},
            **(DR_RANGES if randomized else dict.fromkeys(DR_RANGES)),
        },
    }.items():
        same(episode.get(key), value, f"episode.{key}")
    provider = episode["provider"]
    delay = provider["sensor_delay_steps"]
    require(type(delay) is int, "measurement delay must be an integer")
    if randomized:
        require(0 <= delay <= 2, "sampled measurement delay outside declared DR")
    else:
        same(delay, arguments["measurement_delay"], "episode measurement delay")
    same(
        provider,
        {
            "kind": "imu_encoder_fusion",
            "sensor_delay_steps": delay,
            "sensor_noise": noise,
            "initial_position_m": [-3.8, 0.0, 0.455],
            "initial_rpy_rad": [0.0, 0.0, 0.0],
            "impairments": None,
        },
        "episode provider",
    )
    actuator = episode["actuator"]
    expected = {
        "n_axes": 16,
        "physics_dt_s": 0.002,
        "torque_limit_nm": [80.0, 80.0, 80.0, 12.0] * 4,
        "delay_steps": 0,
        "gain": 1.0,
        "time_constant_s": 0.0,
    }
    if randomized:
        for name, low, high in (
            ("delay_steps", 0, 3),
            ("gain", 0.95, 1.05),
            ("time_constant_s", 0.0, 0.006),
        ):
            value = actuator[name]
            kinds = (int,) if name == "delay_steps" else (int, float)
            require(
                type(value) in kinds and math.isfinite(value) and low <= value <= high,
                f"sampled actuator {name} outside declared DR",
            )
            expected[name] = value
    same(actuator, expected, "episode actuator")
    require(
        type(episode.get("measurement_seed")) is int and episode["measurement_seed"] >= 0,
        "invalid measurement seed",
    )


def check_checkpoint(directory, arguments, protocol):
    training = audit.read_json(audit.artifact(directory, "training.json"))
    metadata = audit.read_json(audit.artifact(directory, "checkpoint.json"))
    for value in (training.get("num_timesteps"), metadata.get("extra", {}).get("num_timesteps")):
        same(value, 32768, "training budget")
    same(training.get("updates"), 256, "training epochs")
    same(metadata.get("extra", {}).get("training_seed"), arguments["seed"], "checkpoint seed")
    sha = audit.digest(audit.artifact(directory, "checkpoint.zip"))
    same(training.get("checkpoint_sha256"), sha, "training checkpoint SHA256")
    same(metadata.get("model_sha256"), sha, "sidecar checkpoint SHA256")
    same(audit.audit_wheel_parameters(metadata, arguments), GAINS, "checkpoint gains")
    history = arguments["history"]
    schema = "d1-proprio-current-control82-v1"
    if history > 1:
        schema += "+history/v1;length=4;order=oldest_first;padding=repeat_initial_observation"
    for key, expected in {
        "schema": "d1-locomotion-checkpoint-v1",
        "baseline": "wheel_leg",
        "history_length": history,
        "observation_dim": 82 * history,
        "action_dim": 8,
        "observation_schema": schema,
        "source_schema": "d1-synchronized-imu-encoder-fusion-v1",
    }.items():
        same(metadata.get(key), expected, f"checkpoint.{key}")
    episode = metadata["recorded_episode"]
    check_episode(
        episode, arguments, protocol["sensor_noise"], randomized=arguments["delay_randomization"]
    )
    same(episode["terrain"], protocol["terrain_splits"]["train"][0], "checkpoint reference terrain")
    return {"training_seed": arguments["seed"], "num_timesteps": 32768, "checkpoint_sha256": sha}


def summarize_cases(cases):
    complete = [case for case in cases if case["completed"]]
    failures = [case for case in cases if not case["completed"]]
    return {
        "case_count": len(cases),
        "completed_cases": len(complete),
        "quality_passes": sum(case["quality_pass"] for case in cases),
        "terminal_reason_counts": dict(
            sorted(Counter(case["terminal_reason"] for case in cases).items())
        ),
        "full_horizon_metrics": {
            key: float(np.mean([case[key] for case in complete])) for key in audit.METRICS
        }
        if complete
        else None,
        "failed_cases": [
            {
                "case": case["case"],
                "duration_s": case["duration_s"],
                "terminal_reason": case["terminal_reason"],
            }
            for case in failures
        ],
        "failure_duration_mean_s": float(np.mean([case["duration_s"] for case in failures]))
        if failures
        else None,
    }


def matched_cases(left, right):
    left, right = ({case["case"]: case for case in cases} for cases in (left, right))
    same(sorted(left), sorted(right), "paired cases")
    for name in sorted(left):
        first, second = left[name], right[name]
        for key in ("seed", "terrain", "schedule", "measurement_seed", "controller_parameters"):
            same(first[key], second[key], f"paired {key}")
        yield name, first, second


def compare_cases(left, right, training_seed):
    pairs = []
    for name, first, second in matched_cases(left, right):
        complete = first["completed"] and second["completed"]
        pairs.append(
            {
                "training_seed": training_seed,
                "case": name,
                "episode_seed": first["seed"],
                "measurement_seed": first["measurement_seed"],
                "comparable_full_horizon": complete,
                "left_completed": first["completed"],
                "right_completed": second["completed"],
                "left_quality_pass": first["quality_pass"],
                "right_quality_pass": second["quality_pass"],
                "left_duration_s": first["duration_s"],
                "right_duration_s": second["duration_s"],
                "left_terminal_reason": first["terminal_reason"],
                "right_terminal_reason": second["terminal_reason"],
                "delta_right_minus_left": {key: second[key] - first[key] for key in audit.METRICS}
                if complete
                else None,
            }
        )
    return pairs


def analyze(study):
    study = Path(study)
    runs = expected_runs()
    check_study_protocol(audit.read_json(audit.artifact(study, "protocol.json")), runs)
    missing = [name for name in runs if not (study / name).is_dir()]
    require(not missing, f"incomplete study, missing runs: {missing}")
    extras = [
        p.name
        for p in study.iterdir()
        if p.is_dir() and (p / "protocol.json").exists() and p.name not in runs
    ]
    require(not extras, f"unplanned study runs: {extras}")
    training, evaluations, sources = {}, {}, {}
    frozen_source, frozen_protocol = None, None
    for name, expected in runs.items():
        directory = study / name
        protocol = audit.read_json(audit.artifact(directory, "protocol.json"))
        arguments = protocol["arguments"]
        check_arguments(arguments, expected)
        same(protocol.get("schema"), "d1-command-locomotion-experiment-v1", "run schema")
        same(protocol.get("ppo_settings"), PPO_SETTINGS, "PPO settings")
        same(protocol.get("quality_thresholds"), audit.QUALITY, "quality thresholds")
        same(
            protocol.get("evaluation_seeds"),
            {"development": [17, 29], "holdout": [617, 629]},
            "evaluation seeds",
        )
        invariant = {
            key: protocol[key]
            for key in (
                "terrain_splits",
                "sensor_noise",
                "versions",
                "reward_scale",
                "discount_horizon_s",
                "selection_rule",
            )
        }
        if frozen_protocol is None:
            frozen_protocol = invariant
            require(
                len(protocol["terrain_splits"]["development"]) == 2,
                "expected two development roads",
            )
        same(invariant, frozen_protocol, "cross-run protocol")
        if arguments["mode"] == "train":
            sources[name] = audit.audit_source(directory)
            training[name] = check_checkpoint(directory, arguments, protocol)
        else:
            checked = audit.audit_evaluation(directory)
            sources[name] = checked["source"]
            cases = checked["cases"]
            same(sorted(case["case"] for case in cases), sorted(CASES), "evaluation case matrix")
            for case in cases:
                episode = audit.read_json(audit.artifact(directory, case["case"] + "_episode.json"))
                check_episode(episode, arguments, protocol["sensor_noise"])
                case["measurement_seed"] = episode["measurement_seed"]
            evaluations[name] = cases
        manifest = audit.read_json(audit.artifact(directory, "source.json"))["sha256"]
        if frozen_source is None:
            frozen_source = manifest
        same(manifest, frozen_source, "cross-run frozen source")
    by_arm = {}
    for arm in ARMS:
        by_arm[arm] = {}
        for delay in DELAYS:
            # The public zero baseline must describe the same scenario/noise
            # stream. Validate once per comparison, but count its four cases
            # only in zero_by_delay, never as twelve independent zero episodes.
            for seed in SEEDS:
                list(
                    matched_cases(
                        evaluations[f"zero_delay{delay}"],
                        evaluations[f"{arm}_seed{seed}_delay{delay}"],
                    )
                )
            per_seed = [
                {
                    "training_seed": seed,
                    **summarize_cases(evaluations[f"{arm}_seed{seed}_delay{delay}"]),
                }
                for seed in SEEDS
            ]
            by_arm[arm][str(delay)] = {
                "independent_training_seeds": 3,
                "evaluated_cases": 12,
                "completed_cases": sum(item["completed_cases"] for item in per_seed),
                "quality_passes": sum(item["quality_passes"] for item in per_seed),
                "per_training_seed": per_seed,
            }
    contrasts = {}
    for left, right in (("h1", "h4"), ("h4", "h4_dr")):
        contrasts[f"{right}_minus_{left}"] = {
            str(delay): [
                pair
                for seed in SEEDS
                for pair in compare_cases(
                    evaluations[f"{left}_seed{seed}_delay{delay}"],
                    evaluations[f"{right}_seed{seed}_delay{delay}"],
                    seed,
                )
            ]
            for delay in DELAYS
        }
    return {
        "schema": "d1-delay-ablation-analysis-v1",
        "study": str(study),
        "study_protocol_sha256": audit.digest(audit.artifact(study, "protocol.json")),
        "analysis_script_sha256": audit.digest(Path(__file__)),
        "training_seeds": list(SEEDS),
        "total_training_timesteps": 9 * 32768,
        "evaluation_delays_s": [delay * 0.01 for delay in DELAYS],
        "controller_parameters": GAINS,
        "training": training,
        "sources": sources,
        "by_arm": by_arm,
        "zero_by_delay": {
            str(delay): summarize_cases(evaluations[f"zero_delay{delay}"]) for delay in DELAYS
        },
        "paired_contrasts": contrasts,
        "bootstrap_ci": None,
        "limitations": [
            "Three independent training seeds per arm, not twelve independent training replicas. No bootstrap interval.",
            "Full-horizon means use only completed 60 s cases; early failures retain durations and receive no paired continuous delta.",
            "Each shared zero baseline has four cases and is not replicated for the three policies.",
            "Development roads informed controller choice; no fresh terrain holdout or hardware validation.",
            "DR changes both measurement and actuator channels; this study cannot isolate their individual effects.",
            "Evaluation protocols retain external policy paths, not per-evaluation model snapshots. Current checkpoint/sidecar hashes and expected study paths do not independently prove the bytes loaded at evaluation time.",
            "Source hashes establish archive byte consistency, not authenticity, optimizer correctness or independent dynamics reconstruction.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists() and not args.output.is_symlink(), "output must be new")
    report = analyze(args.study)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print("checked nine training models, 108 policy cases and 12 shared zero cases")


if __name__ == "__main__":
    main()
