"""Recompute the same-base policy comparison without loading trained models.

Training seeds are the replication unit. Failed episodes retain their durations
and metrics, but never receive a full-horizon paired improvement claim.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from . import audit_d1_locomotion as audit
    from . import audit_d1_ppo_math as ppo_math
except ImportError:
    import audit_d1_locomotion as audit
    import audit_d1_ppo_math as ppo_math

MODES = ("shared2", "independent8")
GAINS = {
    "wheel_kp": 0.55,
    "wheel_ki": 1.5,
    "yaw_feedback_gain": 4.0,
    "leg_feedback_scale": 1.0,
    "attitude_feedback_scale": 0.25,
}
require = audit.require


def compare_cases(left, right):
    """Right minus left for equally long, matched completed episodes only."""
    first = {row["case"]: row for row in left}
    second = {row["case"]: row for row in right}
    require(len(first) == len(left) and len(second) == len(right), "duplicate paired case")
    require(first.keys() == second.keys(), "paired case labels differ")
    rows = []
    for name in sorted(first):
        a, b = first[name], second[name]
        for key in ("seed", "terrain", "schedule", "controller_parameters", "measurement_seed"):
            require(a[key] == b[key], f"paired {key} differs")
        complete = a["completed"] and b["completed"]
        if complete:
            require(a["executed_steps"] == b["executed_steps"], "paired completed horizons differ")
        rows.append(
            {
                "case": name,
                "both_completed": complete,
                "left_completed": a["completed"],
                "right_completed": b["completed"],
                "left_duration_s": a["duration_s"],
                "right_duration_s": b["duration_s"],
                "left_terminal_reason": a["terminal_reason"],
                "right_terminal_reason": b["terminal_reason"],
                "right_minus_left": {metric: b[metric] - a[metric] for metric in audit.METRICS}
                if complete
                else None,
            }
        )
    return rows


def summarize_cases(cases):
    complete = [row for row in cases if row["completed"]]
    return {
        "cases": len(cases),
        "completed": len(complete),
        "failed": len(cases) - len(complete),
        "terminal_reason_counts": dict(Counter(row["terminal_reason"] for row in cases)),
        "quality_passes": sum(row["quality_pass"] for row in cases),
        "completed_only_means": {
            metric: float(np.mean([row[metric] for row in complete])) for metric in audit.METRICS
        }
        if complete
        else None,
        "all_case_duration_s": [row["duration_s"] for row in cases],
        "all_case_nonflat_fraction": [row["terrain_exposure"]["nonflat_fraction"] for row in cases],
    }


def common_arguments(arguments, *, duration, steps, mode, seed):
    expected = {
        "baseline": "wheel_leg",
        "action_mode": mode,
        "source": "imu_encoder_fusion",
        "history": 1,
        "duration": duration,
        "steps": steps,
        "workers": 4,
        "delay_randomization": False,
        "measurement_delay": 0,
        "terrain_suite": "action_compare_v1",
        "seed": seed,
    }
    for key, value in expected.items():
        actual = arguments.get(key)
        require(type(actual) is type(value) and actual == value, f"study argument differs: {key}")
    require(all(key in arguments for key in GAINS), "study controller gain argument missing")
    require(audit.wheel_cli_parameters(arguments) == GAINS, "same-base controller gains differ")


def audit_training(directory, arguments, budget, expected_roads):
    source = audit.audit_source(directory)
    training = audit.read_json(directory / "training.json")
    metadata = audit.read_json(directory / "checkpoint.json")
    require(
        type(training["num_timesteps"]) is int and training["num_timesteps"] == budget,
        "training budget differs",
    )
    for key, expected in (("num_timesteps", budget), ("training_seed", arguments["seed"])):
        actual = metadata["extra"][key]
        require(type(actual) is int and actual == expected, f"checkpoint {key} differs")
    digest = audit.digest(directory / "checkpoint.zip")
    require(
        training["checkpoint_sha256"] == metadata["model_sha256"] == digest,
        "checkpoint hash differs",
    )
    require(
        type(metadata["observation_dim"]) is int
        and metadata["observation_dim"] == 82
        and type(metadata["history_length"]) is int
        and metadata["history_length"] == 1,
        "shared observation/history differs",
    )
    contract = audit.audit_action_contract(metadata, arguments)
    require(
        type(metadata["action_dim"]) is int
        and metadata["action_dim"] == contract["policy_action_size"],
        "saved policy dimension differs",
    )
    require(
        type(training["policy_parameter_count"]) is int and training["policy_parameter_count"] > 0,
        "invalid policy parameter count",
    )
    for key in ("policy_action_size", "physical_action_size"):
        require(
            type(training[key]) is int and training[key] == contract[key],
            "training action size differs",
        )
    for key, value in contract.items():
        if key != "modern_record":
            require(training["action_contract"][key] == value, "training action contract differs")
    require(audit.audit_wheel_parameters(metadata, arguments) == GAINS, "checkpoint gains differ")
    require(
        audit.audit_action_contract(metadata["recorded_episode"], arguments) == contract,
        "checkpoint episode mapping differs",
    )
    episode_counts = []
    for worker in range(4):
        records = [
            json.loads(line)
            for line in (directory / f"worker{worker}_episodes.jsonl").read_text().splitlines()
        ]
        require(records, "training worker has no recorded resets")
        for index, episode in enumerate(records):
            require(episode["episode"] == index, "worker episode order differs")
            require(episode["terrain"] == expected_roads[worker], "worker training road differs")
            require(
                episode["provider"]["kind"] == "imu_encoder_fusion", "training state source differs"
            )
            require(
                episode["provider"]["sensor_delay_steps"] == 0, "training measurement delay differs"
            )
            require(
                audit.audit_action_contract(episode, arguments) == contract,
                "worker action contract differs",
            )
            require(
                audit.audit_wheel_parameters(episode, arguments) == GAINS, "worker gains differ"
            )
        episode_counts.append(len(records))
    with (directory / "training_samples.csv").open(newline="") as stream:
        sample_rows = list(csv.DictReader(stream))
    require(len(sample_rows) == budget, "training transition count differs")
    size = contract["policy_action_size"]
    raw = np.asarray([[float(row[f"raw_action_{i}"]) for i in range(size)] for row in sample_rows])
    clipped = np.asarray(
        [[float(row[f"applied_action_{i}"]) for i in range(size)] for row in sample_rows]
    )
    require(np.isfinite(raw).all() and np.isfinite(clipped).all(), "nonfinite training actions")
    require(np.array_equal(np.clip(raw, -1, 1), clipped), "SB3 policy clipping differs")
    for prefix, expected_values in (("policy_raw_action", raw), ("policy_clipped_action", clipped)):
        alias = np.asarray(
            [[float(row[f"{prefix}_{i}"]) for i in range(size)] for row in sample_rows]
        )
        require(np.array_equal(alias, expected_values), "training policy action alias differs")
    physical = np.asarray(
        [
            [
                float(row[f"physical_applied_action_{i}"])
                for i in range(contract["physical_action_size"])
            ]
            for row in sample_rows
        ]
    )
    require(
        np.array_equal(physical, clipped[:, contract["policy_to_physical_indices"]]),
        "training physical action broadcast differs",
    )
    update_report = ppo_math.audit_directory(directory / "updates")
    expected_updates = budget // (4 * 128)
    require(update_report["updates_checked"] == expected_updates, "PPO train-call count differs")
    for index in range(expected_updates):
        with np.load(
            directory / "updates" / f"sample_{index:06d}.npz", allow_pickle=False
        ) as arrays:
            require(
                arrays["actions_raw"].shape == (128, size),
                "Gaussian probability uses physical rather than policy dimensions",
            )
    return {
        "source": source,
        "num_timesteps": budget,
        "policy_action_size": size,
        "physical_action_size": contract["physical_action_size"],
        "policy_parameter_count": training["policy_parameter_count"],
        "checkpoint_sha256": digest,
        "worker_episode_counts": episode_counts,
        "training_action_rms": float(np.sqrt(np.mean(raw**2))),
        "policy_clip_fraction": float(np.mean(raw != clipped)),
        "ppo_math": update_report,
    }


def analyze(study):
    study = Path(study).resolve()
    protocol = audit.read_json(study / "protocol.json")
    require(protocol["schema"] == "d1-shared-action-study-v1", "unsupported action study")
    smoke = protocol["kind"] == "smoke"
    require(protocol["kind"] in ("formal", "smoke"), "unknown study kind")
    seeds, steps, duration = ([31001], 512, 0.2) if smoke else ([31000, 32000, 33000], 32768, 60.0)
    require(
        protocol["training_seeds"] == seeds and protocol["action_modes"] == list(MODES),
        "predeclared seeds/modes differ",
    )
    require(
        protocol["timesteps_per_model"] == steps and protocol["duration_s"] == duration,
        "predeclared budget/horizon differs",
    )
    expected = {}
    for mode in MODES:
        for seed in seeds:
            name = f"{mode}_seed{seed}"
            expected[name] = ("train", mode, seed, "development")
            for split in ("development", "holdout"):
                expected[f"{name}_{split}"] = ("evaluate", mode, seed, split)
    for split in ("development", "holdout"):
        expected[f"zero_{split}"] = ("evaluate", "independent8", 31000, split)
    require(len(protocol["runs"]) == len(expected), "parent run count differs")
    require({row["name"] for row in protocol["runs"]} == set(expected), "parent run names differ")
    summary = audit.read_json(study / "summary.json")
    require(summary["status"] == "commands_completed", "study commands did not complete")
    for key in ("commands_completed", "planned_commands"):
        require(
            type(summary[key]) is int and summary[key] == len(expected),
            "parent command count differs",
        )
    parent_hashes = audit.read_json(study / "source.json")["sha256"]
    consistency = audit.read_json(study / "source_consistency.json")
    require(
        consistency["unchanged"] is True and consistency["sha256"] == parent_hashes,
        "parent source consistency differs",
    )
    training, evaluations, source_hashes = {}, {}, None
    for name, (kind, mode, seed, split) in expected.items():
        directory = study / name
        child = audit.read_json(directory / "protocol.json")
        args = child["arguments"]
        common_arguments(args, duration=duration, steps=steps, mode=mode, seed=seed)
        require(args["mode"] == kind and args["split"] == split, "child run kind/split differs")
        require(
            child["evaluation_seeds"] == {"development": [1017, 1029], "holdout": [1617, 1629]},
            "new evaluation reset seeds differ",
        )
        hashes = audit.read_json(directory / "source.json")["sha256"]
        require(
            all(parent_hashes.get(path) == digest for path, digest in hashes.items()),
            "child source differs from parent inventory",
        )
        if source_hashes is None:
            source_hashes = hashes
        require(hashes == source_hashes, "source changed across arms/runs")
        if kind == "train":
            training[name] = audit_training(
                directory, args, steps, child["terrain_splits"]["train"]
            )
        else:
            zero = name.startswith("zero_")
            require(bool(args["policy"]) is (not zero), "zero/policy run relabelled")
            if not zero:
                parent = study / f"{mode}_seed{seed}"
                require(
                    Path(args["policy"]).resolve() == (parent / "checkpoint.zip").resolve(),
                    "evaluation policy path differs",
                )
                require(
                    Path(args["metadata"]).resolve() == (parent / "checkpoint.json").resolve(),
                    "evaluation sidecar path differs",
                )
            result = audit.audit_evaluation(directory)
            for case in result["cases"]:
                episode = audit.read_json(directory / f"{case['case']}_episode.json")
                case["measurement_seed"] = episode["measurement_seed"]
            evaluations[name] = result["cases"]
    comparisons = {}
    for split in ("development", "holdout"):
        zero = evaluations[f"zero_{split}"]
        comparisons[split] = {
            "zero_physical_baseline": summarize_cases(zero),
            "per_seed": [
                {
                    "training_seed": seed,
                    "modes": {
                        mode: summarize_cases(evaluations[f"{mode}_seed{seed}_{split}"])
                        for mode in MODES
                    },
                    "independent8_minus_shared2": compare_cases(
                        evaluations[f"shared2_seed{seed}_{split}"],
                        evaluations[f"independent8_seed{seed}_{split}"],
                    ),
                    "policy_minus_zero": {
                        mode: compare_cases(zero, evaluations[f"{mode}_seed{seed}_{split}"])
                        for mode in MODES
                    },
                }
                for seed in seeds
            ],
        }
    return {
        "schema": "d1-shared-action-analysis-v1",
        "kind": protocol["kind"],
        "study": str(study),
        "parent_protocol_sha256": audit.digest(study / "protocol.json"),
        "training_seeds": seeds,
        "total_training_timesteps": len(training) * steps,
        "training": training,
        "evaluations": evaluations,
        "comparisons": comparisons,
        "limitations": [
            "Only three training seeds in the formal study; no confidence interval or universal action-dimension conclusion.",
            "Both modes share low-level control; output-layer parameter counts and physical-action correlations still differ.",
            "New evaluation terrain parameter instances reuse known layout families and existing command schedules.",
            "Completed-only means exclude failures and must be read with all-case durations and completion counts.",
            "Recorded Gaussian/GAE arithmetic is not an optimizer replay, dynamics verification, or hardware claim.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(
        not args.output.exists() and not args.output.is_symlink(), "analysis output must be new"
    )
    output, study = args.output.resolve(), args.study.resolve()
    require(
        not output.is_relative_to(study) and not study.is_relative_to(output),
        "analysis output overlaps study",
    )
    report = analyze(study)
    output.mkdir(parents=True)
    with (output / "analysis.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    sources = (Path(__file__), Path(audit.__file__), Path(ppo_math.__file__))
    for source in sources:
        with (output / source.name).open("xb") as stream:
            stream.write(source.read_bytes())
    with (output / "manifest.json").open("x") as stream:
        json.dump(
            {
                "sha256": {
                    p.name: audit.digest(p)
                    for p in sorted(output.iterdir())
                    if p.name != "manifest.json"
                }
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    return report


if __name__ == "__main__":
    main()
