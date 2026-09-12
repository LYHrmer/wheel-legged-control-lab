"""Audit a continuous-training checkpoint ladder using only NumPy and stdlib.

Checkpoint parameter hashes are linked to recorded post-training reload
witnesses. This reader never deserializes a Torch model or replays an optimizer.
Budgets on one trajectory are dependent observations, not extra training seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path, PurePosixPath

import numpy as np

try:
    from . import audit_d1_locomotion as audit
    from . import audit_d1_ppo_math as ppo_math
except ImportError:
    import audit_d1_locomotion as audit
    import audit_d1_ppo_math as ppo_math

require = audit.require
MODES = ["shared2", "independent8"]
GAINS = {
    "wheel_kp": 0.55,
    "wheel_ki": 1.5,
    "yaw_feedback_gain": 4.0,
    "leg_feedback_scale": 1.0,
    "attitude_feedback_scale": 0.25,
}
PPO_SETTINGS = {
    "learning_rate": 3e-4,
    "n_steps": 128,
    "batch_size": 128,
    "n_epochs": 4,
    "gamma": math.exp(-0.01 / 2.0),
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "policy_kwargs": {"net_arch": [64, 64], "log_std_init": -2.0},
}
EVALUATION_SEEDS = {"development": [1017, 1029], "holdout": [4617, 4629]}
EPISODE_SCHEMAS = {
    "baseline": "wheel_leg",
    "task_schema": "d1-fixed-road-command-task-v1",
    "control_schema": "d1-synchronized-control-loop-v1",
    "controller_schema": "d1-wheel-leg-joint-pd-wheel-pi-yaw-feedback-configured-v4",
    "reward_schema": "d1-command-tracking-rate-v1",
    "source_schema": "d1-synchronized-imu-encoder-fusion-v1",
    "observation_schema": "d1-proprio-current-control82-v1",
}
TERRAIN_FIELDS = (
    "layout",
    "slope_deg",
    "cross_slope_deg",
    "ripple_amplitude_m",
    "wavelength_x_m",
    "wavelength_y_m",
    "phase_x_rad",
    "phase_y_rad",
    "step_height_m",
)
TERRAIN_ROWS = {
    "train": [
        ("straight", 1.0, 0.5, 0.005, 0.6, 0.9, 0.0, 0.2, 0.005),
        ("straight", -1.5, -0.5, 0.01, 0.9, 1.2, 0.6, 0.8, 0.01),
        ("left_offset", 1.5, -1.0, 0.0075, 0.6, 1.2, 0.6, 0.2, 0.0075),
        ("left_offset", -1.0, 1.0, 0.005, 0.9, 0.9, 0.0, 0.8, 0.005),
    ],
    "development": [
        ("right_offset", 1.3, -0.7, 0.0065, 0.77, 1.07, 0.45, 0.65, 0.0065),
        ("right_offset", -1.3, 0.7, 0.007, 0.87, 1.17, 1.05, 1.25, 0.007),
    ],
    "holdout": [
        ("s_bend", 1.2, 0.55, 0.008, 0.74, 1.04, 1.7, 1.9, 0.008),
        ("diagonal", -1.2, -0.55, 0.0075, 0.84, 1.14, 2.3, 2.5, 0.0075),
    ],
}
TERRAINS = {
    split: [
        {**dict(zip(TERRAIN_FIELDS, row, strict=True)), "schema": "d1-fixed-2d-road-v1"}
        for row in rows
    ]
    for split, rows in TERRAIN_ROWS.items()
}


def equal(actual, expected, label):
    """JSON equality without bool/int or scalar type coercion."""
    require(type(actual) is type(expected), f"{label}: type differs")
    if isinstance(expected, dict):
        require(actual.keys() == expected.keys(), f"{label}: fields differ")
        for key in expected:
            equal(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        require(len(actual) == len(expected), f"{label}: length differs")
        for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
            equal(a, b, f"{label}[{index}]")
    else:
        require(actual == expected, f"{label}: value differs")


def fields(actual, expected, label):
    require(type(actual) is dict, f"{label}: not an object")
    for key, value in expected.items():
        require(key in actual, f"{label}: missing {key}")
        equal(actual[key], value, f"{label}.{key}")


def read_jsonl(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSONL key: {key}")
            result[key] = value
        return result

    def finite(value):
        value = float(value)
        require(math.isfinite(value), "nonfinite JSONL number")
        return value

    rows = []
    with path.open() as stream:
        for line in stream:
            require(bool(line.strip()), "empty JSONL row")
            row = json.loads(
                line, object_pairs_hook=pairs, parse_float=finite, parse_constant=finite
            )
            require(type(row) is dict, "JSONL row is not an object")
            rows.append(row)
    return rows


def recorded_path(value):
    """Validate lexical provenance without resolving a former machine's path."""
    require(
        type(value) is str and value and "\\" not in value and "\x00" not in value,
        "invalid recorded path",
    )
    path = PurePosixPath(value)
    require(path.is_absolute() and ".." not in path.parts, "unsafe recorded path")
    return path


def timestamp(value, label):
    require(type(value) is str, f"{label}: timestamp must be a string")
    parsed = datetime.fromisoformat(value)
    require(parsed.utcoffset() is not None, f"{label}: timestamp lacks timezone")
    return parsed


def nonnegative(value, label):
    require(
        type(value) in (float, int) and math.isfinite(value) and value >= 0,
        f"{label}: invalid elapsed seconds",
    )
    return float(value)


def expected_matrix(protocol):
    require(protocol.get("schema") == "d1-budget-study-v1", "unsupported budget study")
    require(protocol.get("kind") in ("formal", "smoke"), "unknown study kind")
    smoke = protocol["kind"] == "smoke"
    seeds = [31001] if smoke else [31000, 32000, 33000]
    budgets = [512, 1024] if smoke else [32768, 65536, 131072, 262144]
    total, duration = budgets[-1], 0.2 if smoke else 60.0
    fields(
        protocol,
        {
            "action_modes": MODES,
            "training_seeds": seeds,
            "checkpoint_budgets": budgets,
            "total_timesteps_per_model": total,
            "training_mode": "single_continuous_learn",
            "workers": 4,
            "n_steps": 128,
            "duration_s": duration,
            "history": 1,
            "source": "imu_encoder_fusion",
            "measurement_delay_steps": 0,
            "delay_randomization": False,
            "terrain_suite": "budget_compare_v1",
            "controller_parameters": GAINS,
            "ppo_settings": PPO_SETTINGS,
            "evaluation_seeds": EVALUATION_SEEDS,
        },
        "parent protocol",
    )
    models = [
        {
            "model_id": f"{mode}_seed{seed}",
            "action_mode": mode,
            "seed": seed,
            "training_run": f"{mode}_seed{seed}",
        }
        for mode in MODES
        for seed in seeds
    ]
    checkpoints = []
    for model in models:
        name = model["model_id"]
        for budget in budgets:
            directory = f"{name}/checkpoints/budget{budget}"
            checkpoints.append(
                {
                    "checkpoint_id": f"{name}_budget{budget}",
                    "model_id": name,
                    "budget": budget,
                    "training_run": name,
                    "relative_zip": f"{directory}/checkpoint.zip",
                    "relative_metadata": f"{directory}/checkpoint.json",
                    "expected_update_index": budget // 512 - 1,
                }
            )
    equal(protocol.get("models"), models, "models")
    equal(protocol.get("checkpoints"), checkpoints, "checkpoints")

    def zero(split):
        return {
            "name": f"zero_{split}",
            "mode": "evaluate",
            "action_mode": "independent8",
            "seed": 31000,
            "budget": None,
            "split": split,
            "phase": split,
            "model_name": None,
            "checkpoint_id": None,
        }

    runs = [zero("development")]
    runs.extend(
        {
            "name": m["model_id"],
            "mode": "train",
            "action_mode": m["action_mode"],
            "seed": m["seed"],
            "budget": total,
            "split": None,
            "phase": "train",
            "model_name": m["model_id"],
            "checkpoint_id": None,
        }
        for m in models
    )
    for split in ("development", "holdout"):
        if split == "holdout":
            runs.append(zero(split))
        for checkpoint in checkpoints:
            model = next(m for m in models if m["model_id"] == checkpoint["model_id"])
            runs.append(
                {
                    "name": f"{checkpoint['checkpoint_id']}_{split}",
                    "mode": "evaluate",
                    "action_mode": model["action_mode"],
                    "seed": model["seed"],
                    "budget": checkpoint["budget"],
                    "split": split,
                    "phase": split,
                    "model_name": model["model_id"],
                    "checkpoint_id": checkpoint["checkpoint_id"],
                }
            )
    equal(len(protocol.get("runs", [])), len(runs), "run count")
    for actual, expected in zip(protocol["runs"], runs, strict=True):
        fields(actual, expected, "planned run")
    fields(
        protocol,
        {
            "continuous_training_trajectories": len(models),
            "checkpoint_count": len(checkpoints),
            "planned_commands": len(runs),
            "planned_evaluation_cases": (len(runs) - len(models)) * 4,
            "total_training_transitions": len(models) * total,
            "expected_train_calls": len(models) * total // 512,
            "terrain_counts": {"train": 4, "development": 2, "holdout": 2},
            "terrains": TERRAINS,
        },
        "parent counts/terrains",
    )
    return models, checkpoints, runs


def common_arguments(arguments, protocol, run):
    expected = dict(
        baseline="wheel_leg",
        action_mode=run["action_mode"],
        source="imu_encoder_fusion",
        history=1,
        workers=4,
        duration=protocol["duration_s"],
        delay_randomization=False,
        measurement_delay=0,
        terrain_suite="budget_compare_v1",
        seed=run["seed"],
        steps=run["budget"] or protocol["total_timesteps_per_model"],
        mode=run["mode"],
        split=run["split"] or "development",
        **GAINS,
    )
    fields(arguments, expected, "child arguments")
    equal(
        arguments.get("checkpoint_budgets"),
        protocol["checkpoint_budgets"] if run["mode"] == "train" else None,
        "child checkpoint budgets",
    )


def audit_training_samples(directory, *, total, contract, budgets):
    """One row at a time, rather than retaining 262144 Python row dictionaries."""
    size = contract["policy_action_size"]
    indices = contract["policy_to_physical_indices"]
    totals = {
        "raw_square": 0.0,
        "clipped_square": 0.0,
        "clipped_coordinates": 0,
        "nonflat_steps": 0,
        "terminated_episodes": 0,
    }
    prefixes, count = {}, 0
    path = audit.artifact(directory, "training_samples.csv")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(
            reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames),
            "invalid training CSV columns",
        )
        for row in reader:
            require(None not in row and None not in row.values(), "malformed training CSV row")
            worker, sample = count % 4, 4 * (count // 4 + 1)
            require(
                row["worker"] == str(worker) and row["sample"] == str(sample),
                "training worker/sample sequence differs",
            )
            require(
                all(row[k] in ("True", "False") for k in ("nonflat_now", "terminated")),
                "invalid training boolean",
            )
            values = {
                k: float(v)
                for k, v in row.items()
                if k not in ("worker", "sample", "nonflat_now", "terminated")
            }
            require(all(math.isfinite(v) for v in values.values()), "nonfinite training sample")
            for prefix, width in (
                ("raw_action_", size),
                ("applied_action_", size),
                ("policy_raw_action_", size),
                ("policy_clipped_action_", size),
                ("physical_applied_action_", 8),
            ):
                columns = {k for k in row if k.startswith(prefix) and k[len(prefix) :].isdigit()}
                equal(columns, {f"{prefix}{i}" for i in range(width)}, "training action columns")
            raw = [values[f"raw_action_{i}"] for i in range(size)]
            clipped = [max(-1.0, min(1.0, v)) for v in raw]
            for i in range(size):
                require(
                    values[f"applied_action_{i}"] == clipped[i], "training policy clipping differs"
                )
                require(
                    values[f"policy_raw_action_{i}"] == raw[i]
                    and values[f"policy_clipped_action_{i}"] == clipped[i],
                    "training policy alias differs",
                )
            physical = [values[f"physical_applied_action_{i}"] for i in range(8)]
            require(
                physical == [clipped[i] for i in indices], "training physical broadcast differs"
            )
            squares = [sum(v * v for v in raw), sum(v * v for v in clipped)]
            changed = sum(a != b for a, b in zip(raw, clipped, strict=True))
            expected = {
                "raw_action_rms": math.sqrt(squares[0] / size),
                "applied_action_rms": math.sqrt(squares[1] / size),
                "policy_raw_action_rms": math.sqrt(squares[0] / size),
                "policy_clipped_action_rms": math.sqrt(squares[1] / size),
                "physical_applied_action_rms": math.sqrt(sum(v * v for v in physical) / 8),
                "raw_action_clipped_fraction": changed / size,
            }
            for key, value in expected.items():
                require(
                    math.isclose(values[key], value, rel_tol=2e-6, abs_tol=1e-8),
                    f"training {key} differs",
                )
            totals["raw_square"] += squares[0]
            totals["clipped_square"] += squares[1]
            totals["clipped_coordinates"] += changed
            totals["nonflat_steps"] += row["nonflat_now"] == "True"
            totals["terminated_episodes"] += row["terminated"] == "True"
            count += 1
            require(count <= total, "too many training samples")
            if count in budgets:
                prefixes[str(count)] = {
                    "samples": count,
                    "training_action_rms": math.sqrt(totals["raw_square"] / (count * size)),
                    "policy_clip_fraction": totals["clipped_coordinates"] / (count * size),
                    "nonflat_fraction": totals["nonflat_steps"] / count,
                    "terminated_episodes": totals["terminated_episodes"],
                }
    equal(count, total, "training transition count")
    return {"samples": count, "sha256": audit.digest(path), "prefixes": prefixes}


def audit_updates(directory, *, total, policy_size):
    rows = read_jsonl(audit.artifact(directory, "updates.jsonl"))
    equal(len(rows), total // 512, "PPO train-call count")
    for index, row in enumerate(rows):
        fields(
            row,
            {
                "audit_index": index,
                "num_timesteps": (index + 1) * 512,
                "n_updates": (index + 1) * PPO_SETTINGS["n_epochs"],
                "n_audit_samples": 128,
                "npz": f"sample_{index:06d}.npz",
            },
            "PPO update",
        )
        with np.load(audit.artifact(directory, row["npz"]), allow_pickle=False) as arrays:
            require(
                arrays["actions_raw"].shape == (128, policy_size),
                "PPO policy action dimension differs",
            )
            require(arrays["observations"].shape == (128, 82), "PPO observation dimension differs")
            require(arrays["gae_rewards"].shape == (128, 4), "PPO rollout dimensions differ")
            require(
                float(arrays["gae_gamma"]) == PPO_SETTINGS["gamma"]
                and float(arrays["gae_lambda"]) == PPO_SETTINGS["gae_lambda"],
                "PPO discounts differ",
            )
    report = ppo_math.audit_directory(directory)
    equal(report["updates_checked"], total // 512, "audited PPO train-call count")
    equal(
        {p.name for p in directory.iterdir()},
        {"updates.jsonl", *(r["npz"] for r in rows)},
        "PPO audit file inventory",
    )
    return report, rows


def compare_cases(left, right):
    first, second = {r["case"]: r for r in left}, {r["case"]: r for r in right}
    require(len(first) == len(left) and len(second) == len(right), "duplicate paired case")
    require(first.keys() == second.keys(), "paired case labels differ")
    pairs = []
    for name in sorted(first):
        a, b = first[name], second[name]
        for key in ("seed", "measurement_seed", "terrain", "schedule", "controller_parameters"):
            equal(a[key], b[key], f"paired {key}")
        both = a["completed"] and b["completed"]
        if both:
            equal(a["executed_steps"], b["executed_steps"], "paired completed horizons")
        pairs.append(
            {
                "case": name,
                "both_completed": both,
                "left_completed": a["completed"],
                "right_completed": b["completed"],
                "left_duration_s": a["duration_s"],
                "right_duration_s": b["duration_s"],
                "left_terminal_reason": a["terminal_reason"],
                "right_terminal_reason": b["terminal_reason"],
                "right_minus_left": {k: b[k] - a[k] for k in audit.METRICS} if both else None,
            }
        )
    return pairs


def summarize_cases(cases, duration):
    complete = [r for r in cases if r["completed"]]
    return {
        "cases": len(cases),
        "completed": len(complete),
        "failed": len(cases) - len(complete),
        "full_horizon_failures": sum(
            not r["completed"] and r["executed_steps"] == round(duration / audit.DT) for r in cases
        ),
        "quality_passes": sum(r["quality_pass"] for r in cases),
        "terminal_reason_counts": dict(Counter(r["terminal_reason"] for r in cases)),
        "completed_only_means": {k: float(np.mean([r[k] for r in complete])) for k in audit.METRICS}
        if complete
        else None,
        "all_case_duration_s": [r["duration_s"] for r in cases],
        "all_case_nonflat_fraction": [r["terrain_exposure"]["nonflat_fraction"] for r in cases],
    }


def audit_ledger(study, protocol):
    ledger = read_jsonl(audit.artifact(study, "runs.jsonl"))
    equal(len(ledger), len(protocol["runs"]), "completed ledger length")
    with audit.artifact(study, "runs.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(
            reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)),
            "invalid ledger CSV columns",
        )
        csv_rows = list(reader)
    equal(len(csv_rows), len(ledger), "CSV/JSON ledger count")
    previous_end, child_hashes = None, {}
    for index, (record, planned, csv_row) in enumerate(
        zip(ledger, protocol["runs"], csv_rows, strict=True)
    ):
        fields(
            record,
            {
                **planned,
                "sequence": index + 1,
                "returncode": 0,
                "log": f"logs/{planned['name']}.log",
            },
            "run ledger",
        )
        require(not record.get("error_type") and not record.get("error"), "ledger reports failure")
        start, end = (datetime.fromisoformat(record[key]) for key in ("started_utc", "ended_utc"))
        require(
            start.utcoffset() is not None and end.utcoffset() is not None and end >= start,
            "invalid run timestamps",
        )
        require(previous_end is None or start >= previous_end, "overlapping/reordered run phases")
        previous_end = end
        elapsed = record["elapsed_s"]
        require(
            type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0,
            "invalid run elapsed time",
        )
        audit.artifact(study, record["log"])
        declared = record["child_artifact_sha256"]
        names = {
            "protocol.json",
            "source.json",
            "source.tar.gz",
            "training.json" if planned["mode"] == "train" else "evaluation.json",
        }
        require(
            type(declared) is dict and declared.keys() == names,
            "child artifact ledger fields differ",
        )
        directory = study / planned["name"]
        for name, digest in declared.items():
            equal(audit.digest(audit.artifact(directory, name)), digest, "ledger child SHA")
        child_hashes[planned["name"]] = declared
        require(None not in csv_row and None not in csv_row.values(), "malformed ledger CSV")
        for key, value in csv_row.items():
            if key in ("command_json", "child_artifact_sha256_json"):
                source_key = "command" if key == "command_json" else "child_artifact_sha256"
                equal(json.loads(value), record[source_key], f"ledger CSV {key}")
            else:
                expected = record.get(key)
                require(
                    value == ("" if expected is None else str(expected)),
                    f"ledger CSV {key} differs",
                )
        require(
            {"sequence", "name", "returncode", "command_json", "child_artifact_sha256_json"}
            <= csv_row.keys(),
            "ledger CSV identity fields missing",
        )
    return {
        "completed_commands": len(ledger),
        "ledger_sha256": audit.digest(study / "runs.jsonl"),
        "csv_sha256": audit.digest(study / "runs.csv"),
        "child_artifact_sha256": child_hashes,
    }


def command_arguments(command, protocol, run):
    require(type(command) is list and all(type(v) is str and v for v in command), "invalid argv")
    prefix = [
        protocol["python_executable"],
        str(
            recorded_path(protocol["working_directory"]) / "scripts/run_d1_locomotion_experiment.py"
        ),
        run["mode"],
    ]
    equal(command[:3], prefix, "child command entrypoint")
    values, index = {}, 3
    while index < len(command):
        key = command[index]
        require(key.startswith("--") and key not in values, "duplicate or malformed argv option")
        index += 1
        stop = index
        while stop < len(command) and not command[stop].startswith("--"):
            stop += 1
        raw = command[index:stop]
        require(bool(raw), "missing argv value")
        require(key == "--checkpoint-budgets" or len(raw) == 1, "extra argv values")
        values[key] = raw if key == "--checkpoint-budgets" else raw[0]
        index = stop
    expected = {
        "--baseline": "wheel_leg",
        "--source": "imu_encoder_fusion",
        "--history": "1",
        "--measurement-delay": "0",
        "--workers": "4",
        "--duration": str(protocol["duration_s"]),
        "--terrain-suite": "budget_compare_v1",
        "--wheel-kp": "0.55",
        "--wheel-ki": "1.5",
        "--yaw-feedback-gain": "4",
        "--leg-feedback-scale": "1",
        "--attitude-feedback-scale": "0.25",
        "--action-mode": run["action_mode"],
        "--seed": str(run["seed"]),
        "--steps": str(run["budget"] or protocol["total_timesteps_per_model"]),
    }
    if run["mode"] == "train":
        expected["--checkpoint-budgets"] = [str(b) for b in protocol["checkpoint_budgets"]]
    else:
        expected["--split"] = run["split"]
    expected["--output"] = values.get("--output")
    output = recorded_path(expected["--output"])
    require(output.name == run["name"], "recorded run directory differs")
    if run["checkpoint_id"] is not None:
        checkpoint = output.parent / run["model_name"] / "checkpoints" / f"budget{run['budget']}"
        expected.update(
            {
                "--policy": str(checkpoint / "checkpoint.zip"),
                "--metadata": str(checkpoint / "checkpoint.json"),
            }
        )
    equal(values, expected, "child argv")
    return values


def audit_checkpoints(directory, model, checkpoints, protocol, training, update_rows):
    manifest_path = audit.artifact(directory, "checkpoints/manifest.json")
    manifest = audit.read_json(manifest_path)
    fields(
        manifest,
        {
            "schema": "d1-budget-checkpoints-v1",
            "budgets": protocol["checkpoint_budgets"],
            "pending_budgets": [],
            "complete": True,
            "failed": False,
            "failure": None,
        },
        "checkpoint manifest",
    )
    saved = manifest["saved"]
    equal(len(saved), len(checkpoints), "saved checkpoint count")
    reload_path = audit.artifact(directory, "checkpoint_reload.json")
    reload_doc = audit.read_json(reload_path)
    fields(
        reload_doc,
        {
            "schema": "d1-budget-checkpoint-reload-v1",
            "phase": "after_training",
            "training_final_num_timesteps": protocol["total_timesteps_per_model"],
            "training_finished_at_utc": training["training_finished_at_utc"],
        },
        "reload phase",
    )
    started = timestamp(training["training_started_at_utc"], "learn start")
    finished = timestamp(training["training_finished_at_utc"], "learn finish")
    require(finished >= started, "learning timestamps reversed")
    require(
        timestamp(reload_doc["started_at_utc"], "reload start") >= finished,
        "checkpoint reloaded before training finished",
    )
    nonnegative(reload_doc["elapsed_s"], "reload")
    wall = nonnegative(training["wall_s"], "training")
    witnesses = reload_doc["records"]
    equal(len(witnesses), len(checkpoints), "reload witness count")
    outputs, previous_elapsed, previous_saved_at = {}, 0.0, started
    for planned, record, witness in zip(checkpoints, saved, witnesses, strict=True):
        budget, index = planned["budget"], planned["expected_update_index"]
        update = update_rows[index]
        parameter_hash = update["param_sha256_after"]
        elapsed = nonnegative(record["elapsed_s"], "checkpoint elapsed")
        save_duration = nonnegative(record["save_duration_s"], "checkpoint save duration")
        saved_at = timestamp(record["saved_at_utc"], "checkpoint save")
        require(
            previous_elapsed <= elapsed <= wall + 1e-6 and save_duration <= elapsed + 1e-6,
            "checkpoint elapsed is outside continuous learn interval",
        )
        require(
            previous_saved_at <= saved_at <= finished, "checkpoint timestamp outside learn order"
        )
        previous_elapsed, previous_saved_at = elapsed, saved_at
        fields(
            record,
            {
                "budget": budget,
                "num_timesteps": budget,
                "after_update_index": index,
                "n_updates": (index + 1) * PPO_SETTINGS["n_epochs"],
                "param_sha256_after": parameter_hash,
                "model_path": f"budget{budget}/checkpoint.zip",
                "metadata_path": f"budget{budget}/checkpoint.json",
            },
            "saved checkpoint",
        )
        zip_path = audit.artifact(directory / "checkpoints", record["model_path"])
        metadata_path = audit.artifact(directory / "checkpoints", record["metadata_path"])
        zip_hash, metadata_hash = audit.digest(zip_path), audit.digest(metadata_path)
        fields(
            record,
            {"zip_sha256": zip_hash, "metadata_sha256": metadata_hash},
            "saved checkpoint hashes",
        )
        fields(
            witness,
            {
                "budget": budget,
                "after_update_index": index,
                "reloaded_policy_param_sha256": parameter_hash,
                "zip_sha256": zip_hash,
                "metadata_sha256": metadata_hash,
            },
            "actual reload witness",
        )
        metadata = audit.read_json(metadata_path)
        fields(metadata, EPISODE_SCHEMAS, "checkpoint schemas")
        fields(metadata["recorded_episode"], EPISODE_SCHEMAS, "checkpoint episode schemas")
        fields(
            metadata,
            {
                "schema": "d1-locomotion-checkpoint-v1",
                "model_sha256": zip_hash,
                "observation_dim": 82,
                "history_length": 1,
                "action_dim": 2 if model["action_mode"] == "shared2" else 8,
            },
            "checkpoint metadata",
        )
        fields(
            metadata["extra"],
            {
                "training_seed": model["seed"],
                "num_timesteps": budget,
                "checkpoint_budget": budget,
                "after_update_index": index,
                "param_sha256_after": parameter_hash,
            },
            "checkpoint extra",
        )
        args = dict(baseline="wheel_leg", action_mode=model["action_mode"], **GAINS)
        contract = audit.audit_action_contract(metadata, args)
        equal(
            metadata["action_space"],
            {
                "shape": [metadata["action_dim"]],
                "dtype": "float32",
                "low": [-1.0] * metadata["action_dim"],
                "high": [1.0] * metadata["action_dim"],
            },
            "checkpoint action space",
        )
        observation_space = metadata["observation_space"]
        fields(
            observation_space, {"shape": [82], "dtype": "float32"}, "checkpoint observation space"
        )
        low, high = np.asarray(observation_space["low"]), np.asarray(observation_space["high"])
        require(
            low.shape == high.shape == (82,)
            and np.isfinite(low).all()
            and np.isfinite(high).all()
            and np.all(low < high),
            "invalid checkpoint observation bounds",
        )
        equal(
            audit.audit_action_contract(metadata["recorded_episode"], args),
            contract,
            "checkpoint episode mapping",
        )
        equal(
            audit.audit_wheel_parameters(metadata, args), GAINS, "checkpoint controller parameters"
        )
        equal(
            audit.audit_wheel_parameters(metadata["recorded_episode"], args),
            GAINS,
            "checkpoint episode controller",
        )
        equal(
            {
                **{k: v for k, v in contract.items() if k != "modern_record"},
                "action_schema": metadata["action_schema"],
            },
            training["action_contract"],
            "checkpoint training action contract",
        )
        outputs[planned["checkpoint_id"]] = {
            "budget": budget,
            "after_update_index": index,
            "param_sha256_after": parameter_hash,
            "zip_sha256": zip_hash,
            "metadata_sha256": metadata_hash,
            "reloaded_policy_param_sha256": witness["reloaded_policy_param_sha256"],
            "parameter_hash_evidence": "post-training reload witness, not independent Torch ZIP deserialization",
        }
    phases = read_jsonl(audit.artifact(directory, "checkpoints/phases.jsonl"))
    expected_phases = []
    for index in range(len(update_rows)):
        expected_phases.append(("update", None))
        budget = (index + 1) * 512
        if budget in protocol["checkpoint_budgets"]:
            expected_phases.append(("save", budget))
    equal(len(phases), len(expected_phases), "checkpoint phase count")
    previous_perf, previous_wall, start_offsets = None, started, []
    measured = {"update": 0.0, "save": 0.0}
    for phase, (name, budget) in zip(phases, expected_phases, strict=True):
        fields(phase, {"phase": name, "budget": budget}, "checkpoint phase order")
        begin, end = (nonnegative(phase[k], k) for k in ("start_perf", "end_perf"))
        duration = nonnegative(phase["duration_s"], "phase duration")
        require(
            end >= begin and (previous_perf is None or begin >= previous_perf),
            "phase monotonic clocks overlap",
        )
        require(
            math.isclose(end - begin, duration, rel_tol=0, abs_tol=1e-8), "phase duration differs"
        )
        begin_wall = timestamp(phase["wall_start_utc"], "phase start")
        end_wall = timestamp(phase["wall_end_utc"], "phase end")
        require(
            previous_wall <= begin_wall <= end_wall <= finished, "phase UTC outside training order"
        )
        previous_perf, previous_wall = end, end_wall
        measured[name] += duration
        if name == "save":
            saved_record = saved[protocol["checkpoint_budgets"].index(budget)]
            require(
                math.isclose(saved_record["save_duration_s"], duration, rel_tol=0, abs_tol=1e-8),
                "saved duration differs from phase",
            )
            require(
                begin_wall <= timestamp(saved_record["saved_at_utc"], "saved at") <= end_wall,
                "saved timestamp outside save phase",
            )
            start_offsets.append(end - saved_record["elapsed_s"])
    require(
        all(math.isclose(v, start_offsets[0], rel_tol=0, abs_tol=1e-8) for v in start_offsets),
        "checkpoint cumulative elapsed does not share one learn start",
    )
    require(sum(measured.values()) <= wall + 1e-6, "measured phases exceed learn time")
    expected_files = {"manifest.json", "phases.jsonl"} | {
        f"budget{b}/{name}"
        for b in protocol["checkpoint_budgets"]
        for name in ("checkpoint.zip", "checkpoint.json")
    }
    actual_files = {
        str(p.relative_to(directory / "checkpoints"))
        for p in (directory / "checkpoints").rglob("*")
        if p.is_file()
    }
    equal(actual_files, expected_files, "checkpoint file inventory")
    final = reload_doc["final_checkpoint"]
    last = checkpoints[-1]
    final_parameter_hash = update_rows[-1]["param_sha256_after"]
    fields(
        final,
        {
            "model_path": "checkpoint.zip",
            "metadata_path": "checkpoint.json",
            "num_timesteps": protocol["total_timesteps_per_model"],
            "after_update_index": last["expected_update_index"],
            "reloaded_policy_param_sha256": final_parameter_hash,
        },
        "compatibility checkpoint",
    )
    final_zip = audit.digest(audit.artifact(directory, "checkpoint.zip"))
    final_meta_path = audit.artifact(directory, "checkpoint.json")
    final_meta_hash = audit.digest(final_meta_path)
    fields(
        final,
        {"zip_sha256": final_zip, "metadata_sha256": final_meta_hash},
        "final checkpoint hashes",
    )
    equal(training["checkpoint_sha256"], final_zip, "training final checkpoint hash")
    final_metadata = audit.read_json(final_meta_path)
    fields(
        final_metadata,
        {
            **EPISODE_SCHEMAS,
            "schema": "d1-locomotion-checkpoint-v1",
            "model_sha256": final_zip,
            "observation_dim": 82,
            "history_length": 1,
            "action_dim": 2 if model["action_mode"] == "shared2" else 8,
        },
        "final metadata",
    )
    fields(
        final_metadata["extra"],
        {"training_seed": model["seed"], "num_timesteps": protocol["total_timesteps_per_model"]},
        "final metadata extra",
    )
    equal(
        audit.audit_action_contract(final_metadata, args),
        contract,
        "final checkpoint action contract",
    )
    equal(audit.audit_wheel_parameters(final_metadata, args), GAINS, "final checkpoint controller")
    return {
        "checkpoints": outputs,
        "manifest_sha256": audit.digest(manifest_path),
        "reload_sha256": audit.digest(reload_path),
        "final_checkpoint": final,
        "phase_timing": {
            "phases": len(phases),
            "measured_seconds": measured,
            "sha256": audit.digest(directory / "checkpoints/phases.jsonl"),
        },
    }


def audit_training(directory, model, checkpoints, protocol, child):
    training = audit.read_json(audit.artifact(directory, "training.json"))
    total = protocol["total_timesteps_per_model"]
    size = 2 if model["action_mode"] == "shared2" else 8
    fields(
        training,
        {
            "num_timesteps": total,
            "learn_calls": 1,
            "training_mode": "single_continuous_learn",
            "checkpoint_budgets": protocol["checkpoint_budgets"],
            "policy_action_size": size,
            "physical_action_size": 8,
            "policy_parameter_count": 19141 if size == 2 else 19537,
            "updates": total // 512 * PPO_SETTINGS["n_epochs"],
        },
        "continuous training",
    )
    contract = audit.audit_action_contract(
        training["action_contract"],
        child["arguments"],
    )
    equal(
        audit.audit_action_contract(child["action_contract"], child["arguments"]),
        contract,
        "training/declared action contract",
    )
    episodes = []
    for worker in range(4):
        resets = read_jsonl(audit.artifact(directory, f"worker{worker}_episodes.jsonl"))
        require(resets, "training worker has no resets")
        for index, episode in enumerate(resets):
            fields(episode, EPISODE_SCHEMAS, "training episode schemas")
            fields(
                episode, {"episode": index, "terrain": TERRAINS["train"][worker]}, "worker episode"
            )
            fields(
                episode["provider"],
                {"kind": "imu_encoder_fusion", "sensor_delay_steps": 0},
                "worker provider",
            )
            equal(
                audit.audit_action_contract(episode, child["arguments"]),
                contract,
                "worker action contract",
            )
            equal(
                audit.audit_wheel_parameters(episode, child["arguments"]),
                GAINS,
                "worker controller",
            )
        episodes.append(len(resets))
    samples = audit_training_samples(
        directory, total=total, contract=contract, budgets=protocol["checkpoint_budgets"]
    )
    ppo_report, update_rows = audit_updates(directory / "updates", total=total, policy_size=size)
    checkpoint_report = audit_checkpoints(
        directory, model, checkpoints, protocol, training, update_rows
    )
    return dict(
        num_timesteps=total,
        learn_calls=1,
        worker_episode_counts=episodes,
        policy_parameter_count=training["policy_parameter_count"],
        training_samples=samples,
        ppo_math=ppo_report,
        **checkpoint_report,
    )


def analyze(study):
    raw_study = Path(study).absolute()
    require(not any(p.is_symlink() for p in (raw_study, *raw_study.parents)), "symlink study")
    study = raw_study.resolve()
    require(not (study / "failure.json").exists(), "study contains failure report")
    protocol = audit.read_json(audit.artifact(study, "protocol.json"))
    models, checkpoints, runs = expected_matrix(protocol)
    summary = audit.read_json(audit.artifact(study, "summary.json"))
    fields(
        summary,
        {
            "status": "commands_completed",
            "kind": protocol["kind"],
            "commands_completed": len(runs),
            "planned_commands": len(runs),
            "planned_evaluation_cases": protocol["planned_evaluation_cases"],
        },
        "study summary",
    )
    parent_source = audit.read_json(audit.artifact(study, "source.json"))
    before = parent_source["sha256"]
    equal(protocol.get("source_sha256"), before, "protocol source SHA")
    consistency = audit.read_json(audit.artifact(study, "source_consistency.json"))
    fields(consistency, {"unchanged": True, "sha256": before}, "source consistency")
    require(not consistency.get("changed"), "parent source changed")
    ledger = audit_ledger(study, protocol)
    training, evaluations, sources, common_child_source = {}, {}, {}, None
    source_union, recorded_study, common_noise = {}, None, None
    for run, planned in zip(runs, protocol["runs"], strict=True):
        directory = study / run["name"]
        require(not (directory / "failure.json").exists(), "child failure report")
        argv = command_arguments(planned["command"], protocol, run)
        output = recorded_path(argv["--output"])
        if recorded_study is None:
            recorded_study = output.parent
        require(output.parent == recorded_study, "recorded studies differ")
        child = audit.read_json(audit.artifact(directory, "protocol.json"))
        fields(
            child,
            {
                "schema": "d1-command-locomotion-experiment-v1",
                "ppo_settings": PPO_SETTINGS,
                "terrain_splits": TERRAINS,
                "terrain_suite": "budget_compare_v1",
                "evaluation_seeds": EVALUATION_SEEDS,
                "quality_thresholds": audit.QUALITY,
                "reward_scale": 1.0,
                "discount_horizon_s": 2.0,
            },
            "child protocol",
        )
        common_arguments(child["arguments"], protocol, run)
        for key in ("output", "policy", "metadata"):
            equal(child["arguments"].get(key), argv.get("--" + key), f"recorded child {key}")
        if common_noise is None:
            common_noise = child["sensor_noise"]
        equal(child["sensor_noise"], common_noise, "sensor noise changed across runs")
        hashes = audit.read_json(audit.artifact(directory, "source.json"))["sha256"]
        if common_child_source is None:
            common_child_source = hashes
        equal(hashes, common_child_source, "child source changed across runs")
        require(
            all(before.get(k) == v for k, v in hashes.items()),
            "child source not in parent inventory",
        )
        source_union.update(hashes)
        if run["mode"] == "train":
            sources[run["name"]] = audit.audit_source(directory)
            model = next(m for m in models if m["model_id"] == run["model_name"])
            selected = [c for c in checkpoints if c["model_id"] == model["model_id"]]
            training[run["name"]] = audit_training(directory, model, selected, protocol, child)
        else:
            result = audit.audit_evaluation(directory)
            sources[run["name"]] = result["source"]
            for case in result["cases"]:
                episode = audit.read_json(audit.artifact(directory, f"{case['case']}_episode.json"))
                fields(episode, EPISODE_SCHEMAS, "evaluation episode schemas")
                fields(
                    episode["provider"],
                    {"kind": "imu_encoder_fusion", "sensor_delay_steps": 0},
                    "evaluation provider",
                )
                seed = episode["measurement_seed"]
                require(type(seed) is int and seed >= 0, "invalid measurement seed")
                case["measurement_seed"] = seed
            evaluations[run["name"]] = result["cases"]
    extras = parent_source["archived_extras"]
    require(type(extras) is dict, "invalid parent source extras")
    for original, saved in extras.items():
        audit.safe_relative(original)
        digest = audit.digest(audit.artifact(study, saved))
        equal(digest, before.get(original), "parent source extra SHA")
        require(original not in source_union, "duplicate source extra")
        source_union[original] = digest
    equal(source_union, before, "parent source closure")
    comparisons = {}
    for split in ("development", "holdout"):
        zero = evaluations[f"zero_{split}"]
        per_seed = []
        for seed in protocol["training_seeds"]:
            by_budget = []
            previous = {}
            for budget in protocol["checkpoint_budgets"]:
                current = {
                    mode: evaluations[f"{mode}_seed{seed}_budget{budget}_{split}"] for mode in MODES
                }
                by_budget.append(
                    {
                        "budget": budget,
                        "modes": {
                            mode: summarize_cases(rows, protocol["duration_s"])
                            for mode, rows in current.items()
                        },
                        "independent8_minus_shared2": compare_cases(
                            current["shared2"], current["independent8"]
                        ),
                        "policy_minus_zero": {
                            mode: compare_cases(zero, rows) for mode, rows in current.items()
                        },
                        "current_minus_previous_budget": {
                            mode: compare_cases(previous[mode], rows)
                            for mode, rows in current.items()
                        }
                        if previous
                        else None,
                    }
                )
                previous = current
            per_seed.append({"training_seed": seed, "budgets": by_budget})
        comparisons[split] = {
            "zero_physical_baseline": summarize_cases(zero, protocol["duration_s"]),
            "per_seed": per_seed,
        }
    all_cases = [case for cases in evaluations.values() for case in cases]
    equal(len(all_cases), protocol["planned_evaluation_cases"], "audited evaluation case count")
    equal(
        sum(t["ppo_math"]["updates_checked"] for t in training.values()),
        protocol["expected_train_calls"],
        "total audited train calls",
    )
    return {
        "schema": "d1-budget-analysis-v1",
        "kind": protocol["kind"],
        "study": str(study),
        "parent_protocol_sha256": audit.digest(study / "protocol.json"),
        "parent_source_sha256": audit.digest(study / "source.json"),
        "source_files": len(before),
        "training_seeds": protocol["training_seeds"],
        "checkpoint_budgets": protocol["checkpoint_budgets"],
        "total_training_timesteps": protocol["total_training_transitions"],
        "checkpoint_count": len(checkpoints),
        "audited_train_calls": protocol["expected_train_calls"],
        "evaluation_summary": summarize_cases(all_cases, protocol["duration_s"]),
        "ledger": ledger,
        "source": sources,
        "training": training,
        "evaluations": evaluations,
        "comparisons": comparisons,
        "limitations": [
            "One continuous training trajectory per mode/seed; budgets are dependent checkpoints, not extra replicates.",
            "All predeclared checkpoints are reported, without best-checkpoint or best-seed selection.",
            "Only matched, equally long completed pairs receive RMSE/activity deltas; failures retain all durations and metrics.",
            "PPO audit covers every train-call record but only first-128 env-major likelihood samples; no optimizer replay.",
            "ZIP hashes are linked to recorded actual reload parameter hashes; no independent Torch model deserialization.",
            "CSV mechanical activity and geometric exposure labels are checked, not independently reconstructed contact or battery power.",
            "Smoke is an interface check, not performance evidence; no hardware or unknown-layout generalization claim.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output, study = args.output.absolute(), args.study.absolute()
    require(
        not output.exists() and not any(p.is_symlink() for p in (output, *output.parents)),
        "analysis output must be new and non-symlink",
    )
    output, study = output.resolve(), study.resolve()
    require(
        not output.is_relative_to(study) and not study.is_relative_to(output),
        "analysis output overlaps study",
    )
    report = analyze(args.study)
    output.mkdir(parents=True)
    with (output / "analysis.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for source in (Path(__file__), Path(audit.__file__), Path(ppo_math.__file__)):
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
            allow_nan=False,
        )
        stream.write("\n")
    return report


if __name__ == "__main__":
    main()
