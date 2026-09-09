"""Recompute recorded locomotion metrics without importing the simulator/trainer.

The CSV/NPZ arithmetic and source-archive checks adapt a Claude Opus draft.
Matching hashes establish byte consistency, not authenticity. Exposure flags
are recorded geometric labels; this audit does not reconstruct ground contact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tarfile
from pathlib import Path

import numpy as np

DT = 0.01
METRICS = (
    "velocity_rmse_mps",
    "yaw_rmse_rps",
    "height_rmse_m",
    "attitude_rmse_rad",
    "mean_mechanical_power_w",
    "action_rms",
)
QUALITY = dict(zip(METRICS[:4], (0.08, 0.08, 0.025, 0.15), strict=True))
TERMINAL_REASONS = {
    "time_limit",
    "fall_or_body_contact",
    "map_boundary",
    "ground_query_outside_map",
    "control_reference_out_of_envelope",
}
DEFAULT_WHEEL_PARAMETERS = {
    "wheel_kp": 2.2,
    "wheel_ki": 3.0,
    "yaw_feedback_gain": 4.0,
    "leg_feedback_scale": 1.0,
    "attitude_feedback_scale": 1.0,
}
UNSCALED_GAIN_FIELDS = {"wheel_kp", "wheel_ki", "yaw_feedback_gain"}
FEEDBACK_SCALE_FIELDS = {"leg_feedback_scale", "attitude_feedback_scale"}
DEFAULT_WHEEL_SCHEMA = "d1-wheel-leg-joint-pd-wheel-pi-yaw-feedback-v2"
CONFIGURED_WHEEL_SCHEMA = "d1-wheel-leg-joint-pd-wheel-pi-yaw-feedback-configured-v3"
SCALED_WHEEL_SCHEMA = "d1-wheel-leg-joint-pd-wheel-pi-yaw-feedback-configured-v4"
CASE_FIELDS = set(METRICS) | {
    "case",
    "seed",
    "terrain",
    "executed_steps",
    "duration_s",
    "terminal_reason",
    "completed",
    "quality_pass",
    "terrain_exposure",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"nonfinite JSON literal: {value}")


def _finite_float(value):
    result = float(value)
    require(math.isfinite(result), "nonfinite JSON number")
    return result


def read_json(path):
    return json.loads(
        Path(path).read_text(),
        object_pairs_hook=_pairs,
        parse_constant=_constant,
        parse_float=_finite_float,
    )


def digest(path):
    with Path(path).open("rb") as stream:
        return stream_digest(stream)


def stream_digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        value.update(block)
    return value.hexdigest()


def safe_relative(name):
    require(
        isinstance(name, str) and name and "\x00" not in name and "\\" not in name,
        "unsafe artifact path",
    )
    require(
        not name.startswith("/") and all(p not in ("", ".", "..") for p in name.split("/")),
        f"unsafe artifact path: {name}",
    )
    return name


def artifact(directory, name):
    safe_relative(name)
    path = directory / name
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink artifact")
    require(path.is_file(), f"missing artifact: {name}")
    return path


def close(actual, expected, label, *, atol=1e-10):
    require(
        not isinstance(actual, bool) and isinstance(actual, (int, float)), f"{label}: not numeric"
    )
    require(
        math.isfinite(actual) and math.isclose(actual, expected, abs_tol=atol, rel_tol=1e-8),
        f"{label}: declared {actual!r}, recomputed {expected!r}",
    )


def wheel_cli_parameters(arguments):
    """Decode CLI defaults without importing the runtime controller."""
    parameters = {
        key: default if arguments.get(key) is None else arguments[key]
        for key, default in DEFAULT_WHEEL_PARAMETERS.items()
    }
    validate_wheel_parameters(parameters)
    return {key: float(value) for key, value in parameters.items()}


def validate_wheel_parameters(parameters):
    require(
        type(parameters) is dict and parameters.keys() == DEFAULT_WHEEL_PARAMETERS.keys(),
        "controller parameter fields mismatch",
    )
    for key, value in parameters.items():
        require(
            type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f"invalid controller parameter: {key}",
        )
        if key == "wheel_kp" or key in FEEDBACK_SCALE_FIELDS:
            require(value > 0, f"controller parameter must be positive: {key}")


def audit_wheel_parameters(metadata, arguments):
    """Normalize historical records, then enforce the exact five-gain law.

    Missing gains mean only the historical fixed v2 controller. Three-gain
    v2/v3 records mean both bandwidth scales are one; a v4 record must carry
    all five fields. Schema and recorded values must agree with the CLI.
    """
    expected = wheel_cli_parameters(arguments)
    unscaled = all(expected[name] == 1.0 for name in FEEDBACK_SCALE_FIELDS)
    schema = metadata.get("controller_schema")
    expected_schema = (
        SCALED_WHEEL_SCHEMA
        if not unscaled
        else DEFAULT_WHEEL_SCHEMA
        if expected == DEFAULT_WHEEL_PARAMETERS
        else CONFIGURED_WHEEL_SCHEMA
    )
    require(schema == expected_schema, "controller schema does not match CLI parameters")
    if "controller_parameters" not in metadata:
        require(
            schema == DEFAULT_WHEEL_SCHEMA and expected == DEFAULT_WHEEL_PARAMETERS,
            "configured controller must record its parameters",
        )
        return dict(expected)
    declared = metadata["controller_parameters"]
    require(type(declared) is dict, "controller parameters must be an object")
    if declared.keys() == UNSCALED_GAIN_FIELDS:
        require(
            schema in (DEFAULT_WHEEL_SCHEMA, CONFIGURED_WHEEL_SCHEMA) and unscaled,
            "three-gain controller records require unscaled v2/v3 feedback",
        )
        declared = {**declared, **dict.fromkeys(FEEDBACK_SCALE_FIELDS, 1.0)}
    validate_wheel_parameters(declared)
    require(declared == expected, "episode controller parameters mismatch")
    return {key: float(declared[key]) for key in DEFAULT_WHEEL_PARAMETERS}


def audit_source(directory):
    consistency = read_json(artifact(directory, "source_consistency.json"))
    require(
        consistency.get("unchanged") is True and consistency.get("changed") == [],
        "source changed during run",
    )
    document = read_json(artifact(directory, "source.json"))
    manifest = document["sha256"]
    require(isinstance(manifest, dict) and manifest, "empty source manifest")
    for name, value in manifest.items():
        safe_relative(name)
        require(
            isinstance(value, str)
            and len(value) == 64
            and all(c in "0123456789abcdef" for c in value),
            "invalid source digest",
        )
    path = artifact(directory, "source.tar.gz")
    require(digest(path) == document["archive_sha256"], "archive digest mismatch")
    members = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            name = safe_relative(member.name)
            require(member.isfile(), "archive contains non-regular member")
            require(name not in members, "duplicate archive member")
            with archive.extractfile(member) as stream:
                members[name] = stream_digest(stream)
    require(members == manifest, "archive members differ from source manifest")
    return {"file_count": len(members), "archive_sha256": document["archive_sha256"]}


def audit_case(directory, record, arguments):
    name = safe_relative(record["case"])
    require("/" not in name, "case must be a basename")
    csv_path, npz_path = artifact(directory, name + ".csv"), artifact(directory, name + ".npz")
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(len(reader.fieldnames) == len(set(reader.fieldnames)), "duplicate CSV column")
        rows = list(reader)
    require(
        rows and all(None not in row and None not in row.values() for row in rows),
        "invalid CSV rows",
    )
    flags_text = [row["nonflat_now"] for row in rows]
    require(all(v in ("True", "False") for v in flags_text), "invalid exposure boolean")
    flags = np.asarray([v == "True" for v in flags_text])
    cols = {
        key: np.asarray([float(row[key]) for row in rows])
        for key in rows[0]
        if key != "nonflat_now"
    }
    require(all(np.isfinite(v).all() for v in cols.values()), "nonfinite CSV")
    n = len(rows)
    require(
        type(record["executed_steps"]) is int and record["executed_steps"] == n,
        "executed steps mismatch",
    )
    require(n <= round(arguments["duration"] / DT), "episode exceeds horizon")
    require(
        np.allclose(cols["time_s"], np.arange(1, n + 1) * DT, rtol=0, atol=1e-8), "clock mismatch"
    )
    close(record["duration_s"], n * DT, "duration", atol=1e-8)
    reason = record["terminal_reason"]
    require(reason in TERMINAL_REASONS, "unknown terminal reason")
    completed = reason == "time_limit"
    require(not completed or n == round(arguments["duration"] / DT), "premature time limit")
    require(record["completed"] is completed, "completed mismatch")
    require(
        np.all(cols["mechanical_power_w"] >= 0) and np.all(cols["action_mean_square"] >= 0),
        "negative power/action square",
    )
    for measurement, command, error in (
        ("velocity_mps", "command_forward_velocity_mps", "velocity_error_mps"),
        ("yaw_rate_rps", "command_yaw_rate_rps", "yaw_rate_error_rps"),
        ("clearance_m", "command_clearance_m", "height_error_m"),
    ):
        require(
            np.allclose(cols[measurement] - cols[command], cols[error], rtol=1e-8, atol=1e-10),
            f"recorded error mismatch: {error}",
        )
    require(
        np.allclose(
            cols["z_m"] - cols["ground_height_m"], cols["clearance_m"], rtol=1e-8, atol=1e-10
        ),
        "clearance mismatch",
    )
    rms = lambda key: float(np.sqrt(np.mean(cols[key] ** 2)))
    recomputed = {
        "velocity_rmse_mps": rms("velocity_error_mps"),
        "yaw_rmse_rps": rms("yaw_rate_error_rps"),
        "height_rmse_m": rms("height_error_m"),
        "attitude_rmse_rad": float(
            np.sqrt(np.mean(cols["roll_error_rad"] ** 2 + cols["pitch_error_rad"] ** 2))
        ),
        "mean_mechanical_power_w": float(np.mean(cols["mechanical_power_w"])),
        "action_rms": float(np.sqrt(np.mean(cols["action_mean_square"]))),
    }
    for key, value in recomputed.items():
        close(record[key], value, key)
    quality = completed and all(recomputed[key] <= limit for key, limit in QUALITY.items())
    require(record["quality_pass"] is quality, "quality mismatch")
    with np.load(npz_path, allow_pickle=False) as archive:
        qpos, actions = archive["qpos"], archive["actions"]
    axes = 8 if arguments["baseline"] == "wheel_leg" else 2
    require(qpos.shape == (n + 1, 23) and actions.shape == (n, axes), "NPZ shape mismatch")
    require(np.isfinite(qpos).all() and np.isfinite(actions).all(), "nonfinite NPZ")
    require(np.all(np.abs(actions) <= 1), "action outside normalized bounds")
    if not arguments.get("policy"):
        require(np.all(actions == 0), "zero residual run contains nonzero actions")
    require(
        np.allclose(
            np.mean(actions.astype(float) ** 2, axis=1),
            cols["action_mean_square"],
            rtol=2e-7,
            atol=1e-8,
        ),
        "NPZ actions differ from CSV",
    )
    require(np.allclose(qpos[0, :3], (-3.8, 0, 0.455), rtol=0, atol=1e-10), "initial pose mismatch")
    require(
        np.allclose(
            qpos[1:, :3],
            np.stack([cols[k] for k in ("x_m", "y_m", "z_m")], axis=1),
            rtol=0,
            atol=1e-10,
        ),
        "NPZ pose differs from CSV",
    )
    increments = np.linalg.norm(np.diff(qpos[:, :2], axis=0), axis=1)
    exposure = {
        "nonflat_steps": int(flags.sum()),
        "nonflat_fraction": float(flags.mean()),
        "nonflat_now": bool(flags[-1]),
        "path_m": float(increments.sum()),
        "nonflat_path_m": float(increments[flags].sum()),
        "unique_nonflat_0p25m_cells": len(np.unique(np.floor(qpos[1:, :2][flags] / 0.25), axis=0)),
    }
    for key, value in exposure.items():
        declared = record["terrain_exposure"][key]
        if type(value) in (bool, int):
            require(
                type(declared) is type(value) and declared == value, f"exposure mismatch: {key}"
            )
        else:
            close(declared, value, key)
    episode = read_json(artifact(directory, name + "_episode.json"))
    require(episode["terrain"] == record["terrain"], "episode terrain mismatch")
    require(episode["provider"]["kind"] == arguments["source"], "episode source mismatch")
    require(episode["baseline"] == arguments["baseline"], "episode baseline mismatch")
    parameters = None
    if arguments["baseline"] == "wheel_leg":
        parameters = audit_wheel_parameters(episode, arguments)
    close(episode["duration_s"], arguments["duration"], "episode duration")
    return {
        **record,
        **recomputed,
        "terrain_exposure": exposure,
        "csv_sha256": digest(csv_path),
        "npz_sha256": digest(npz_path),
        "schedule": episode["schedule"],
        "controller_parameters": parameters,
    }


def audit_evaluation(directory):
    directory = Path(directory)
    protocol = read_json(artifact(directory, "protocol.json"))
    require(protocol["schema"] == "d1-command-locomotion-experiment-v1", "unsupported protocol")
    arguments = protocol["arguments"]
    require(arguments["mode"] == "evaluate", "not an evaluation")
    duration = arguments["duration"]
    require(
        type(duration) in (int, float) and math.isfinite(duration) and duration >= DT,
        "invalid duration",
    )
    require(
        math.isclose(duration / DT, round(duration / DT), rel_tol=0, abs_tol=1e-8),
        "fractional horizon",
    )
    require(protocol["quality_thresholds"] == QUALITY, "quality thresholds changed")
    source = audit_source(directory)
    records = read_json(artifact(directory, "evaluation.json"))
    split = arguments["split"]
    seeds = protocol["evaluation_seeds"]["holdout" if split == "holdout" else "development"]
    terrains = protocol["terrain_splits"][split] if split != "flat" else None
    count = len(terrains) if terrains is not None else 1
    expected = {f"road{road}_seed{seed}" for road in range(count) for seed in seeds}
    require(
        isinstance(records, list) and len(records) == len(expected), "incomplete evaluation matrix"
    )
    require(
        all(isinstance(r, dict) and r.keys() == CASE_FIELDS for r in records),
        "case fields differ from the audited schema",
    )
    require({r["case"] for r in records} == expected, "case matrix mismatch")
    checked = []
    for record in records:
        road = int(record["case"].split("_")[0][4:])
        require(record["case"] == f"road{road}_seed{record['seed']}", "seed label mismatch")
        if terrains is not None:
            require(record["terrain"] == terrains[road], "protocol terrain mismatch")
        checked.append(audit_case(directory, record, arguments))
    return {
        "directory": str(directory),
        "source": source,
        "arguments": arguments,
        "cases": checked,
        "limitations": "CSV/NPZ arithmetic and archive consistency only; no independent dynamics, power or geometric-label reconstruction",
    }


def paired_deltas(policy, zero):
    """Oracle duplicates collapse only after byte-identical CSV and NPZ checks.

    Non-oracle cases remain paired individually. Incomplete or incompatible
    pairs retain their failure status but receive no full-horizon RMSE delta.
    """
    for key in ("baseline", "source", "duration", "split", "history", "measurement_delay"):
        require(
            policy["arguments"][key] == zero["arguments"][key], f"pair configuration differs: {key}"
        )
    require(not zero["arguments"]["policy"], "comparison is not zero residual")
    left, right = ({r["case"]: r for r in report["cases"]} for report in (policy, zero))
    require(left.keys() == right.keys(), "paired cases differ")
    paired, seen = [], {}
    for name in sorted(left):
        p, z = left[name], right[name]
        require(
            p.get("controller_parameters") == z.get("controller_parameters"),
            "paired controller parameters differ",
        )
        require(
            p["terrain"] == z["terrain"] and p["schedule"] == z["schedule"],
            "paired terrain/schedule mismatch",
        )
        terrain = json.dumps(p["terrain"], sort_keys=True)
        fingerprint = tuple(r[k] for r in (p, z) for k in ("csv_sha256", "npz_sha256"))
        if policy["arguments"]["source"] == "oracle" and terrain in seen:
            previous, item = seen[terrain]
            require(fingerprint == previous, "oracle seed trajectories are not identical")
            item["paired_case_labels"].append(name)
            continue
        comparable = p["completed"] and z["completed"]
        item = {
            "terrain": p["terrain"],
            "paired_case_labels": [name],
            "comparable_full_horizon": comparable,
            "policy_completed": p["completed"],
            "zero_completed": z["completed"],
            "policy_terminal_reason": p["terminal_reason"],
            "zero_terminal_reason": z["terminal_reason"],
            "policy_duration_s": p["duration_s"],
            "zero_duration_s": z["duration_s"],
            "policy_metrics": {k: p[k] for k in METRICS},
            "zero_metrics": {k: z[k] for k in METRICS},
            "policy_quality_pass": p["quality_pass"],
            "zero_quality_pass": z["quality_pass"],
            "policy_exposure": p["terrain_exposure"],
            "zero_exposure": z["terrain_exposure"],
            "delta_policy_minus_zero": {k: p[k] - z[k] for k in METRICS} if comparable else None,
        }
        paired.append(item)
        seen[terrain] = fingerprint, item
    return {
        "pairs": paired,
        "bootstrap_ci": None,
        "note": "No resampling interval at three training seeds. Terrain labels are not independent training replicas; actual paths/exposure may differ.",
    }


def audit_matrix(root):
    """The predeclared phase-4 endpoints, with equal weight for each road/seed."""
    root = Path(root)
    seeds = (24000, 25000, 26000)
    report = {"schema": "d1-phase4-independent-arithmetic-v1", "training": [], "splits": {}}
    for baseline in ("lqr", "wheel_leg"):
        for seed in seeds:
            directory = root / "d1_v3_locomotion_ppo" / f"{baseline}_seed{seed}"
            source = audit_source(directory)
            training = read_json(artifact(directory, "training.json"))
            metadata = read_json(artifact(directory, "checkpoint.json"))
            protocol = read_json(artifact(directory, "protocol.json"))
            args = protocol["arguments"]
            require(
                args["seed"] == seed and args["baseline"] == baseline, "training label mismatch"
            )
            require(
                args["steps"]
                == training["num_timesteps"]
                == metadata["extra"]["num_timesteps"]
                == 32768,
                "training budget mismatch",
            )
            require(
                args["source"] == "oracle" and args["history"] == 1 and args["workers"] == 4,
                "phase4 training configuration mismatch",
            )
            if baseline == "wheel_leg":
                require(
                    wheel_cli_parameters(args) == DEFAULT_WHEEL_PARAMETERS,
                    "phase4 wheel parameters changed",
                )
                audit_wheel_parameters(metadata, args)
                require(
                    type(metadata.get("recorded_episode")) is dict,
                    "checkpoint is missing its recorded episode",
                )
                audit_wheel_parameters(metadata["recorded_episode"], args)
            require(
                training["checkpoint_sha256"]
                == metadata["model_sha256"]
                == digest(artifact(directory, "checkpoint.zip")),
                "checkpoint digest mismatch",
            )
            report["training"].append(
                {"baseline": baseline, "seed": seed, **training, "source": source}
            )
    for split in ("development", "holdout"):
        section = {}
        for baseline in ("lqr", "wheel_leg"):
            zero_label = "wheel" if baseline == "wheel_leg" and split == "development" else baseline
            zero = audit_evaluation(root / f"d1_v3_locomotion_zero_{zero_label}_{split}")
            if baseline == "wheel_leg":
                require(
                    wheel_cli_parameters(zero["arguments"]) == DEFAULT_WHEEL_PARAMETERS,
                    "phase4 wheel parameters changed",
                )
            models = []
            for seed in seeds:
                directory = root / "d1_v3_locomotion_evaluation" / f"{baseline}_seed{seed}_{split}"
                audit = audit_evaluation(directory)
                if baseline == "wheel_leg":
                    require(
                        wheel_cli_parameters(audit["arguments"]) == DEFAULT_WHEEL_PARAMETERS,
                        "phase4 wheel parameters changed",
                    )
                require(
                    Path(audit["arguments"]["policy"]).parent.name == f"{baseline}_seed{seed}",
                    "evaluation checkpoint label mismatch",
                )
                models.append(
                    {
                        "training_seed": seed,
                        "source": audit["source"],
                        "comparison": paired_deltas(audit, zero),
                    }
                )
            pairs = [p for model in models for p in model["comparison"]["pairs"]]
            require(len(pairs) == 6, "phase4 expects three training seeds by two distinct roads")
            completed = [p for p in pairs if p["comparable_full_horizon"]]
            section[baseline] = {
                "models": models,
                "zero_source": zero["source"],
                "unique_roads": 2,
                "training_seeds": list(seeds),
                "full_horizon_pairs": len(completed),
                "policy_quality_passes": sum(p["policy_quality_pass"] for p in pairs),
                "zero_quality_passes_distinct_roads": sum(
                    p["zero_quality_pass"] for p in models[0]["comparison"]["pairs"]
                ),
                "mean_over_full_horizon_pairs": {
                    side: {
                        key: float(np.mean([p[f"{side}_metrics"][key] for p in completed]))
                        for key in METRICS
                    }
                    for side in ("policy", "zero")
                }
                if completed
                else None,
                "velocity_error_reduced_pairs": sum(
                    p["delta_policy_minus_zero"]["velocity_rmse_mps"] < 0 for p in completed
                ),
            }
        report["splits"][split] = section
    report["limitations"] = [
        "Two fixed roads per split; oracle noise-seed duplicates collapsed by byte checks.",
        "Only three independent training seeds, 32768 samples each; no confidence interval.",
        "LQR versus wheel_leg changes low-level structure too, not a pure action-dimension ablation.",
        "Report per-controller zero residual first; controller gains are not RL gains.",
        "Mechanical activity is not battery power. Exposure is not contact/traversal proof.",
        "Hashes establish current/archive byte consistency, not independent provenance.",
    ]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?")
    parser.add_argument("--matrix-root", type=Path)
    parser.add_argument("--zero", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.directory) == bool(args.matrix_root) or (args.matrix_root and args.zero):
        parser.error("choose directory [--zero] or --matrix-root")
    require(not args.output.exists(), "output must be new")
    report = (
        audit_matrix(args.matrix_root) if args.matrix_root else audit_evaluation(args.directory)
    )
    if args.zero:
        report["paired_comparison"] = paired_deltas(report, audit_evaluation(args.zero))
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        "checked frozen six-model matrix"
        if args.matrix_root
        else f"checked {len(report['cases'])} cases, {report['source']['file_count']} source files"
    )


if __name__ == "__main__":
    main()
