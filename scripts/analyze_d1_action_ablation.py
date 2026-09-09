"""Independently audit action-study samples and compare all fixed-budget candidates.

This reads completed training and holdout artifacts. It neither trains nor
simulates a robot, and never writes into an input directory. Arithmetic below
does not call the training/evaluation runner's metric functions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import tarfile
import zipfile
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (9000, 10000, 11000)
VARIANTS = ("full_gaussian", "bounded_mean", "fx_only")
BUDGETS = (32768, 65536, 131072)
FIXED_INDICES = (0, 4, 5, 10, 11, 22)
DT = 0.01
MEASURES = ("velocity_rmse_mps", "tail_velocity_rmse_mps", "clearance_rmse_m", "episode_return")
PPO_SETTINGS = {
    "learning_rate": 3e-4,
    "n_steps": 128,
    "batch_size": 128,
    "n_epochs": 4,
    "gamma": math.exp(-DT / 2),
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "policy_kwargs": {"net_arch": [64, 64]},
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def column(rows, field):
    value = np.asarray([float(row[field]) for row in rows])
    if not np.isfinite(value).all():
        raise ValueError(f"nonfinite {field}")
    return value


def close(actual, expected, label, tolerance=1e-9):
    difference = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    if not math.isfinite(difference) or difference > tolerance:
        raise ValueError(f"{label} mismatch: maximum absolute error {difference:g}")
    return difference


def audit_manifest(root):
    """Require the complete raw artifact set, including every CSV and checkpoint."""
    root = root.resolve()
    manifest = read_json(root / "manifest.json")
    # Only the root manifest is excluded; nested manifests are ordinary artifacts.
    files = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p != root / "manifest.json"
    }
    if set(manifest) != files:
        raise ValueError("manifest does not exactly cover all input artifacts")
    for relative, digest in manifest.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or sha256(path) != digest:
            raise ValueError(f"artifact SHA256 mismatch: {relative}")
    return {**manifest, "manifest.json": sha256(root / "manifest.json")}


def audit_snapshot(root, sources):
    with tarfile.open(root / "runtime_source.tar.gz") as archive:
        members = archive.getmembers()
        if len(members) != len(sources) or {m.name for m in members} != set(sources):
            raise ValueError("source snapshot inventory mismatch")
        for member in members:
            if (
                not member.isfile()
                or hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                != sources[member.name]
            ):
                raise ValueError(f"source snapshot SHA256 mismatch: {member.name}")


def recompute_episode(rows, case, criteria, variant):
    if not rows:
        raise ValueError("empty episode")
    count = len(rows)
    close(column(rows, "step"), np.arange(1, count + 1), "step sequence")
    close(column(rows, "time_s"), DT * np.arange(1, count + 1), "sample time")
    velocity = column(rows, "velocity_error_mps")
    height = column(rows, "clearance_error_m")
    close(
        velocity,
        column(rows, "forward_velocity_mps") - column(rows, "command_velocity_mps"),
        "velocity error",
    )
    close(
        height, column(rows, "clearance_m") - column(rows, "command_clearance_m"), "clearance error"
    )
    terminal, truncated = column(rows, "terminated"), column(rows, "truncated")
    if (
        not np.isin(terminal, (0, 1)).all()
        or not np.isin(truncated, (0, 1)).all()
        or np.any(terminal[:-1])
        or np.any(truncated[:-1])
    ):
        raise ValueError("episode contains invalid flags or samples after termination")
    enabled = int(variant != "zero_residual")
    close(column(rows, "policy_enabled"), np.full(count, enabled), "policy enabled")
    close(column(rows, "policy_gated"), np.zeros(count), "ungated policy")
    for axis, action in (("x", "longitudinal"), ("z", "vertical")):
        executed = column(rows, f"action_{action}")
        if not enabled:
            close(executed, np.zeros(count), "zero residual")
        else:
            mean, std = column(rows, f"raw_mean_{axis}"), column(rows, f"std_{axis}")
            active = not (variant == "fx_only" and axis == "z")
            if active and np.any(std <= 0):
                raise ValueError("active Gaussian standard deviation must be positive")
            if not active:
                close(mean, np.zeros(count), "absent Fz mean")
                close(std, np.zeros(count), "absent Fz standard deviation")
            if variant == "bounded_mean" and np.any(np.abs(mean) > 1):
                raise ValueError("bounded mean outside [-1, 1]")
            close(executed, np.clip(mean, -1, 1), "deterministic action clipping")
    if enabled:
        close(
            column(rows, "vertical_channel_active"),
            np.full(count, int(variant != "fx_only")),
            "vertical action dimension",
        )
    initial = column(rows, "initial_position_x_m")
    close(initial, np.full(count, initial[0]), "initial position")
    displacement = math.fsum(column(rows, "command_velocity_mps")) * DT
    progress = float(rows[-1]["position_x_m"]) - initial[0]
    fraction = float(progress / displacement) if abs(displacement) > 1e-12 else None
    rms = lambda x: float(np.sqrt(np.mean(x * x)))
    tail_count = round(criteria["tail_window_seconds"] / DT)
    tail = velocity[-tail_count:]
    result = {
        "control_steps": count,
        "simulated_duration_s": count * DT,
        "completed": int(truncated[-1] == 1 and terminal[-1] == 0),
        "terminated": int(terminal[-1]),
        "termination_reason": rows[-1]["termination_reason"],
        "velocity_rmse_mps": rms(velocity),
        "clearance_rmse_m": rms(height),
        "max_abs_yaw_rad": float(np.max(np.abs(column(rows, "yaw_rad")))),
        "final_position_x_m": float(rows[-1]["position_x_m"]),
        "progress_m": float(progress),
        "commanded_displacement_m": displacement,
        "progress_fraction": fraction,
        "mean_torque_saturation_fraction": float(column(rows, "torque_saturation_fraction").mean()),
        "policy_enabled_fraction": float(enabled),
        "policy_gated_fraction": 0.0,
        "episode_return": math.fsum(column(rows, "reward")),
        "tail_velocity_rmse_mps": rms(tail),
        "tail_velocity_mean_mps": float(column(rows, "forward_velocity_mps")[-tail_count:].mean()),
        "tail_command_velocity_mean_mps": float(
            column(rows, "command_velocity_mps")[-tail_count:].mean()
        ),
        "tail_observed_seconds": len(tail) * DT,
    }
    checks = {
        "incomplete_episode": bool(result["completed"]) and count == round(4.0 / DT),
        "progress": fraction is not None
        and criteria["progress_fraction_min"] <= fraction <= criteria["progress_fraction_max"],
        "tail_velocity": rms(tail)
        <= max(
            criteria["tail_velocity_rmse_absolute_mps"],
            criteria["tail_velocity_rmse_target_fraction"] * case["velocity_mps"],
        ),
        "clearance": rms(height) <= criteria["clearance_rmse_max_m"],
        "yaw": result["max_abs_yaw_rad"] <= criteria["max_abs_yaw_rad"],
        "torque_saturation": result["mean_torque_saturation_fraction"]
        <= criteria["mean_torque_saturation_fraction_max"],
    }
    result["quality_success"] = int(all(checks.values()))
    result["quality_failure_reasons"] = "|".join(
        key for key, passed in checks.items() if not passed
    )
    return result


def compare_metrics(actual, stored):
    largest = 0.0
    for key, value in actual.items():
        expected = stored[key]
        if value is None or isinstance(value, str):
            if ("" if value is None else value) != expected:
                raise ValueError(f"metric {key} mismatch")
        else:
            largest = max(largest, close(value, float(expected), f"metric {key}"))
    return largest


def audit_evaluation(directory, cases, variant, seed, budget, criteria):
    stored = read_csv(directory / "metrics.csv")
    indexed = {row["case_id"]: row for row in stored}
    if len(stored) != len(cases) or set(indexed) != {case["case_id"] for case in cases}:
        raise ValueError("evaluation case inventory mismatch")
    results, largest = [], 0.0
    for case in cases:
        record = indexed[case["case_id"]]
        if record["variant"] != variant:
            raise ValueError("evaluation variant mismatch")
        actual = recompute_episode(
            read_csv(directory / f"{case['case_id']}.csv"), case, criteria, variant
        )
        largest = max(largest, compare_metrics(actual, record))
        results.append(
            {
                "variant": variant,
                "training_seed": seed,
                "budget": budget,
                "case_id": case["case_id"],
                "terrain_kind": case["terrain"]["kind"],
                **actual,
            }
        )
    return results, largest


def action_statistics(data, variant, budget, envs=4):
    dimension = 1 if variant == "fx_only" else 2
    if set(data) != {"timesteps", "raw_mean", "std", "raw_action", "executed_action", "log_prob"}:
        raise ValueError("unexpected action sample fields")
    for key, value in data.items():
        expected = (budget,) if key in {"timesteps", "log_prob"} else (budget, dimension)
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError(f"invalid shape or nonfinite samples: {key}")
    close(
        data["timesteps"],
        np.repeat(np.arange(envs, budget + 1, envs), envs),
        "worker sample ordering",
        0,
    )
    raw, executed, mean, std = (
        data[key] for key in ("raw_action", "executed_action", "raw_mean", "std")
    )
    close(executed, np.clip(raw, -1, 1), "sampled action clipping", 0)
    if np.any(std <= 0):
        raise ValueError("Gaussian standard deviation must be positive")
    if variant == "bounded_mean" and np.any(np.abs(mean) > 1):
        raise ValueError("bounded mean outside [-1, 1]")
    raw64, mean64, std64 = raw.astype(float), mean.astype(float), std.astype(float)
    log_prob = np.sum(
        -0.5 * ((raw64 - mean64) / std64) ** 2 - np.log(std64) - 0.5 * math.log(2 * math.pi), axis=1
    )
    lp_error = close(log_prob, data["log_prob"], "raw Normal log probability", 1e-5)
    axes = []
    for axis in range(dimension):
        axes.append(
            {
                "axis": "x" if axis == 0 else "z",
                "samples": budget,
                "sample_clip_fraction": float(np.mean(raw[:, axis] != executed[:, axis])),
                "sample_lower_fraction": float(np.mean(raw[:, axis] < -1)),
                "sample_upper_fraction": float(np.mean(raw[:, axis] > 1)),
                "deterministic_lower_fraction": float(np.mean(mean[:, axis] <= -1)),
                "deterministic_upper_fraction": float(np.mean(mean[:, axis] >= 1)),
                "mu_mean": float(mean[:, axis].mean()),
                "mu_abs_max": float(np.abs(mean[:, axis]).max()),
                "std_mean": float(std[:, axis].mean()),
                "std_min": float(std[:, axis].min()),
                "std_max": float(std[:, axis].max()),
                "executed_action_mean": float(executed[:, axis].mean()),
                "raw_log_prob_max_error": lp_error,
            }
        )
    return axes


def audit_actions(directory, variant, seed, budget, envs=4):
    with np.load(directory / "training_action_samples.npz", allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    axes = action_statistics(data, variant, budget, envs)
    stored = read_csv(directory / "training_action_diagnostics.csv")
    rollout_size = 128 * envs
    if len(stored) != budget // rollout_size:
        raise ValueError("rollout diagnostic count mismatch")
    largest = 0.0
    for index, row in enumerate(stored):
        start, end = index * rollout_size, (index + 1) * rollout_size
        if int(row["timesteps"]) != end or int(row["samples"]) != rollout_size:
            raise ValueError("rollout diagnostic sample boundaries mismatch")
        if float(row["max_unchanged_log_prob_error"]) > 1e-5:
            raise ValueError("unchanged policy log probability sentinel failed during training")
        for axis in range(data["raw_mean"].shape[1]):
            mu, std, raw, executed = (
                data[key][start:end, axis]
                for key in ("raw_mean", "std", "raw_action", "executed_action")
            )
            recomputed = {
                "mu_mean": float(mu.mean()),
                "mu_abs_max": float(np.abs(mu).max()),
                "std_mean": float(std.mean()),
                "sample_clip_fraction": float(np.mean(raw != executed)),
                "deterministic_upper_fraction": float(np.mean(mu >= 1)),
                "executed_action_mean": float(executed.mean()),
            }
            for key, value in recomputed.items():
                largest = max(
                    largest, close(value, float(row[f"{key}_{axis}"]), f"rollout {key}", 1e-6)
                )
    return [
        {"variant": variant, "training_seed": seed, **row, "rollout_summary_max_error": largest}
        for row in axes
    ]


def aggregate_cases(rows, scope):
    groups = {}
    for row in rows:
        groups.setdefault((row["variant"], row["training_seed"], row["budget"]), []).append(row)
    result = []
    for (variant, seed, budget), cases in groups.items():
        if len({case["case_id"] for case in cases}) != len(cases):
            raise ValueError("duplicate case in controller aggregate")
        result.append(
            {
                "variant": variant,
                "training_seed": seed,
                "budget": budget,
                "scope": scope,
                "cases": len(cases),
                "completed": sum(case["completed"] for case in cases),
                "quality_successes": sum(case["quality_success"] for case in cases),
                **{
                    f"mean_{key}": float(np.mean([case[key] for case in cases])) for key in MEASURES
                },
            }
        )
    return result


def development_curves(rows, fixed_case_ids):
    if len(fixed_case_ids) != 6 or len(set(fixed_case_ids)) != 6:
        raise ValueError("fixed development comparison requires six distinct cases")
    fixed = aggregate_cases(
        [row for row in rows if row["case_id"] in fixed_case_ids], "fixed_six_development_cases"
    )
    if any(row["cases"] != 6 for row in fixed):
        raise ValueError("a checkpoint is missing a fixed development case")
    full = aggregate_cases(
        [row for row in rows if row["budget"] == BUDGETS[-1]], "all_24_final_development_cases"
    )
    if any(row["cases"] != 24 for row in full):
        raise ValueError("final development aggregate requires all 24 cases")
    return fixed, full


def paired_comparisons(rows):
    groups = {}
    for row in rows:
        identity = (row["variant"], row["training_seed"])
        group = groups.setdefault(identity, {})
        if row["case_id"] in group:
            raise ValueError("duplicate holdout case")
        group[row["case_id"]] = row
    expected = {(v, s) for v in VARIANTS for s in SEEDS} | {("zero_residual", None)}
    if set(groups) != expected:
        raise ValueError("holdout needs all nine models and one unseeded baseline")
    baseline = groups[("zero_residual", None)]
    if len(baseline) != 24 or any(set(group) != set(baseline) for group in groups.values()):
        raise ValueError("holdout controllers must share the same 24 cases")
    pairs = []
    for seed in SEEDS:
        for variant in VARIANTS:
            references = [("zero_residual", None)]
            if variant != "full_gaussian":
                references.append(("full_gaussian", seed))
            for reference, reference_seed in references:
                candidate, control = groups[(variant, seed)], groups[(reference, reference_seed)]
                pairs.append(
                    {
                        "variant": variant,
                        "training_seed": seed,
                        "reference": reference,
                        "reference_training_seed": reference_seed,
                        "cases": 24,
                        "delta_quality_successes": sum(
                            row["quality_success"] for row in candidate.values()
                        )
                        - sum(row["quality_success"] for row in control.values()),
                        **{
                            f"delta_mean_{key}": float(
                                np.mean(
                                    [candidate[case][key] - control[case][key] for case in baseline]
                                )
                            )
                            for key in MEASURES
                        },
                    }
                )
    return pairs


def audit_holdout_summary(rows, summary):
    """Check JSON aggregates as well as the raw and per-controller CSVs."""
    controllers = summary["controllers"]
    expected = aggregate_cases(rows, "holdout")
    indexed = {(row["variant"], row["training_seed"]): row for row in controllers}
    if len(controllers) != len(expected) or len(indexed) != len(expected):
        raise ValueError("holdout summary controller inventory mismatch")
    largest = 0.0
    for row in expected:
        stored = indexed[(row["variant"], row["training_seed"])]
        for key, value in row.items():
            if key not in {"variant", "training_seed", "budget", "scope"}:
                largest = max(largest, close(value, stored[key], f"holdout summary {key}"))
    pairs = [row for row in paired_comparisons(rows) if row["reference"] == "full_gaussian"]
    stored_pairs = summary["paired_seed_differences"]
    indexed_pairs = {(row["variant"], row["training_seed"]): row for row in stored_pairs}
    if len(stored_pairs) != len(pairs) or len(indexed_pairs) != len(pairs):
        raise ValueError("holdout summary paired-seed inventory mismatch")
    for row in pairs:
        stored = indexed_pairs[(row["variant"], row["training_seed"])]
        for key, value in row.items():
            if key.startswith("delta_"):
                largest = max(largest, close(value, stored[key], f"holdout paired summary {key}"))
    return largest


def audit_checkpoint(path, seed, budget):
    with zipfile.ZipFile(path) as archive:
        data = json.loads(archive.read("data"))
    expected = {"seed": seed, "num_timesteps": budget, "_n_updates": budget // 512 * 4, "n_envs": 4}
    expected.update(
        {
            key: value
            for key, value in PPO_SETTINGS.items()
            if key not in {"clip_range", "policy_kwargs"}
        }
    )
    for key, value in expected.items():
        if data[key] != value:
            raise ValueError(f"serialized PPO {key} mismatch")
    return {
        "model_sha256": sha256(path),
        "actual_timesteps": data["num_timesteps"],
        "ppo_epochs_completed": data["_n_updates"],
    }


def audit_training(roots):
    samples, development, checkpoints, inventories = [], [], [], []
    identities, sources, cases, criteria = set(), None, None, None
    metric_error = 0.0
    for root in roots:
        inventory = audit_manifest(root)
        protocol, summary = read_json(root / "protocol.json"), read_json(root / "summary.json")
        if (
            protocol["budgets"] != list(BUDGETS)
            or protocol["ppo"] != PPO_SETTINGS
            or protocol["variants"] != list(VARIANTS)
            or protocol["envs"] != 4
        ):
            raise ValueError("different action training protocol")
        if (
            protocol["intermediate_development_indices"] != list(FIXED_INDICES)
            or len(protocol["development_cases"]) != 24
        ):
            raise ValueError("different development case protocol")
        if (
            protocol["training_mode"] != "mixed"
            or protocol["training_reward_scale"] != 0.01
            or protocol["episode_seconds"] != 4.0
        ):
            raise ValueError("different training mode, reward scale or duration")
        if (
            summary["source_unchanged_during_run"] is not True
            or summary["training_runs"] != len(protocol["seeds"]) * 3
            or len(summary["models"]) != summary["training_runs"]
        ):
            raise ValueError("incomplete training summary")
        if sources is None:
            sources, cases, criteria = (
                protocol["source_sha256"],
                protocol["development_cases"],
                protocol["quality_criteria"],
            )
        elif (sources, cases, criteria) != (
            protocol["source_sha256"],
            protocol["development_cases"],
            protocol["quality_criteria"],
        ):
            raise ValueError("source or development cases differ across training seeds")
        audit_snapshot(root, sources)
        inventories.append({"root": str(root.resolve()), "sha256": inventory})
        for record in summary["models"]:
            variant, seed = record["variant"], record["training_seed"]
            identity = (variant, seed)
            if (
                identity in identities
                or variant not in VARIANTS
                or seed not in SEEDS
                or seed not in protocol["seeds"]
            ):
                raise ValueError("duplicate or unregistered training identity")
            identities.add(identity)
            directory = root / f"{variant}_seed_{seed}"
            metadata = read_json(directory / "metadata.json")
            if (
                metadata != record
                or metadata["source_sha256"] != sources
                or metadata["ppo"] != PPO_SETTINGS
            ):
                raise ValueError("model metadata differs from training protocol/summary")
            final = audit_checkpoint(directory / "model.zip", seed, BUDGETS[-1])
            if any(metadata[key] != value for key, value in final.items()):
                raise ValueError("final model metadata mismatch")
            if (
                sum(metadata["exposure"]["stage_steps"].values()) != BUDGETS[-1]
                or sum(metadata["exposure"]["terrain_steps"].values()) != BUDGETS[-1]
            ):
                raise ValueError("exposure does not account for every training transition")
            if metadata["episode_starts_sha256"] != sha256(directory / "episode_starts.json"):
                raise ValueError("episode starts SHA256 mismatch")
            samples.extend(audit_actions(directory, variant, seed, BUDGETS[-1]))
            curve = read_csv(directory / "development_curve.csv")
            curve_index = {(int(row["timesteps"]), row["case_id"]): row for row in curve}
            if len(curve) != 36 or len(curve_index) != 36:
                raise ValueError("development curve must contain 6 + 6 + 24 unique rows")
            for budget in BUDGETS:
                checkpoint = audit_checkpoint(directory / f"checkpoint_{budget}.zip", seed, budget)
                checkpoints.append(
                    {"variant": variant, "training_seed": seed, "budget": budget, **checkpoint}
                )
                selected = (
                    cases if budget == BUDGETS[-1] else [cases[index] for index in FIXED_INDICES]
                )
                rows, error = audit_evaluation(
                    directory / f"development_{budget}", selected, variant, seed, budget, criteria
                )
                metric_error = max(metric_error, error)
                for row in rows:
                    stored = curve_index[(budget, row["case_id"])]
                    if stored["variant"] != variant or int(stored["training_seed"]) != seed:
                        raise ValueError("development curve identity mismatch")
                    metric_error = max(
                        metric_error,
                        compare_metrics(
                            {
                                k: v
                                for k, v in row.items()
                                if k
                                not in {
                                    "variant",
                                    "training_seed",
                                    "budget",
                                    "case_id",
                                    "terrain_kind",
                                }
                            },
                            stored,
                        ),
                    )
                development.extend(rows)
    if identities != {(v, s) for v in VARIANTS for s in SEEDS}:
        raise ValueError("all nine completed action candidates are required")
    return {
        "samples": samples,
        "development": development,
        "checkpoints": checkpoints,
        "inventories": inventories,
        "metric_max_error": metric_error,
        "sources": sources,
        "cases": cases,
        "criteria": criteria,
    }


def audit_holdout(root, training):
    inventory = audit_manifest(root)
    protocol, summary = read_json(root / "protocol.json"), read_json(root / "summary.json")
    if summary["source_unchanged_during_evaluation"] is not True or summary["episodes"] != 240:
        raise ValueError("incomplete or source-changing holdout")
    if protocol["quality_criteria"] != training["criteria"]:
        raise ValueError("holdout quality thresholds differ from training")
    for item in training["inventories"]:
        if (
            read_json(Path(item["root"]) / "protocol.json")["new_holdout_cases"]
            != protocol["cases"]
        ):
            raise ValueError("holdout cases differ from pre-training declaration")
    for path, digest in training["sources"].items():
        if protocol["source_sha256"].get(path) != digest:
            raise ValueError("holdout controller sources differ from training")
    audit_snapshot(root, protocol["source_sha256"])
    declared = {item["directory"]: item for item in protocol["inputs"]}
    if len(declared) != 9 or len(protocol["inputs"]) != 9:
        raise ValueError("holdout input model inventory mismatch")
    for item in training["inventories"]:
        train_root = Path(item["root"])
        for record in read_json(train_root / "summary.json")["models"]:
            directory = train_root / f"{record['variant']}_seed_{record['training_seed']}"
            actual = declared[str(directory)]
            if actual["model_sha256"] != record["model_sha256"] or actual[
                "metadata_sha256"
            ] != sha256(directory / "metadata.json"):
                raise ValueError("holdout checkpoint/metadata SHA256 mismatch")
    rows, largest = [], 0.0
    cohort = [("zero_residual", None)] + [(v, s) for s in SEEDS for v in VARIANTS]
    for variant, seed in cohort:
        directory = root / (variant if seed is None else f"{variant}_seed_{seed}")
        actual, error = audit_evaluation(
            directory, protocol["cases"], variant, seed, BUDGETS[-1], protocol["quality_criteria"]
        )
        rows.extend(actual)
        largest = max(largest, error)
    stored = read_csv(root / "metrics.csv")
    indexed = {
        (r["variant"], int(r["training_seed"]) if r["training_seed"] else None, r["case_id"]): r
        for r in stored
    }
    if len(stored) != 240 or len(indexed) != 240:
        raise ValueError("holdout aggregate CSV has duplicate/missing cases")
    for row in rows:
        actual = {
            k: v
            for k, v in row.items()
            if k not in {"variant", "training_seed", "budget", "case_id", "terrain_kind"}
        }
        largest = max(
            largest,
            compare_metrics(
                actual, indexed[(row["variant"], row["training_seed"], row["case_id"])]
            ),
        )
    largest = max(largest, audit_holdout_summary(rows, summary))
    return rows, {"root": str(root.resolve()), "sha256": inventory}, largest


def plot_results(output, fixed, actions, pairs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"full_gaussian": "#3c638e", "bounded_mean": "#cf7b3c", "fx_only": "#2b8065"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for seed, ax in zip(SEEDS, axes, strict=True):
        for variant in VARIANTS:
            rows = sorted(
                (r for r in fixed if r["training_seed"] == seed and r["variant"] == variant),
                key=lambda r: r["budget"],
            )
            ax.plot(
                [r["budget"] for r in rows],
                [r["mean_tail_velocity_rmse_mps"] for r in rows],
                marker="o",
                color=colors[variant],
                label=variant,
            )
        ax.set_title(f"Training seed {seed}")
        ax.set_xlabel("Training transitions")
        ax.set_ylabel("Tail velocity RMSE (m/s)")
        ax.grid(alpha=0.2)
    axes[-1].legend(fontsize=8)
    fig.suptitle("The same six development cases at every checkpoint")
    fig.savefig(output / "development_fixed_six.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for axis_name, ax in zip(("x", "z"), axes, strict=True):
        for offset, variant in enumerate(VARIANTS):
            rows = [r for r in actions if r["variant"] == variant and r["axis"] == axis_name]
            if rows:
                ax.scatter(
                    [offset] * len(rows),
                    [r["sample_clip_fraction"] for r in rows],
                    s=40,
                    color=colors[variant],
                    alpha=0.8,
                )
        ax.set_xticks(range(3), VARIANTS, rotation=12)
        ax.set_xlim(-0.35, 2.45)
        ax.set_ylim(0, 1)
        ax.set_title(f"{axis_name} action; one point per training seed")
        ax.set_ylabel("Actual sampled-action clip fraction")
        ax.grid(axis="y", alpha=0.2)
    axes[1].text(2, 0.04, "No z action", ha="center", fontsize=9)
    fig.savefig(output / "sampled_action_clipping.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for ax, key, label, scale in zip(
        axes,
        ("delta_mean_tail_velocity_rmse_mps", "delta_mean_clearance_rmse_m"),
        ("Tail velocity RMSE difference (m/s)", "Height RMSE difference (mm)"),
        (1, 1000),
        strict=True,
    ):
        for index, variant in enumerate(("bounded_mean", "fx_only")):
            rows = sorted(
                (r for r in pairs if r["variant"] == variant and r["reference"] == "full_gaussian"),
                key=lambda r: r["training_seed"],
            )
            ax.scatter(
                [index] * len(rows), [r[key] * scale for r in rows], color=colors[variant], s=45
            )
            for row in rows:
                ax.annotate(
                    str(row["training_seed"]),
                    (index, row[key] * scale),
                    xytext=(5, 2),
                    textcoords="offset points",
                    fontsize=8,
                )
        ax.axhline(0, color="0.5", linewidth=1)
        ax.set_xticks((0, 1), ("bounded_mean", "fx_only"))
        ax.set_ylabel(label)
        ax.set_xlim(-0.35, 1.65)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("New holdout: candidate minus same-seed full Gaussian")
    fig.savefig(output / "holdout_paired_seed_differences.png", dpi=170)
    plt.close(fig)


def run(training_roots, holdout, output, plots=True):
    script = Path(__file__).resolve()
    analysis_digest = sha256(script)
    if output.exists():
        raise ValueError("analysis output must not exist")
    inputs = [root.resolve() for root in training_roots] + [holdout.resolve()]
    if any(
        output.resolve().is_relative_to(root) or root.is_relative_to(output.resolve())
        for root in inputs
    ):
        raise ValueError("analysis output must be separate from all input directories")
    training = audit_training(training_roots)
    holdout_rows, holdout_inventory, holdout_error = audit_holdout(holdout, training)
    fixed, final = development_curves(
        training["development"], [training["cases"][index]["case_id"] for index in FIXED_INDICES]
    )
    pairs = paired_comparisons(holdout_rows)
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(output / "analysis_source.tar.gz", "w:gz") as archive:
        archive.add(script, arcname=str(script.relative_to(ROOT)), recursive=False)
    inventories = training["inventories"] + [holdout_inventory]
    write_json(
        output / "input_inventory.json",
        {
            "inputs": inventories,
            "analysis_source_sha256": {str(script.relative_to(ROOT)): analysis_digest},
            "analysis_dependencies": {
                "python": platform.python_version(),
                "numpy": version("numpy"),
                "matplotlib": version("matplotlib") if plots else None,
            },
        },
    )
    for filename, rows in (
        ("training_action_statistics.csv", training["samples"]),
        ("checkpoint_audit.csv", training["checkpoints"]),
        ("development_episode_metrics.csv", training["development"]),
        ("development_fixed_six.csv", fixed),
        ("development_final_24.csv", final),
        ("holdout_episode_metrics.csv", holdout_rows),
        ("holdout_controller_means.csv", aggregate_cases(holdout_rows, "all_24_new_holdout_cases")),
        ("holdout_paired_differences.csv", pairs),
    ):
        write_csv(output / filename, rows)
    if plots:
        plot_results(output, fixed, training["samples"], pairs)
    unchanged = all(audit_manifest(Path(item["root"])) == item["sha256"] for item in inventories)
    source_unchanged = sha256(script) == analysis_digest
    report = {
        "training_models": 9,
        "transitions_per_model": BUDGETS[-1],
        "total_training_transitions": 9 * BUDGETS[-1],
        "ppo_epochs_per_model": BUDGETS[-1] // 512 * 4,
        "training_seeds": list(SEEDS),
        "independent_training_replicates_per_variant": 3,
        "baseline_episodes": 24,
        "holdout_episodes": len(holdout_rows),
        "development_episodes": len(training["development"]),
        "training_artifact_files_checked": sum(
            len(item["sha256"]) for item in training["inventories"]
        ),
        "holdout_artifact_files_checked": len(holdout_inventory["sha256"]),
        "development_metric_max_abs_error": training["metric_max_error"],
        "holdout_metric_max_abs_error": holdout_error,
        "raw_normal_log_prob_max_abs_error": max(
            row["raw_log_prob_max_error"] for row in training["samples"]
        ),
        "input_artifacts_unchanged": unchanged,
        "analysis_source_unchanged": source_unchanged,
        "limits": [
            "Training seeds, not terrain cases or transitions, are the independent training replicates.",
            "No significance claim or small-n generalization interval is computed.",
            "Wall-clock evaluation times cannot be reconstructed from state CSVs and are not audited.",
            "Source/metadata hashes establish artifact consistency, not independently measured robot fidelity.",
        ],
    }
    write_json(output / "audit.json", report)
    write_json(
        output / "manifest.json",
        {
            str(path.relative_to(output)): sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    )
    if not unchanged or not source_unchanged:
        raise ValueError("input artifacts or analysis source changed during analysis")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(run(args.training_roots, args.holdout, args.output, not args.no_plots), indent=2)
    )


if __name__ == "__main__":
    main()
