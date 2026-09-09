"""Recompute continuous-policy results from raw records without retraining."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import tarfile
import zipfile
from pathlib import Path

import numpy as np

FIELDS = (
    "velocity_error_mps",
    "lateral_velocity_mps",
    "yaw_rate_error_rps",
    "clearance_error_m",
    "lateral_path_error_m",
    "longitudinal_path_error_m",
)
STAGES = (
    "settle",
    "cruise_bumps",
    "stop_one",
    "climb_and_descend",
    "slow_exit",
    "turn_left",
    "turn_right",
    "stop_final",
)
THRESHOLDS = dict(zip(FIELDS[:4], (0.14, 0.10, 0.06, 0.025), strict=True))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, payload) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rmse(rows, field) -> float:
    values = np.asarray([float(row[field]) for row in rows])
    if not np.isfinite(values).all():
        raise ValueError(f"nonfinite telemetry: {field}")
    return float(np.sqrt(np.mean(np.square(values))))


def recompute(rows: list[dict]) -> dict:
    """Independent formulas; do not import the experiment's summary function."""
    if not rows:
        raise ValueError("empty rollout")
    completed = bool(int(rows[-1]["truncated"])) and not bool(int(rows[-1]["terminated"]))
    stages = []
    for name in STAGES:
        selected = [row for row in rows if row["stage"] == name]
        if not selected:
            stages.append({"stage": name, "quality_pass": False})
            continue
        steady = [row for row in selected if float(row["stage_time_s"]) >= 1.0] or selected
        errors = {field: rmse(steady, field) for field in THRESHOLDS}
        actual_turn = float(selected[-1]["yaw_rad"]) - float(selected[0]["yaw_rad"])
        command_turn = float(selected[-1]["reference_yaw_rad"]) - float(
            selected[0]["reference_yaw_rad"]
        )
        fraction = actual_turn / command_turn if abs(command_turn) > 0.1 else None
        quality = all(errors[field] <= limit for field, limit in THRESHOLDS.items())
        if name in ("turn_left", "turn_right"):
            quality = quality and fraction is not None and 0.25 <= fraction <= 1.75
        stages.append(
            {
                "stage": name,
                "quality_pass": quality,
                "steady_rmse": errors,
                "yaw_change_rad": actual_turn,
                "commanded_yaw_change_rad": command_turn,
                "turn_progress_fraction": fraction,
            }
        )
    coverage = {"flat", "bumps", "ramp_up", "ramp_down"}.issubset(
        {row["terrain_section"] for row in rows}
    )
    raw_return = sum(
        math.exp(-((float(row["velocity_error_mps"]) / 0.2) ** 2))
        + math.exp(-((float(row["yaw_rate_error_rps"]) / 0.15) ** 2))
        - (float(row["clearance_error_m"]) / 0.08) ** 2
        - 5 * int(row["terminated"])
        for row in rows
    )
    np.testing.assert_allclose(raw_return, sum(float(row["reward"]) for row in rows), atol=1e-8)
    return {
        "completed": completed,
        "quality_pass": completed and coverage and all(stage["quality_pass"] for stage in stages),
        "duration_s": float(rows[-1]["time_s"]),
        "required_terrain_visited": coverage,
        "stage_summaries": stages,
        "whole_task_rmse": {field: rmse(rows, field) for field in FIELDS},
        "raw_return": raw_return,
        "mean_raw_reward": raw_return / len(rows),
    }


def compare_summary(actual: dict, stored: dict) -> float:
    error = 0.0
    for key in ("completed", "quality_pass", "required_terrain_visited"):
        if actual[key] != stored[key]:
            raise ValueError(f"stored result disagrees with telemetry: {key}")
    np.testing.assert_allclose(actual["duration_s"], stored["duration_s"], atol=1e-12, rtol=0)
    for field, value in actual["whole_task_rmse"].items():
        error = max(error, abs(value - stored["whole_task_rmse"][field]))
    for actual_stage, stored_stage in zip(
        actual["stage_summaries"], stored["stage_summaries"], strict=True
    ):
        if actual_stage["stage"] != stored_stage["stage"] or (
            actual_stage["quality_pass"] != stored_stage["quality_pass"]
        ):
            raise ValueError("stored stage result disagrees with telemetry")
        for field, value in actual_stage.get("steady_rmse", {}).items():
            error = max(error, abs(value - stored_stage["steady_rmse"][field]))
    if error > 1e-12:
        raise ValueError(f"stored RMSE differs by {error:g}")
    return error


def audit_poststep_position(qpos, qvel, telemetry_xyz, physics_dt_s: float) -> float:
    """Check the existing implicitfast phase convention, not a loose tolerance.

    The frozen plant calls mj_step without a subsequent kinematics refresh.
    Its xpos truth sample is before the last 2 ms integration; qpos is after it.
    Free-joint translation is exactly reversible using the final qvel.
    """
    np.testing.assert_allclose(
        qpos[:, :3] - physics_dt_s * qvel[:, :3], telemetry_xyz, atol=1e-12, rtol=0
    )
    return float(np.max(np.abs(qpos[:, :3] - telemetry_xyz)))


def audit_rollout(directory: Path, physics_dt_s: float = 0.002) -> tuple[dict, float]:
    manifest = read_json(directory / "manifest.json")
    for name, expected in manifest["sha256"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"rollout artifact SHA mismatch: {directory / name}")
    compiled = manifest["compiled_model"]
    if compiled["path"] != f"../models/{compiled['uncompressed_sha256']}.mjb.gz":
        raise ValueError("unexpected compiled model reference")
    if sha256(directory / compiled["path"]) != compiled["sha256"]:
        raise ValueError("compiled model compressed hash mismatch")
    protocol = read_json(directory / "protocol.json")
    stored = read_json(directory / "summary.json")
    with (directory / "telemetry.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected_steps = np.arange(1, len(rows) + 1)
    np.testing.assert_array_equal([int(row["step"]) for row in rows], expected_steps)
    np.testing.assert_allclose(
        [float(row["time_s"]) for row in rows],
        expected_steps * protocol["control_dt_s"],
        atol=1e-10,
        rtol=0,
    )
    with np.load(directory / "states.npz", allow_pickle=False) as states:
        if len(states["qpos"]) != len(rows) + 1 or len(states["qvel"]) != len(rows) + 1:
            raise ValueError("state recording must contain initial state and every control step")
        for key in ("qpos", "qvel", "time_s"):
            if not np.isfinite(states[key]).all():
                raise ValueError(f"nonfinite recorded state: {key}")
        np.testing.assert_allclose(
            states["time_s"], np.arange(len(rows) + 1) * protocol["control_dt_s"], atol=1e-10
        )
        position_phase_offset = audit_poststep_position(
            states["qpos"][1:],
            states["qvel"][1:],
            np.asarray([[float(row[key]) for key in ("x_m", "y_m", "z_m")] for row in rows]),
            physics_dt_s,
        )
    calculated = recompute(rows)
    error = compare_summary(calculated, stored)
    if protocol["mode"] == "zero" and any(
        float(row[key]) != 0 for row in rows for key in ("action_longitudinal", "action_vertical")
    ):
        raise ValueError("zero baseline has a nonzero policy action")
    if protocol["mode"] == "policy":
        metadata = protocol["checkpoint_metadata"]
        if metadata["model_sha256"] != protocol["checkpoint_sha256"]:
            raise ValueError("rollout checkpoint hash differs from metadata")
        for key in ("control_schema", "observation_schema", "action_schema"):
            if metadata[key] != protocol[key]:
                raise ValueError(f"rollout/checkpoint schema mismatch: {key}")
    return {
        "rollout": directory.name,
        "mode": protocol["mode"],
        "case": f"seed{protocol['seed']}_{protocol['config']['duration_s']:g}s",
        "training_seed": (protocol["checkpoint_metadata"] or {}).get("training_seed", 0),
        "budget": (protocol["checkpoint_metadata"] or {}).get("num_timesteps", 0),
        "split": "holdout"
        if directory.name.startswith("holdout")
        else "development"
        if protocol["mode"] == "policy"
        else "baseline",
        "completed": calculated["completed"],
        "quality_pass": calculated["quality_pass"],
        "duration_s": calculated["duration_s"],
        "raw_return": calculated["raw_return"],
        "mean_raw_reward": calculated["mean_raw_reward"],
        "max_truth_vs_poststep_position_offset_m": position_phase_offset,
        "failed_stages": ";".join(
            row["stage"] for row in calculated["stage_summaries"] if not row["quality_pass"]
        ),
        **calculated["whole_task_rmse"],
    }, error


def verify_source(root: Path) -> int:
    source = read_json(root / "source.json")
    if sha256(root / "source.tar.gz") != source["archive_sha256"]:
        raise ValueError("source archive hash mismatch")
    with tarfile.open(root / "source.tar.gz", "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        if set(members) != set(source["sha256"]):
            raise ValueError("source archive membership differs from hash inventory")
        for name, member in members.items():
            if (
                hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                != source["sha256"][name]
            ):
                raise ValueError(f"archived source mismatch: {name}")
    return len(source["sha256"])


def audit_training(
    root: Path, seeds: list[int], budgets: list[int]
) -> tuple[list[dict], list[dict]]:
    actions, episodes = [], []
    for seed in seeds:
        directory = root / f"training_seed{seed}"
        stored = read_json(directory / "action_exposure_summary.json")
        with np.load(directory / "training_samples.npz", allow_pickle=False) as data:
            raw, clipped = data["raw_gaussian"], data["clipped_action"]
            if raw.shape != (budgets[-1] // 4, 4, 2):
                raise ValueError("training sample count differs from fixed budget")
            np.testing.assert_array_equal(clipped, np.clip(raw, -1, 1))
            np.testing.assert_allclose(data["scaled_reward"], 0.01 * data["raw_reward"], atol=1e-6)
            fractions = (np.abs(raw) > 1).mean(axis=(0, 1))
            np.testing.assert_array_equal(fractions, stored["clipped_fraction_per_action"])
            if sum(stored["stage_steps"].values()) != budgets[-1]:
                raise ValueError("stage exposure does not account for every physical step")
            if sum(stored["terrain_steps"].values()) != budgets[-1]:
                raise ValueError("terrain exposure does not account for every physical step")
            actions.append(
                {
                    "training_seed": seed,
                    "samples": budgets[-1],
                    "fx_sample_clip_fraction": float(fractions[0]),
                    "fz_sample_clip_fraction": float(fractions[1]),
                    "sample_mean_fx": float(raw[..., 0].mean()),
                    "sample_mean_fz": float(raw[..., 1].mean()),
                }
            )
        for budget in budgets:
            checkpoint = root / "checkpoints" / f"seed{seed}" / f"step{budget}"
            metadata = read_json(checkpoint / "metadata.json")
            if metadata["model_sha256"] != sha256(checkpoint / "model.zip"):
                raise ValueError("checkpoint SHA mismatch")
            if metadata["num_timesteps"] != budget or metadata["requested_budget"] != budget:
                raise ValueError("checkpoint is not the requested fixed endpoint")
            if metadata["protocol_sha256"] != sha256(root / "protocol.json"):
                raise ValueError("checkpoint protocol mismatch")
            if metadata["source_sha256"] != read_json(root / "source.json")["sha256"]:
                raise ValueError("checkpoint source differs from experiment snapshot")
            with zipfile.ZipFile(checkpoint / "model.zip") as archive:
                saved_model = json.loads(archive.read("data"))
            if saved_model["num_timesteps"] != budget or saved_model["seed"] != seed:
                raise ValueError("serialized PPO counters differ from metadata")
            if saved_model["_n_updates"] != budget // (4 * 128) * 4:
                raise ValueError("serialized PPO update count differs from fixed protocol")
        progress = read_json(directory / "rollout_progress.json")
        if len(progress) != budgets[-1] // (4 * 128):
            raise ValueError("rollout progress count differs from budget")
        if [row["timesteps"] for row in progress] != list(range(512, budgets[-1] + 1, 512)):
            raise ValueError("rollout progress skips or repeats physical samples")
        for worker in range(4):
            records = [
                json.loads(line)
                for line in (directory / f"episode_seeds_{worker}.jsonl").read_text().splitlines()
            ]
            stream = np.random.default_rng(np.random.SeedSequence([seed + worker, worker]))
            expected = stream.integers(1_000_000, 2**31 - 1, size=len(records)).tolist()
            if [record["sensor_seed"] for record in records] != expected:
                raise ValueError("episode noise seeds differ from the declared reproducible stream")
            with (directory / f"worker_{worker}.monitor.csv").open(newline="") as file:
                next(file)  # SB3's JSON comment, not a data row.
                length = 0
                for index, row in enumerate(csv.DictReader(file)):
                    length += int(row["l"])
                    episodes.append(
                        {
                            "training_seed": seed,
                            "worker": worker,
                            "episode": index,
                            "global_physical_steps_at_end": 4 * length,
                            "episode_steps": int(row["l"]),
                            "raw_return": float(row["r"]),
                            "mean_raw_reward": float(row["r"]) / int(row["l"]),
                        }
                    )
    return actions, episodes


def paired_rows(metrics: list[dict]) -> list[dict]:
    baselines = {row["case"]: row for row in metrics if row["mode"] == "zero"}
    pairs = []
    for row in metrics:
        if row["mode"] != "policy":
            continue
        zero = baselines[row["case"]]
        pairs.append(
            {
                "training_seed": row["training_seed"],
                "budget": row["budget"],
                "split": row["split"],
                "case": row["case"],
                "policy_completed": row["completed"],
                "baseline_completed": zero["completed"],
                "policy_quality_pass": row["quality_pass"],
                "baseline_quality_pass": zero["quality_pass"],
                **{f"delta_{field}": row[field] - zero[field] for field in FIELDS},
            }
        )
    return pairs


def plot_results(output: Path, metrics: list[dict], episodes: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    seeds = sorted({row["training_seed"] for row in episodes})
    for seed in seeds:
        rows = [row for row in episodes if row["training_seed"] == seed]
        axes[0].scatter(
            [row["global_physical_steps_at_end"] for row in rows],
            [row["mean_raw_reward"] for row in rows],
            s=12,
            alpha=0.6,
            label=str(seed),
        )
    axes[0].set(xlabel="Physical training samples", ylabel="Raw episode reward / step")
    axes[0].legend(title="Training seed", fontsize=8)
    holdout_cases = {row["case"] for row in metrics if row["split"] == "holdout"}
    for axis, field, scale, label in (
        (axes[1], "clearance_error_m", 1000, "Whole-task clearance RMSE (mm)"),
        (axes[2], "yaw_rate_error_rps", 1, "Whole-task yaw-rate RMSE (rad/s)"),
    ):
        for index, seed in enumerate(seeds):
            selected = [
                row for row in metrics if row["training_seed"] == seed and row["split"] == "holdout"
            ]
            values = [scale * row[field] for row in selected]
            axis.scatter(np.full(len(values), index), values, s=18, alpha=0.6)
            axis.scatter(index, np.mean(values), marker="_", color="black", s=140)
        zero = [
            scale * row[field]
            for row in metrics
            if row["mode"] == "zero" and row["case"] in holdout_cases
        ]
        axis.axhline(np.mean(zero), color="0.4", linestyle="--", label="Paired zero mean")
        axis.set(xticks=range(len(seeds)), xticklabels=seeds, xlabel="Training seed", ylabel=label)
        axis.legend(fontsize=8)
    fig.savefig(output / "training_and_holdout.png", dpi=150)
    plt.close(fig)


def run(root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    manifest = read_json(root / "manifest.json")
    for name, expected in manifest["sha256"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"experiment artifact mismatch: {name}")
    protocol = read_json(root / "protocol.json")
    if protocol["smoke_only"]:
        raise ValueError("smoke runs cannot support the full experiment analysis")
    source_count = verify_source(root)
    source_hashes = read_json(root / "source.json")["sha256"]
    metrics, max_error = [], 0.0
    for directory in sorted((root / "rollouts").iterdir()):
        if directory.name == "models":
            continue
        if read_json(directory / "manifest.json")["source_sha256"] != source_hashes:
            raise ValueError("rollout source differs from frozen experiment snapshot")
        row, error = audit_rollout(directory)
        metrics.append(row)
        max_error = max(max_error, error)
    for path in (root / "rollouts/models").glob("*.mjb.gz"):
        raw_sha = hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest()
        if path.name != raw_sha + ".mjb.gz":
            raise ValueError("compiled model content address mismatch")
    expected_runs = {
        f"zero_seed{case['seed']}_{case['duration_s']:g}s"
        for cases in protocol["evaluation_cases"].values()
        for case in cases
    }
    for seed in protocol["training_seeds"]:
        for budget in protocol["checkpoint_budgets"]:
            development = protocol["evaluation_cases"]["development"]
            cases = (
                development
                if budget == protocol["checkpoint_budgets"][-1]
                else [development[0 if budget == protocol["checkpoint_budgets"][0] else 1]]
            )
            expected_runs.update(
                f"dev_train{seed}_step{budget}_seed{case['seed']}_{case['duration_s']:g}s"
                for case in cases
            )
        expected_runs.update(
            f"holdout_train{seed}_seed{case['seed']}_{case['duration_s']:g}s"
            for case in protocol["evaluation_cases"]["holdout"]
        )
    if {row["rollout"] for row in metrics} != expected_runs:
        raise ValueError("missing or extra evaluation cases versus frozen protocol")
    actions, episodes = audit_training(
        root, protocol["training_seeds"], protocol["checkpoint_budgets"]
    )
    pairs = paired_rows(metrics)
    groups = []
    for seed in protocol["training_seeds"]:
        selected = [
            row for row in pairs if row["training_seed"] == seed and row["split"] == "holdout"
        ]
        groups.append(
            {
                "training_seed": seed,
                "cases": len(selected),
                "policy_complete": sum(row["policy_completed"] for row in selected),
                "baseline_complete": sum(row["baseline_completed"] for row in selected),
                "policy_quality": sum(row["policy_quality_pass"] for row in selected),
                "baseline_quality": sum(row["baseline_quality_pass"] for row in selected),
                **{
                    f"mean_paired_delta_{field}": float(
                        np.mean([row[f"delta_{field}"] for row in selected])
                    )
                    for field in FIELDS
                },
            }
        )
    output.mkdir(parents=True)
    write_csv(output / "metrics_recomputed.csv", metrics)
    write_csv(output / "paired_cases.csv", pairs)
    write_csv(output / "holdout_by_training_seed.csv", groups)
    write_csv(output / "action_clipping.csv", actions)
    write_csv(output / "episode_return_history.csv", episodes)
    plot_results(output, metrics, episodes)
    result = {
        "schema": "d1-continuous-policy-independent-audit-v1",
        "input_manifest_sha256": sha256(root / "manifest.json"),
        "analyzer_sha256": sha256(Path(__file__)),
        "artifact_sha_checks": len(manifest["sha256"]),
        "archived_source_sha_checks": source_count,
        "raw_rollouts_recomputed": len(metrics),
        "max_summary_rmse_discrepancy": max_error,
        "complete_case_inventory_matches_protocol": True,
        "training_samples": sum(row["samples"] for row in actions),
        "holdout_by_training_seed": groups,
        "statistical_unit": "3 training seeds; 6 paired evaluation cases per seed; no control-step pseudoreplication",
        "scope": "one fixed composite road, new measurement-noise seeds, 45/50 s command schedules; no terrain-map holdout or real-hardware claim",
        "action_check": "actual sampled Normal actions and actual elementwise clip; no theoretical clipping probability substituted",
        "state_phase_convention": {
            "physics_substep_s": 0.002,
            "truth_xyz": "xpos before the final implicitfast integration substep",
            "recorded_qpos": "qpos after the final integration substep",
            "verified_identity": "truth_xyz == recorded_qpos_xyz - 0.002 * recorded_qvel_xyz (absolute tolerance 1e-12 m)",
            "max_position_offset_m": max(
                row["max_truth_vs_poststep_position_offset_m"] for row in metrics
            ),
            "limitation": "current plant does not refresh derived kinematics after mj_step; telemetry and video are not exactly synchronous",
        },
    }
    write_json(output / "independent_audit.json", result)
    write_json(
        output / "manifest.json",
        {
            "sha256": {
                path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()
            }
        },
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.run, args.output), indent=2))
