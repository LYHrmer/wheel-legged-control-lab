"""Frozen-policy, development-only D1 residual ablations and feedback replay.

All variants retain the same LQR/VMC feedback. Only the RL residual layer is
removed, masked, replaced by a constant, or replayed from its nominal episode.
No model is trained or changed. Historical holdout CSVs are classification only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from wheel_legged_control.d1.terrain_tracking_env import D1TerrainTrackingEnv

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "terrain_feedback_metrics", ROOT / "scripts/run_d1_terrain_curriculum.py"
)
assert _SPEC is not None and _SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(RUNNER)
DEFAULT_CASES = (1, 4, 5, 10, 11, 22)
FEEDBACK_CASES = (4, 10)
PERTURBATIONS = ("pitch_minus", "pitch_plus", "push_minus", "push_plus")
REPLAY_FIELDS = (
    "position_x_m",
    "position_y_m",
    "yaw_rad",
    "forward_velocity_mps",
    "clearance_m",
    "measured_pitch_rad",
    "measured_roll_rad",
    "action_longitudinal",
    "action_vertical",
    "reward",
    "terminated",
    "truncated",
    "push_force_n",
)
REPLAY_TOLERANCE = 1e-10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def error_decomposition(values: Any) -> dict[str, Any]:
    errors = np.asarray(values, dtype=np.float64)
    if errors.ndim != 1 or not len(errors) or not np.isfinite(errors).all():
        raise ValueError("errors must be a nonempty finite vector")
    bias = float(errors.mean())
    variance = float(np.mean((errors - bias) ** 2))
    mse = float(np.mean(errors**2))
    if not math.isclose(mse, bias**2 + variance, rel_tol=1e-12, abs_tol=1e-15):
        raise AssertionError("RMSE decomposition failed")
    share = bias**2 / mse if mse > 1e-16 else 0.0
    label = "mixed"
    if mse <= 1e-16:
        label = "negligible"
    elif share >= 0.8:
        label = "bias_dominated_slow" if bias < 0 else "bias_dominated_fast"
    elif share <= 0.2:
        label = "fluctuation_dominated"
    return {
        "bias_mps": bias,
        "variance_m2ps2": variance,
        "std_mps": math.sqrt(variance),
        "rmse_mps": math.sqrt(mse),
        "bias_share_of_mse": share,
        "failure_type": label,
    }


def gaussian_probe(model: Any, observation: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Inspect the current raw Gaussian; evaluation executes its clipped mean.

    No random sample is drawn, so this instrumentation cannot perturb the policy
    RNG or the nominal replay. These are actor outputs, not PPO ratio clipping.
    """
    import torch

    with torch.no_grad():
        tensor, _ = model.policy.obs_to_tensor(observation)
        distribution = model.policy.get_distribution(tensor).distribution
        mu = distribution.mean.cpu().numpy().reshape(-1).astype(np.float64)
        sigma = distribution.stddev.cpu().numpy().reshape(-1).astype(np.float64)
    if mu.shape != (2,) or sigma.shape != (2,) or not np.isfinite(mu).all():
        raise ValueError("expected a finite two-dimensional Gaussian actor")
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("Gaussian standard deviations must be finite and positive")
    action = np.asarray(model.predict(observation, deterministic=True)[0], dtype=np.float64)
    if not np.allclose(action, np.clip(mu, -1, 1), rtol=0, atol=1e-6):
        raise ValueError("diagnostic requires an unsquashed Gaussian with clipped-mean execution")
    fields = {}
    for index, axis in enumerate(("x", "z")):
        fields[f"raw_mu_{axis}"] = float(mu[index])
        fields[f"raw_std_{axis}"] = float(sigma[index])
        fields[f"raw_upper_probability_{axis}"] = 0.5 * math.erfc(
            (1 - mu[index]) / (sigma[index] * math.sqrt(2))
        )
    return action, fields


def select_action(mode: str, action: np.ndarray | None, replay: Any = None) -> np.ndarray:
    if mode == "zero":
        return np.zeros(2)
    if mode == "constant_fz":
        return np.array([0.0, 1.0])
    if mode == "replay":
        selected = np.asarray(replay, dtype=np.float64)
    elif mode in {"full", "fx_only", "fz_only"} and action is not None:
        selected = np.asarray(action, dtype=np.float64).copy()
        if mode == "fx_only":
            selected[1] = 0.0
        elif mode == "fz_only":
            selected[0] = 0.0
    else:
        raise ValueError("unknown mode or missing policy/replay action")
    if selected.shape != (2,) or not np.isfinite(selected).all():
        raise ValueError("residual must have two finite values")
    return np.clip(selected, -1, 1)


def configure_perturbation(env: Any, case: dict, perturbation: str) -> tuple[Any, dict]:
    if perturbation not in ("nominal", *PERTURBATIONS):
        raise ValueError("unknown perturbation")
    options = {key: case[key] for key in ("terrain", "velocity_mps", "height_m")}
    options["initial_pitch"] = {"pitch_minus": -0.02, "pitch_plus": 0.02}.get(perturbation, 0.0)
    observation, info = env.reset(seed=case["environment_seed"], options=options)
    if perturbation.startswith("push_"):
        # Explicit experimental use of the existing pulse mechanism. Reset in
        # the terrain environment normally disables it; controller code stays intact.
        env._push_start = 200
        env._push_end = 210
        env._push_force_n = 20.0 if perturbation == "push_plus" else -20.0
    return observation, info


def run_episode(
    case: dict,
    mode: str,
    *,
    model: Any = None,
    perturbation: str = "nominal",
    tape: np.ndarray | None = None,
    seconds: float = 4.0,
    environment: Any = D1TerrainTrackingEnv,
) -> tuple[list[dict], dict]:
    env = environment(
        baseline="lqr", episode_seconds=seconds, training_mode="flat", randomize=False
    )
    rows: list[dict] = []
    try:
        observation, reset_info = configure_perturbation(env, case, perturbation)
        for step in range(round(seconds / 0.01)):
            probe = {
                f"raw_{field}_{axis}": None
                for axis in ("x", "z")
                for field in ("mu", "std", "upper_probability")
            }
            policy_action = None
            if model is not None:
                policy_action, probe = gaussian_probe(model, observation)
            if mode == "replay" and (tape is None or step >= len(tape)):
                raise ValueError("nominal action tape is shorter than the requested replay")
            action = select_action(mode, policy_action, tape[step] if mode == "replay" else None)
            observation, reward, terminated, truncated, info = env.step(action)
            applied = np.asarray(info["residual_action"])
            np.testing.assert_allclose(applied, action, rtol=0, atol=1e-12)
            terms = info["reward_terms"]
            if not math.isclose(
                sum(value for key, value in terms.items() if key != "total"),
                reward,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise AssertionError("raw reward terms do not sum to the environment reward")
            rows.append(
                {
                    "step": step + 1,
                    "time_s": (step + 1) * 0.01,
                    **{
                        name: info[name]
                        for name in (*RUNNER.INFO_FIELDS, *RUNNER.TRACKING_INFO_FIELDS)
                    },
                    "initial_position_x_m": reset_info["position_x_m"],
                    "action_longitudinal": float(applied[0]),
                    "action_vertical": float(applied[1]),
                    "residual_fx_n": float(applied[0] * 11.25),
                    "residual_fz_n": float(applied[1] * 20),
                    "policy_enabled": int(mode in {"full", "fx_only", "fz_only"}),
                    "policy_gated": 0,
                    "push_force_n": float(info["push_force_n"]),
                    **probe,
                    "reward": float(reward),
                    **{f"reward_{name}": float(value) for name, value in terms.items()},
                    "qpos_qvel_sha256": hashlib.sha256(
                        np.concatenate(env.plant.simulation_state()).astype("<f8").tobytes()
                    ).hexdigest(),
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                    "termination_reason": info["termination_reason"],
                }
            )
            if terminated or truncated:
                break
    finally:
        env.close()
    metrics = RUNNER.summarize_episode(
        rows,
        quality_criteria=RUNNER.TRACKING_QUALITY_CRITERIA,
        requested_duration_s=seconds,
        target_velocity_mps=case["velocity_mps"],
    )
    for label, window in (("full", rows), ("tail", rows[-100:])):
        metrics.update(
            {
                f"{label}_{key}": value
                for key, value in error_decomposition(
                    [row["velocity_error_mps"] for row in window]
                ).items()
            }
        )
    for key in rows[0]:
        if key.startswith("reward_"):
            metrics[f"return_{key[7:]}"] = float(sum(row[key] for row in rows))
    metrics.update(
        vertical_upper_fraction=float(
            np.mean([row["action_vertical"] >= 1 - 1e-7 for row in rows])
        ),
        vertical_action_mean=float(np.mean([row["action_vertical"] for row in rows])),
        vertical_action_std=float(np.std([row["action_vertical"] for row in rows])),
        longitudinal_action_mean=float(np.mean([row["action_longitudinal"] for row in rows])),
        raw_mu_z_mean=None,
        raw_std_z_mean=None,
        raw_upper_probability_z_mean=None,
    )
    if model is not None:
        metrics.update(
            {
                f"{key}_mean": float(np.mean([row[key] for row in rows]))
                for key in ("raw_mu_z", "raw_std_z", "raw_upper_probability_z")
            }
        )
    return rows, metrics


def verify_nominal_replay(nominal: list[dict], replay: list[dict]) -> float:
    if len(nominal) != len(replay):
        raise AssertionError("nominal replay episode length changed")
    difference = max(
        abs(float(left[key]) - float(right[key]))
        for left, right in zip(nominal, replay, strict=True)
        for key in REPLAY_FIELDS
    )
    if difference > REPLAY_TOLERANCE:
        raise AssertionError(f"nominal replay changed a physical transition: {difference}")
    if any(
        left["termination_reason"] != right["termination_reason"]
        for left, right in zip(nominal, replay, strict=True)
    ):
        raise AssertionError("nominal replay termination reason changed")
    if any(
        left["qpos_qvel_sha256"] != right["qpos_qvel_sha256"]
        for left, right in zip(nominal, replay, strict=True)
    ):
        raise AssertionError("nominal replay qpos/qvel bytes changed")
    return difference


def classify_history(metrics_path: Path) -> tuple[list[dict], dict[str, str]]:
    classified, hashes = [], {str(metrics_path): sha256(metrics_path)}
    for entry in read_csv(metrics_path):
        path = metrics_path.parent / entry["telemetry_csv"]
        rows = read_csv(path)
        hashes[str(path)] = sha256(path)
        classified.append(
            {
                key: entry[key]
                for key in (
                    "condition",
                    "training_seed",
                    "case_id",
                    "terrain_kind",
                    "quality_failure_reasons",
                )
            }
            | {"source_csv": str(path)}
            | {
                f"tail_{key}": value
                for key, value in error_decomposition(
                    [float(row["velocity_error_mps"]) for row in rows[-100:]]
                ).items()
            }
        )
    return classified, hashes


def load_models(checkpoint_root: Path, seeds: list[int]) -> tuple[dict, dict]:
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    models, metadata = {}, {}
    for seed in seeds:
        path = checkpoint_root / "training" / f"mixed_seed_{seed}"
        record = json.loads((path / "metadata.json").read_text())
        if sha256(path / "model.zip") != record["model_sha256"]:
            raise ValueError("checkpoint hash no longer matches frozen training metadata")
        model = PPO.load(path / "model.zip", device="cpu")
        RUNNER.validate_checkpoint(record, model, D1TerrainTrackingEnv)
        if record["condition"] != "mixed" or record["training_seed"] != seed:
            raise ValueError("checkpoint identity does not match the requested mixed seed")
        models[seed], metadata[seed] = model, record
    return models, metadata


def paired_summary(metrics: list[dict], seeds: list[int]) -> list[dict]:
    comparisons = []
    for seed in seeds:
        for comparison, left_mode, right_mode, nominal in (
            ("remove_fz", "fx_only", "full", True),
            ("live_vs_replay", "full", "replay", False),
        ):
            left = {
                (row["case_id"], row["perturbation"]): row
                for row in metrics
                if row["training_seed"] == seed
                and row["mode"] == left_mode
                and (row["perturbation"] == "nominal") == nominal
            }
            right = {
                (row["case_id"], row["perturbation"]): row
                for row in metrics
                if row["training_seed"] == seed
                and row["mode"] == right_mode
                and (row["perturbation"] == "nominal") == nominal
            }
            if not left:
                continue
            if left.keys() != right.keys():
                raise AssertionError("comparison has unmatched cases")
            delta = {
                key: float(np.mean([left[case][key] - right[case][key] for case in left]))
                for key in (
                    "tail_velocity_rmse_mps",
                    "velocity_rmse_mps",
                    "clearance_rmse_m",
                    "episode_return",
                )
            }
            quality_delta = sum(
                left[case]["quality_success"] - right[case]["quality_success"] for case in left
            )
            right_tail = float(np.mean([row["tail_velocity_rmse_mps"] for row in right.values()]))
            threshold_met = (
                delta["clearance_rmse_m"] <= -0.002 and delta["velocity_rmse_mps"] <= 0.005
                if comparison == "remove_fz"
                else delta["tail_velocity_rmse_mps"] <= -max(0.005, 0.1 * right_tail)
            ) and quality_delta >= 0
            comparisons.append(
                {
                    "comparison": comparison,
                    "training_seed": seed,
                    "paired_episodes": len(left),
                    "left_minus_right": delta,
                    "quality_pass_count_delta": quality_delta,
                    "predeclared_threshold_met": bool(threshold_met),
                }
            )
    return comparisons


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("diagnose", "reproduce"), default="diagnose")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=ROOT / "results/d1_terrain_tracking_v2"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1000, 2000])
    parser.add_argument("--case-indices", type=int, nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument(
        "--feedback-case-indices", type=int, nargs="+", default=list(FEEDBACK_CASES)
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.mode == "diagnose" and args.output is None:
        raise ValueError("diagnose requires a new --output directory")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        raise ValueError("seeds must be unique nonnegative integers")
    if not args.case_indices or len(set(args.case_indices)) != len(args.case_indices):
        raise ValueError("case indices must be nonempty and unique")
    if any(index not in range(24) for index in args.case_indices):
        raise ValueError("case indices must select the 24 frozen development cases")
    if len(set(args.feedback_case_indices)) != len(args.feedback_case_indices):
        raise ValueError("feedback case indices must be unique")
    if not set(args.feedback_case_indices) <= set(args.case_indices):
        raise ValueError("feedback cases must be a subset of the nominal cases")


def run(args: argparse.Namespace) -> dict:
    validate_args(args)
    models, metadata = load_models(args.checkpoint_root, args.seeds)
    cases = RUNNER.evaluation_cases("development", "tracking-v2")
    if args.mode == "reproduce":
        _, metrics = run_episode(cases[0], "full", model=models[args.seeds[0]], seconds=2.0)
        print(
            json.dumps(
                {
                    key: metrics[key]
                    for key in ("vertical_upper_fraction", "vertical_action_mean", "raw_mu_z_mean")
                }
            )
        )
        if metrics["vertical_upper_fraction"] >= 0.9:
            raise AssertionError("REPRODUCED: frozen PPO vertical action stays at upper boundary")
        return metrics
    sources = set().union(*(record["source_sha256"] for record in metadata.values()))
    sources.update({str(Path(__file__).relative_to(ROOT)), "scripts/run_d1_terrain_curriculum.py"})
    source_hashes = {name: sha256(ROOT / name) for name in sorted(sources)}
    selected = [cases[index] for index in args.case_indices]
    feedback_ids = {cases[index]["case_id"] for index in args.feedback_case_indices}
    protocol = {
        "schema_version": 1,
        "experiment": "frozen_d1_residual_feedback",
        "evaluation_split": "development",
        "cases": selected,
        "feedback_cases": sorted(feedback_ids),
        "training_seeds": args.seeds,
        "episode_seconds": 4.0,
        "control_dt_s": 0.01,
        "torch_threads": 1,
        "checkpoint_selection": "all requested mixed final checkpoints; no score selection",
        "model_sha256": {str(seed): record["model_sha256"] for seed, record in metadata.items()},
        "baseline": "same LQR/VMC feedback remains active in every variant",
        "nominal_modes": ["zero", "constant_fz", "full", "fx_only", "fz_only", "replay"],
        "constant_action": [0.0, 1.0],
        "residual_scale_n": [11.25, 20.0],
        "perturbations": {
            "pitch_minus": {"initial_pitch_rad": -0.02},
            "pitch_plus": {"initial_pitch_rad": 0.02},
            "push_minus": {"world_fx_n": -20, "start_s": 2.0, "duration_s": 0.1},
            "push_plus": {"world_fx_n": 20, "start_s": 2.0, "duration_s": 0.1},
        },
        "nominal_replay_tolerance": REPLAY_TOLERANCE,
        "replay_checked_fields": list(REPLAY_FIELDS),
        "replay_state_check": "each step also requires identical SHA256 of little-endian float64 qpos+qvel",
        "quality_criteria": RUNNER.TRACKING_QUALITY_CRITERIA,
        "failure_type_threshold": "bias share >= .8: bias-dominated; <= .2: fluctuation-dominated; descriptive only",
        "decision_thresholds": {
            "remove_fz": "mean height RMSE decrease >= .002m; velocity RMSE increase <= .005m/s; no quality-count loss",
            "live_vs_replay": "mean perturbed tail RMSE decrease >= max(.005m/s,10% of replay); no quality-count loss",
        },
        "hypotheses": [
            "Fz is an unnecessary constant bias",
            "raw Gaussian mean drifts outside action bounds",
            "RL feedback has no measurable benefit beyond its nominal tape",
            "bumps failures mix bias and variance",
        ],
        "raw_gaussian_probe": "current observation; deterministic evaluation; no extra Gaussian sample is executed",
        "history_use": "historical holdout classification only, never used to select this matrix or a new model",
    }
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / "episodes").mkdir()
    write_json(output / "protocol.json", protocol)
    write_json(output / "source_hashes.json", source_hashes)
    history, input_hashes = classify_history(args.checkpoint_root / "metrics.csv")
    write_csv(output / "historical_failure_types.csv", history)
    for seed in args.seeds:
        path = args.checkpoint_root / "training" / f"mixed_seed_{seed}"
        for name in ("model.zip", "metadata.json"):
            input_hashes[str(path / name)] = sha256(path / name)
    metrics, replay_checks = [], []
    started = time.perf_counter()

    def evaluate(
        case: dict, mode: str, seed: int | None, perturbation: str = "nominal", tape: Any = None
    ):
        label = f"{case['case_id']}_{perturbation}_{mode}_seed_{seed}"
        rows, result = run_episode(
            case, mode, model=models.get(seed), perturbation=perturbation, tape=tape
        )
        path = output / "episodes" / f"{label}.csv"
        write_csv(path, rows)
        metrics.append(
            {
                "case_id": case["case_id"],
                "terrain_kind": case["terrain"]["kind"],
                "training_seed": seed,
                "mode": mode,
                "perturbation": perturbation,
                "telemetry_csv": str(path.relative_to(output)),
                **result,
            }
        )
        return rows

    for case in selected:
        print(f"Diagnosing {case['case_id']}", flush=True)
        for mode in ("zero", "constant_fz"):
            evaluate(case, mode, None)
        tapes = {}
        for seed in args.seeds:
            nominal = evaluate(case, "full", seed)
            tape = np.asarray(
                [[row["action_longitudinal"], row["action_vertical"]] for row in nominal]
            )
            replay = evaluate(case, "replay", seed, tape=tape)
            difference = verify_nominal_replay(nominal, replay)
            replay_checks.append(
                {
                    "case_id": case["case_id"],
                    "training_seed": seed,
                    "steps": len(nominal),
                    "max_transition_difference": difference,
                }
            )
            tapes[seed] = tape
            for mode in ("fx_only", "fz_only"):
                evaluate(case, mode, seed)
        if case["case_id"] in feedback_ids:
            for perturbation in PERTURBATIONS:
                for mode in ("zero", "constant_fz"):
                    evaluate(case, mode, None, perturbation)
                for seed in args.seeds:
                    evaluate(case, "full", seed, perturbation)
                    evaluate(case, "replay", seed, perturbation, tapes[seed])
    write_csv(output / "metrics.csv", metrics)
    unchanged = all(sha256(ROOT / name) == digest for name, digest in source_hashes.items())
    if not all(sha256(Path(name)) == digest for name, digest in input_hashes.items()):
        raise RuntimeError("an input model or historical artifact changed during diagnosis")
    summary = {
        "episodes": len(metrics),
        "wall_time_s": time.perf_counter() - started,
        "source_unchanged_during_run": unchanged,
        "nominal_replay_checks": replay_checks,
        "paired_comparisons": paired_summary(metrics, args.seeds),
        "interpretation": "controlled diagnostic, not a new holdout or proof of RL benefit; seeds/cases, not control steps, are comparison units",
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "manifest.json",
        {
            "input_sha256": input_hashes,
            "sha256": {
                str(path.relative_to(output)): sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
            "excludes": ["manifest.json", "files added after the run"],
        },
    )
    if not unchanged:
        raise RuntimeError("source changed during diagnosis; artifacts retained but not frozen")
    return summary


def main() -> None:
    try:
        summary = run(build_parser().parse_args())
    except (ValueError, FileExistsError, AssertionError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
