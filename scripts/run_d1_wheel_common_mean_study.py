#!/usr/bin/env python3
"""Train and evaluate a paired wheel-common Gaussian-mean parameterization.

Only the policy mean changes: None is the unbounded control, .05 the candidate.
The frozen 82-observation/eight-action task, reward, curriculum and PPO settings
are shared. Distribution-mean bounds do not guarantee bounds after Box clipping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.d1_course_curriculum import D1CourseCurriculumEnv
from scripts.d1_wheel_common_mean_policy import D1WheelCommonMeanPolicy
from scripts.run_d1_course_curriculum import (
    forward_command,
    summarize_exposure,
)
from scripts.run_d1_course_curriculum import (
    source_hashes as curriculum_source_hashes,
)
from scripts.run_d1_locomotion_experiment import PPO_SETTINGS
from wheel_legged_control.d1.control_loop import D1MotionCommand
from wheel_legged_control.d1.locomotion_checkpoint import (
    load_locomotion_policy,
    write_checkpoint_metadata,
)
from wheel_legged_control.d1.locomotion_env import D1LocomotionEnv
from wheel_legged_control.d1.locomotion_terrain import D1LocomotionTerrainConfig
from wheel_legged_control.d1.state_provider import D1StateProviderConfig

SCHEMA = "d1-wheel-common-mean-study-v1"
EVALUATION_SCHEMA = "d1-wheel-common-mean-evaluation-v1"
POLICY_CLASS = "scripts.d1_wheel_common_mean_policy.D1WheelCommonMeanPolicy"
VARIANTS = {"unbounded": None, "bounded": 0.05}
TRAIN_SEEDS = (48001, 48002, 48003)
TRAIN_STEPS = 16384
TRAIN_SECONDS = 32.0


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot encode {type(value).__name__}")


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False, default=_json_value)
        stream.write("\n")


def source_hashes():
    result = curriculum_source_hashes()
    for path in (Path(__file__), ROOT / "scripts/d1_wheel_common_mean_policy.py"):
        result[str(path.relative_to(ROOT))] = sha256(path)
    return result


def write_manifest(directory):
    write_json(directory / "manifest.json", {
        str(path.relative_to(directory)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != directory / "manifest.json"
    })


def _finite(value, name, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return float(value)


def case_command(spec):
    """Stand, ramp forward and yaw together, then hold; height stays fixed."""
    expected = {"target_forward_mps", "target_yaw_rps", "settle_seconds", "ramp_seconds", "height_m"}
    if not isinstance(spec, dict) or set(spec) != expected:
        raise ValueError("case command fields must match the evaluation schema")
    values = {key: _finite(value, key) for key, value in spec.items()}
    if values["settle_seconds"] < 0 or values["ramp_seconds"] <= 0:
        raise ValueError("settle must be nonnegative and ramp strictly positive")
    D1MotionCommand(values["target_forward_mps"], values["target_yaw_rps"], values["height_m"])

    def command(time_s):
        fraction = float(np.clip(
            (time_s - values["settle_seconds"]) / values["ramp_seconds"], 0.0, 1.0
        ))
        return D1MotionCommand(values["target_forward_mps"] * fraction,
                               values["target_yaw_rps"] * fraction, values["height_m"])

    return command


def read_evaluation_protocol(path, split):
    protocol = json.loads(Path(path).read_text())
    if not isinstance(protocol, dict) or protocol.get("schema") != EVALUATION_SCHEMA:
        raise ValueError("unsupported evaluation protocol schema")
    cases = protocol.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("evaluation protocol needs a nonempty cases list")
    names = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
            "name", "split", "seed", "episode_seconds", "terrain", "command"
        }:
            raise ValueError("evaluation case fields must match the schema")
        name = case["name"]
        if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) is None:
            raise ValueError("case name must be a simple lowercase identifier")
        if name in names:
            raise ValueError("duplicate evaluation case name")
        names.add(name)
        if case["split"] not in ("dev", "final"):
            raise ValueError("case split must be dev or final")
        if type(case["seed"]) is not int or case["seed"] < 0:
            raise ValueError("case seed must be a nonnegative integer")
        seconds = _finite(case["episode_seconds"], "episode_seconds", minimum=.01)
        if not np.isclose(seconds/.01, round(seconds/.01), rtol=0, atol=1e-8):
            raise ValueError("case duration must be a multiple of .01 seconds")
        if not isinstance(case["terrain"], dict):
            raise TypeError("case terrain must be a configuration object")
        D1LocomotionTerrainConfig(**case["terrain"])
        case_command(case["command"])
    selected = [case for case in cases if case["split"] == split]
    if not selected:
        raise ValueError("requested evaluation split has no cases")
    return protocol, selected


def parameter_hash(policy):
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.shape, str(array.dtype))).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _check_variant(model, metadata, seed, variant, *, expected_steps=TRAIN_STEPS, smoke=False):
    extra = metadata.get("extra", {})
    expected = {"experiment_schema": SCHEMA, "policy_class": POLICY_CLASS,
                "seed": seed, "variant": variant,
                "wheel_common_mean_limit": VARIANTS[variant],
                "pipeline_smoke_only": smoke, "actual_transitions": expected_steps,
                "policy_module_sha256": sha256(ROOT / "scripts/d1_wheel_common_mean_policy.py")}
    for key, value in expected.items():
        if key not in extra or type(extra[key]) is not type(value) or extra[key] != value:
            raise ValueError(f"checkpoint {key} does not match the requested study variant")
    if not isinstance(model.policy, D1WheelCommonMeanPolicy):
        raise TypeError("checkpoint does not contain D1WheelCommonMeanPolicy")
    if model.num_timesteps != expected_steps:
        raise ValueError("checkpoint PPO transition count does not match the declared budget")
    if model.policy.wheel_common_mean_limit != VARIANTS[variant]:
        raise ValueError("checkpoint policy limit differs from its declared variant")


def train_variant(output, *, seed, variant, steps, smoke, expected_initial_hash=None):
    """One uninterrupted learn call; a separate reload check cannot affect exposure."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure

    torch.set_num_threads(1)
    output.mkdir()
    settings = deepcopy(PPO_SETTINGS)
    if smoke:
        settings.update(n_steps=32, batch_size=32, n_epochs=1)
    settings["policy_kwargs"]["wheel_common_mean_limit"] = VARIANTS[variant]
    env = D1CourseCurriculumEnv(condition="curriculum", total_steps=steps, seed=seed,
                                episode_seconds=TRAIN_SECONDS, command_source=forward_command)
    started = time.monotonic()
    try:
        model = PPO(D1WheelCommonMeanPolicy, env, seed=seed, device="cpu", verbose=0, **settings)
        initial_hash = parameter_hash(model.policy)
        if expected_initial_hash is not None and initial_hash != expected_initial_hash:
            raise RuntimeError("paired variants did not start with identical policy parameters")
        actual_hyperparameters = {
            "learning_rate": model.learning_rate, "n_steps": model.n_steps,
            "batch_size": model.batch_size, "n_epochs": model.n_epochs,
            "gamma": model.gamma, "gae_lambda": model.gae_lambda,
            "clip_range": float(model.clip_range(1.0)),
            "clip_range_vf": None if model.clip_range_vf is None else float(model.clip_range_vf(1.0)),
            "normalize_advantage": model.normalize_advantage, "ent_coef": model.ent_coef,
            "vf_coef": model.vf_coef, "max_grad_norm": model.max_grad_norm,
            "target_kl": model.target_kl, "use_sde": model.use_sde,
            "sde_sample_freq": model.sde_sample_freq, "num_envs": model.n_envs,
            "policy_kwargs": settings["policy_kwargs"], "device": str(model.device),
            "torch_threads": torch.get_num_threads(),
            "optimizer_class": type(model.policy.optimizer).__name__,
            "optimizer_defaults": model.policy.optimizer.defaults,
            "activation_function": model.policy.activation_fn.__name__,
        }
        write_json(output / "initial_policy.json", {
            "seed": seed, "variant": variant, "initialization_state_dict_sha256": initial_hash,
            "wheel_common_mean_limit": VARIANTS[variant], "num_timesteps": model.num_timesteps,
            "ppo_actual_hyperparameters": actual_hyperparameters,
        })
        model.set_logger(configure(str(output), ["csv"]))
        model.learn(total_timesteps=steps)
        if model.num_timesteps != steps or env.total_transitions != steps:
            raise RuntimeError("PPO and actual physical transition counts differ")
        # SB3 dumps before train(); flush the final update diagnostics explicitly.
        model.logger.dump(step=model.num_timesteps)
        model_path, metadata_path = output / "model.zip", output / "model.metadata.json"
        if model_path.exists() or metadata_path.exists():
            raise FileExistsError("checkpoint output already exists")
        model.save(model_path)
        metadata = write_checkpoint_metadata(model_path, env, metadata_path, extra={
            "experiment_schema": SCHEMA, "policy_class": POLICY_CLASS,
            "variant": variant, "seed": seed, "wheel_common_mean_limit": VARIANTS[variant],
            "baseline": "wheel_leg", "action_mode": "independent8", "state_source": "oracle",
            "pipeline_smoke_only": smoke, "actual_transitions": env.total_transitions,
            "initialization_state_dict_sha256": initial_hash,
            "policy_module_sha256": sha256(ROOT / "scripts/d1_wheel_common_mean_policy.py"),
        })
        check = D1LocomotionEnv(baseline="wheel_leg", action_mode="independent8",
                               provider_config=D1StateProviderConfig("oracle"), episode_seconds=.02)
        try:
            obs, _ = check.reset(seed=99173)
            loaded = load_locomotion_policy(model_path, metadata_path, check)
            _check_variant(loaded, metadata, seed, variant, expected_steps=steps, smoke=smoke)
            expected = model.predict(obs, deterministic=True)[0]
            actual = loaded.predict(obs, deterministic=True)[0]
            if not np.array_equal(expected, actual):
                raise RuntimeError("deterministic action changed after checkpoint reload")
            new_obs, reward, _, _, _ = check.step(actual)
            if not np.isfinite(new_obs).all() or not np.isfinite(reward):
                raise RuntimeError("reloaded model produced a nonfinite physical step")
        finally:
            check.close()
        summary = {
            "seed": seed, "variant": variant, "wheel_common_mean_limit": VARIANTS[variant],
            "actual_transitions": env.total_transitions, "ppo_num_timesteps": int(model.num_timesteps),
            "ppo_optimization_epochs": int(model._n_updates),
            "rollout_train_calls": steps // settings["n_steps"],
            "ppo_updates_definition": "SB3 _n_updates counts completed optimization epochs.",
            "pipeline_smoke_only": smoke, "reloaded_action_exact": True,
            "ppo_actual_hyperparameters": actual_hyperparameters,
            "reloaded_physics_step_finite": True, "reload_check_physical_steps": 1,
            "initialization_state_dict_sha256": initial_hash,
            "final_policy_parameter_sha256": parameter_hash(model.policy),
            "wall_seconds": time.monotonic() - started,
        }
    except Exception as error:
        write_json(output / "failure.json", {"type": type(error).__name__, "error": str(error),
                                             "actual_transitions": env.total_transitions})
        raise
    finally:
        env.close()
        write_json(output / "episodes.json", env.episode_records)
    if sum(record["transitions"] for record in env.episode_records) != steps:
        raise RuntimeError("episode records do not conserve physical transitions")
    summary.update(summarize_exposure(env.episode_records))
    write_json(output / "summary.json", summary)
    write_manifest(output)
    return summary


def policy_diagnostics(model, observation):
    """Record raw actor head, transformed Gaussian mean, and Box-clipped action."""
    import torch

    if model is None:
        raw = mean = std = action = np.zeros(8, dtype=np.float32)
    else:
        policy = model.policy
        policy.set_training_mode(False)
        tensor, _ = policy.obs_to_tensor(observation)
        with torch.no_grad():
            features = policy.extract_features(tensor)
            actor_features = features if policy.share_features_extractor else features[0]
            latent = policy.mlp_extractor.forward_actor(actor_features)
            raw = policy.action_net(latent).cpu().numpy()[0].copy()
            distribution = policy.get_distribution(tensor).distribution
            mean = distribution.mean.cpu().numpy()[0].copy()
            std = distribution.stddev.cpu().numpy()[0].copy()
        action = model.predict(observation, deterministic=True)[0]
        if not np.allclose(action, np.clip(mean, -1, 1), rtol=0, atol=2e-6):
            raise RuntimeError("deterministic policy action differs from clipped Gaussian mean")
    diagnostics = {
        "raw_net_mean": raw, "transformed_gaussian_mean": mean,
        "gaussian_std": std, "box_clipped_action": action,
        "raw_net_wheel_common_mean": float(np.mean(raw[4:])),
        "transformed_gaussian_wheel_common_mean": float(np.mean(mean[4:])),
        "box_clipped_wheel_common_mean": float(np.mean(action[4:])),
        "deterministic_box_clipped_fraction": float(np.mean((mean < -1) | (mean > 1))),
    }
    return action, diagnostics


def _initial_record(env, observation, info):
    return {"episode_metadata": info["episode_metadata"], **{
        name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for name, value in (("observation_sha256", observation),
                            ("qpos_sha256", env.plant.data.qpos),
                            ("qvel_sha256", env.plant.data.qvel))
    }}


def _evaluation_summary(label, rows, initial_x, actual_time):
    def rmse(field):
        return float(np.sqrt(np.mean([row["metrics"][field] ** 2 for row in rows])))

    common_fields = ("raw_net_wheel_common_mean", "transformed_gaussian_wheel_common_mean",
                     "box_clipped_wheel_common_mean", "executed_wheel_common_mean")
    return {
        "policy": label, "actual_transitions": len(rows), "integrated_duration_s": actual_time,
        "terminated": rows[-1]["terminated"], "truncated": rows[-1]["truncated"],
        "terminal_reason": rows[-1]["terminal_reason"],
        "initial_x_m": initial_x, "final_x_m": rows[-1]["metrics"]["x_m"],
        "signed_forward_progress_m": rows[-1]["metrics"]["x_m"] - initial_x,
        "velocity_rmse_mps": rmse("velocity_error_mps"),
        "yaw_rate_rmse_rps": rmse("yaw_rate_error_rps"), "height_rmse_m": rmse("height_error_m"),
        "roll_rmse_rad": rmse("roll_error_rad"), "pitch_rmse_rad": rmse("pitch_error_rad"),
        "cumulative_return": float(sum(row["reward"] for row in rows)),
        "cumulative_reward_terms": {name: float(sum(row["reward_terms"][name] for row in rows))
                                    for name in rows[0]["reward_terms"]},
        "mean_torque_input_clipped_fraction": float(np.mean(
            [row["metrics"]["torque_input_clipped_fraction"] for row in rows])),
        "mean_deterministic_box_clipped_fraction": float(np.mean(
            [row["deterministic_box_clipped_fraction"] for row in rows])),
        "wheel_common_mean_statistics": {field: {
            "mean": float(np.mean([row[field] for row in rows])),
            "min": min(row[field] for row in rows), "max": max(row[field] for row in rows),
            "max_abs": max(abs(row[field]) for row in rows),
        } for field in common_fields},
        "geometric_nonflat_exposure": rows[-1]["terrain_exposure"],
        "minimum_clearance_m": min(row["metrics"]["clearance_m"] for row in rows),
        "maximum_undesired_ground_contacts": max(row["metrics"]["undesired_ground_contacts"] for row in rows),
    }


def evaluate_one(output, *, case, label, checkpoint=None, expected_initial=None):
    """One episode, ending at its first termination; raw records survive errors."""
    output.mkdir(parents=True)
    env = D1LocomotionEnv(baseline="wheel_leg", action_mode="independent8",
                          provider_config=D1StateProviderConfig("oracle"),
                          terrain=D1LocomotionTerrainConfig(**case["terrain"]),
                          episode_seconds=case["episode_seconds"],
                          command_source=case_command(case["command"]))
    rows, observations, positions, velocities, times = [], [], [], [], []
    initial = None
    started = time.monotonic()

    def state_record(observation):
        observations.append(observation.copy())
        positions.append(env.plant.data.qpos.copy())
        velocities.append(env.plant.data.qvel.copy())
        times.append(float(env.plant.data.time))

    try:
        observation, info = env.reset(seed=case["seed"])
        initial = _initial_record(env, observation, info)
        write_json(output / "initial.json", initial)
        if expected_initial is not None and initial != expected_initial:
            raise RuntimeError("policy comparison initial state/observation/metadata differs")
        model = None
        if checkpoint is not None:
            model = load_locomotion_policy(checkpoint["model"], checkpoint["metadata"], env)
            metadata = json.loads(checkpoint["metadata"].read_text())
            _check_variant(model, metadata, checkpoint["seed"], checkpoint["variant"])
        initial_x = float(env.plant.data.qpos[0])
        state_record(observation)
        with (output / "trace.jsonl").open("x") as stream:
            while True:
                action, diagnostic = policy_diagnostics(model, observation)
                observation, reward, terminated, truncated, info = env.step(action)
                if not np.isfinite(observation).all() or not np.isfinite(reward):
                    raise RuntimeError("nonfinite observation or reward")
                row = {"transition": len(rows) + 1, "reward": reward,
                       "terminated": bool(terminated), "truncated": bool(truncated),
                       **diagnostic, **info,
                       "executed_wheel_common_mean": float(np.mean(info["applied_action"][4:]))}
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False, default=_json_value) + "\n")
                stream.flush()
                rows.append(row)
                state_record(observation)
                if terminated or truncated:
                    break
        summary = _evaluation_summary(label, rows, initial_x, float(env.plant.data.time))
        summary.update(initial_state_observation_metadata_same=True,
                       wall_seconds=time.monotonic() - started)
    except Exception as error:  # noqa: BLE001 -- Record failure and finish the remaining declared cases.
        summary = {"policy": label, "error_type": type(error).__name__, "error": str(error),
                   "completed_recorded_transitions": len(rows), "evaluation_completed": False}
        write_json(output / "failure.json", summary)
    finally:
        with (output / "states.npz").open("xb") as stream:
            np.savez_compressed(stream, observation=np.asarray(observations),
                                qpos=np.asarray(positions), qvel=np.asarray(velocities),
                                time_s=np.asarray(times))
        env.close()
    write_json(output / "summary.json", summary)
    write_manifest(output)
    return summary, initial


def checkpoints_for(models_root, seeds):
    checkpoints = []
    for seed in seeds:
        for variant in VARIANTS:
            directory = models_root / f"seed{seed}" / variant
            paths = {name: directory / filename for name, filename in
                     (("model", "model.zip"), ("metadata", "model.metadata.json"))}
            if not all(path.is_file() for path in paths.values()):
                raise FileNotFoundError(f"missing checkpoint pair in {directory}")
            checkpoints.append({"seed": seed, "variant": variant,
                                "label": f"seed{seed}/{variant}", **paths})
    return checkpoints


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke", "train", "evaluate"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--steps", type=int, help="Smoke: positive multiple of 32; train fixed at 16384")
    parser.add_argument("--models", type=Path, help="Evaluate ROOT/seedN/{unbounded,bounded}/model.zip")
    parser.add_argument("--evaluation-protocol", type=Path, help="Frozen case JSON; required for evaluate")
    parser.add_argument("--cases", choices=("dev", "final"), default="dev")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    smoke = args.mode == "smoke"
    seeds = args.seeds if args.seeds is not None else ([48999] if smoke else list(TRAIN_SEEDS))
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be unique nonnegative integers")
    steps = args.steps if args.steps is not None else (64 if smoke else TRAIN_STEPS)
    if smoke and (steps <= 0 or steps % 32):
        raise ValueError("smoke steps must be a positive multiple of 32")
    if not smoke and args.steps is not None and steps != TRAIN_STEPS:
        raise ValueError("training budget is fixed at 16384 steps per checkpoint")
    if args.mode == "evaluate" and (args.models is None or args.evaluation_protocol is None):
        raise ValueError("evaluate requires --models and --evaluation-protocol")
    if args.mode != "evaluate" and args.models is not None:
        raise ValueError("--models is only accepted for evaluation")
    evaluation_protocol, cases, evaluation_sha = None, [], None
    if args.evaluation_protocol is not None:
        evaluation_sha = sha256(args.evaluation_protocol)
        evaluation_protocol, cases = read_evaluation_protocol(args.evaluation_protocol, args.cases)
        if sha256(args.evaluation_protocol) != evaluation_sha:
            raise RuntimeError("evaluation protocol changed while being read")
    checkpoints = checkpoints_for(args.models, seeds) if args.mode == "evaluate" else []
    checkpoint_hashes = {str(item[key]): sha256(item[key]) for item in checkpoints
                         for key in ("model", "metadata")}
    output = args.output.absolute()
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError("output must not use symlinks")
    if output.exists():
        raise FileExistsError(output)
    import torch
    torch.set_num_threads(1)
    before = source_hashes()
    settings = deepcopy(PPO_SETTINGS)
    if smoke:
        settings.update(n_steps=32, batch_size=32, n_epochs=1)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", {
        "schema": SCHEMA, "mode": args.mode, "seeds": seeds,
        "variants": VARIANTS, "policy_class": POLICY_CLASS,
        "training_condition": "curriculum", "training_episode_seconds": TRAIN_SECONDS,
        "training_steps_per_checkpoint": steps if args.mode != "evaluate" else None,
        "training_total_steps": steps * len(seeds) * len(VARIANTS) if args.mode != "evaluate" else 0,
        "pipeline_smoke_only": smoke, "ppo_settings": settings,
        "torch_threads": torch.get_num_threads(),
        "training_command": "forward_command: stand .5s, ramp .5s to .25m/s; yaw0; height.455m",
        "source_sha256": before, "checkpoint_sha256": checkpoint_hashes,
        "versions": {name: version(name) for name in
                     ("numpy", "mujoco", "gymnasium", "stable-baselines3", "torch")},
        "evaluation_protocol": evaluation_protocol,
        "evaluation_protocol_sha256": evaluation_sha,
        "selected_split": args.cases if args.mode == "evaluate" else None,
        "selected_case_names": [case["name"] for case in cases] if args.mode == "evaluate" else [],
        "deterministic_evaluation": True, "zero_runs_per_case": 1,
        "evaluation_rule": "Each case: zero once and all requested final checkpoints; first terminal ends the episode; no retries or checkpoint selection.",
        "mean_bound_scope": "Gaussian mean before Box clipping; stochastic samples and clipped deterministic actions need not preserve the common bound.",
    })
    results = []
    try:
        if args.mode == "evaluate":
            for case in cases:
                case_output = output / case["name"]
                zero, initial = evaluate_one(case_output / "zero", case=case, label="zero")
                results.append({"case": case["name"], **zero})
                if initial is None or "error" in zero:
                    raise RuntimeError("zero comparison failed; no valid common initial reference")
                for checkpoint in checkpoints:
                    summary, _ = evaluate_one(case_output / checkpoint["label"], case=case,
                                               label=checkpoint["label"], checkpoint=checkpoint,
                                               expected_initial=initial)
                    results.append({"case": case["name"], **summary})
        else:
            for seed in seeds:
                seed_output = output / f"seed{seed}"
                seed_output.mkdir()
                initial_hash = None
                for variant in VARIANTS:
                    summary = train_variant(seed_output / variant, seed=seed, variant=variant,
                                            steps=steps, smoke=smoke,
                                            expected_initial_hash=initial_hash)
                    initial_hash = summary["initialization_state_dict_sha256"]
                    results.append(summary)
        unchanged = source_hashes() == before and all(
            sha256(path) == value for path, value in checkpoint_hashes.items()
        )
        unchanged = unchanged and (args.evaluation_protocol is None or
                                   sha256(args.evaluation_protocol) == evaluation_sha)
        if not unchanged:
            raise RuntimeError("source or evaluation checkpoint changed during the study")
        complete = all("error" not in result for result in results)
        write_json(output / "summary.json", {
            "schema": SCHEMA, "mode": args.mode, "results": results,
            "source_and_checkpoints_unchanged": True, "all_runs_completed": complete,
            "skill_improvement_established": False,
        })
    except Exception as error:
        write_json(output / "failure.json", {"type": type(error).__name__, "error": str(error)})
        raise
    finally:
        write_manifest(output)
    print(json.dumps({"output": str(output), "mode": args.mode,
                      "runs": len(results), "all_runs_completed": complete}))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
