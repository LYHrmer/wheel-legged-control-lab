"""Train paired flat/terrain PPO policies and evaluate frozen final checkpoints.

The baseline mode needs only core dependencies. Training uses optional CPU
PyTorch/SB3 dependencies; it never loads a legacy 42-value D1 policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import resource
import time
from collections import Counter
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from numbers import Real
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from wheel_legged_control.provenance import capture_git_provenance

ROOT = Path(__file__).resolve().parents[1]
CONTROL_DT_S = 0.01
OBSERVATION_SCHEMA = "d1-terrain-oracle-v1"
ACTION_SCHEMA = "d1-terrain-residual-quarter-v1"
PROFILE_SCHEMAS = {
    "legacy-v1": {
        "observation_schema": OBSERVATION_SCHEMA,
        "reward_schema": "d1-terrain-clearance-v1",
        "control_schema": None,
    },
    "tracking-v2": {
        "observation_schema": "d1-terrain-tracking-oracle-v2",
        "reward_schema": "d1-terrain-tracking-v2",
        "control_schema": "d1-lqr-vmc-local-tangent-v2",
    },
}
TRACKING_QUALITY_CRITERIA = {
    "minimum_episode_seconds": 4.0,
    "progress_fraction_min": 0.65,
    "progress_fraction_max": 1.35,
    "tail_window_seconds": 1.0,
    "tail_velocity_rmse_absolute_mps": 0.06,
    "tail_velocity_rmse_target_fraction": 0.25,
    "clearance_rmse_max_m": 0.030,
    "max_abs_yaw_rad": 0.15,
    "mean_torque_saturation_fraction_max": 0.05,
}
TRACKING_INFO_FIELDS = (
    "command_roll_rad",
    "command_pitch_rad",
    "reward_roll_target_rad",
    "reward_pitch_target_rad",
    "observation_roll_target_rad",
    "observation_pitch_target_rad",
    "measured_roll_rad",
    "measured_pitch_rad",
    "roll_error_rad",
    "pitch_error_rad",
)
PPO_SETTINGS = {
    "learning_rate": 3e-4,
    "n_steps": 128,
    "batch_size": 128,
    "n_epochs": 4,
    "gamma": math.exp(-CONTROL_DT_S / 2.0),
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "policy_kwargs": {"net_arch": [64, 64]},
}
INFO_FIELDS = (
    "position_x_m",
    "position_y_m",
    "yaw_rad",
    "forward_velocity_mps",
    "command_velocity_mps",
    "velocity_error_mps",
    "clearance_m",
    "command_clearance_m",
    "clearance_error_m",
    "ground_height_m",
    "ground_pitch_rad",
    "ground_roll_rad",
    "torque_saturation_fraction",
    "initial_height_lift_m",
)
TELEMETRY_FIELDS = (
    "step",
    "time_s",
    *INFO_FIELDS,
    "initial_position_x_m",
    "action_longitudinal",
    "action_vertical",
    "policy_enabled",
    "policy_gated",
    "reward",
    "terminated",
    "truncated",
    "termination_reason",
)


class TrainingRewardScale(gym.RewardWrapper):
    """Change training reward units without altering info or physical transitions.

    Applied outside SB3's Monitor: episode monitoring and reward_terms retain
    raw physical-task rewards, while PPO receives the scaled scalar.
    """

    def __init__(self, env: gym.Env, scale: float):
        if isinstance(scale, bool) or not isinstance(scale, Real) or not math.isfinite(scale):
            raise ValueError("reward scale must be a finite positive number")
        if scale <= 0.0:
            raise ValueError("reward scale must be a finite positive number")
        super().__init__(env)
        self.scale = float(scale)

    def reward(self, reward: float) -> float:
        return float(reward) * self.scale


def reward_scale(args: argparse.Namespace) -> float:
    return (
        float(args.reward_scale)
        if args.reward_scale is not None
        else (0.01 if args.profile == "tracking-v2" else 1.0)
    )


def training_conditions(args: argparse.Namespace) -> list[str]:
    if args.conditions is not None:
        return list(args.conditions)
    return (
        ["flat", "mixed", "curriculum"] if args.profile == "tracking-v2" else ["flat", "curriculum"]
    )


def profile_environment(profile: str) -> type:
    if profile == "tracking-v2":
        from wheel_legged_control.d1.terrain_tracking_env import D1TerrainTrackingEnv

        return D1TerrainTrackingEnv
    if profile != "legacy-v1":
        raise ValueError("unknown experiment profile")
    from wheel_legged_control.d1.terrain_env import D1TerrainResidualEnv

    return D1TerrainResidualEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new experiment directory")
    parser.add_argument("--mode", choices=("baseline", "train"), default="train")
    parser.add_argument("--profile", choices=tuple(PROFILE_SCHEMAS), default="legacy-v1")
    parser.add_argument(
        "--reward-scale", type=float, help="training only; v1 default 1, v2 default .01"
    )
    parser.add_argument("--conditions", choices=("flat", "mixed", "curriculum"), nargs="+")
    parser.add_argument("--steps", type=int, default=8192, help="per seed and training condition")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--envs", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--episode-seconds", type=float, default=4.0)
    parser.add_argument(
        "--evaluation-split", choices=("development", "holdout"), default="development"
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def actual_budget(requested_steps: int, n_envs: int) -> int:
    """SB3 finishes whole rollouts, so freeze the rounded budget before training."""

    if requested_steps <= 0 or n_envs not in (1, 2, 4):
        raise ValueError("steps must be positive and envs must be 1, 2, or 4")
    rollout = PPO_SETTINGS["n_steps"] * n_envs
    return math.ceil(requested_steps / rollout) * rollout


def curriculum_stage(num_timesteps: int, total_timesteps: int) -> int:
    if num_timesteps < 0 or total_timesteps <= 0:
        raise ValueError("timesteps must be nonnegative and total_timesteps positive")
    return min(3, 4 * num_timesteps // total_timesteps)


def evaluation_cases(split: str, profile: str = "legacy-v1") -> list[dict[str, Any]]:
    """Sixteen fixed cases; no model scores enter their construction.

    Holdout bumps use wavelengths absent from training, and ramps use +/-3
    degrees rather than the training +/-2 and +/-4 degrees. Flat cases remain
    an in-distribution retention check, not unseen terrain.
    """

    if split not in {"development", "holdout"}:
        raise ValueError("unknown evaluation split")
    if profile not in PROFILE_SCHEMAS:
        raise ValueError("unknown experiment profile")
    tracking = profile == "tracking-v2"
    wavelengths = (0.7, 0.9, 1.1) if split == "holdout" else (0.8, 1.0, 1.2)
    phases = (0.37, 1.41) if split == "holdout" else (0.0, math.pi / 2.0)
    ramp_angle = 3.0 if split == "holdout" else 2.0
    if tracking and split == "holdout":
        wavelengths, phases = (0.75, 0.95, 1.15), (0.53, 2.17)
    cases: list[dict[str, Any]] = []

    def add(terrain: dict[str, Any], velocity: float, label: str) -> None:
        index = len(cases)
        cases.append(
            {
                "case_id": f"{'tracking_v2_' if tracking else ''}{split}_{index:02d}_{label}",
                "terrain": terrain,
                "velocity_mps": velocity,
                "height_m": 0.455,
                "environment_seed": (
                    (44000 if split == "holdout" else 33000)
                    if tracking
                    else (24000 if split == "holdout" else 13000)
                )
                + index,
            }
        )

    for velocity in (0.20, 0.35, 0.45, 0.30):
        add({"kind": "flat"}, velocity, "flat")
    for amplitude in (0.005, 0.010):
        for index in range(6 if tracking else 4):
            add(
                {
                    "kind": "bumps",
                    "amplitude_m": amplitude,
                    "wavelength_m": wavelengths[
                        index // 2 if tracking else index % len(wavelengths)
                    ],
                    "phase_rad": phases[index % len(phases)],
                },
                (0.25, 0.40, 0.35, 0.30, 0.40, 0.25)[index],
                "bumps",
            )
    slopes = (-ramp_angle, ramp_angle)
    if tracking:
        slopes = (-3.5, -2.5, 2.5, 3.5) if split == "holdout" else (-4.0, -2.0, 2.0, 4.0)
    for slope in slopes:
        for velocity in (0.25, 0.40):
            add({"kind": "ramp", "slope_deg": slope}, velocity, "ramp")
    return cases


def validate_args(args: argparse.Namespace) -> None:
    budget = actual_budget(args.steps, args.envs)
    if args.mode == "train" and budget < 2 * PPO_SETTINGS["n_steps"] * args.envs:
        raise ValueError("training requires at least two complete PPO rollouts")
    if not math.isfinite(args.episode_seconds) or args.episode_seconds < 0.1:
        raise ValueError("episode-seconds must be finite and at least 0.1")
    if not args.seeds or any(seed < 0 for seed in args.seeds):
        raise ValueError("seeds must be nonnegative")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("duplicate training seeds do not provide independent runs")
    scale = reward_scale(args)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("reward-scale must be finite and positive")
    conditions = training_conditions(args)
    if len(set(conditions)) != len(conditions):
        raise ValueError("duplicate training conditions are not independent experiments")
    if args.profile == "legacy-v1" and "mixed" in conditions:
        raise ValueError("mixed training requires the tracking-v2 profile")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _source_hashes() -> dict[str, str]:
    paths = [Path(__file__), ROOT / "pyproject.toml"]
    # Include indirectly used controller/model modules and local robot assets.
    paths.extend(
        path
        for path in (ROOT / "src" / "wheel_legged_control").rglob("*")
        if path.is_file() and path.suffix.lower() in {".py", ".xml", ".urdf", ".stl"}
    )
    return {str(path.relative_to(ROOT)): _sha256(path) for path in sorted(set(paths))}


def build_protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 2 if args.profile == "tracking-v2" else 1,
        "profile": args.profile,
        "question": "Does terrain curriculum PPO improve a fixed LQR+VMC residual baseline?",
        "mode": args.mode,
        "evaluation_split": args.evaluation_split,
        "training_conditions": training_conditions(args) if args.mode == "train" else [],
        "training_reward_scale": reward_scale(args),
        "reward_logging": "PPO sees scaled rewards; Monitor, info reward_terms and evaluation retain raw rewards",
        "quality_criteria": TRACKING_QUALITY_CRITERIA if args.profile == "tracking-v2" else None,
        "training_seeds": args.seeds if args.mode == "train" else [],
        "requested_steps_per_condition_seed": args.steps,
        "actual_budget_per_condition_seed": actual_budget(args.steps, args.envs),
        "n_envs": args.envs,
        "episode_seconds": args.episode_seconds,
        "control_dt_s": CONTROL_DT_S,
        "ppo": PPO_SETTINGS,
        "device": "cpu",
        "torch_num_threads": 1,
        "vector_start_method": "spawn" if args.envs > 1 else "in_process",
        "baseline": "lqr",
        "domain_randomization": False,
        "velocity_command": "forward only; stand for 0.3 s, smoothstep ramp during 0.3-0.6 s, then hold",
        **PROFILE_SCHEMAS[args.profile],
        "observation_size": 44,
        "action_schema": ACTION_SCHEMA,
        "residual_scale_n": [11.25, 20.0],
        "state_and_terrain_reference": "oracle_simulator_state_and_collision_geometry",
        "curriculum": {
            "schedule": "four equal timestep ranges; changes apply at the next reset",
            "stage_0": "flat",
            "stage_1": "flat and 5 mm bumps",
            "stage_2": "flat, 5-10 mm bumps, +/-2 degree ramps",
            "stage_3": "retain easy terrain; add +/-4 degree ramps",
            "training_bump_wavelengths_m": [0.8, 1.0, 1.2],
        },
        "checkpoint_selection": "final fixed-budget checkpoint, never selected by evaluation",
        "pairing": "identical case geometry, command and environment seed for every condition",
        "baseline_replication": "one zero-residual episode per case, not one per training seed",
        "metric_semantics": {
            "completed": "time limit reached without task termination; not quality success",
            "errors": "post-transition state versus the command used for that transition",
            "policy_enabled": "RL action path enabled, even when its action happens to be zero",
            "torque_saturation": "mean fraction of the 16 commanded joints at their limits",
            "progress_fraction": "world-x displacement divided by integral of logged velocity command over observed steps; null for zero integral",
            "slip": "not measured; no unvalidated contact/slip proxy is reported",
        },
        "cases": evaluation_cases(args.evaluation_split, args.profile),
    }


def _load_rl_dependencies() -> tuple[Any, ...]:
    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.logger import configure
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as error:
        raise RuntimeError(
            'training requires optional dependencies: pip install -e ".[rl]"'
        ) from error
    torch.set_num_threads(1)
    return PPO, BaseCallback, make_vec_env, configure, DummyVecEnv, SubprocVecEnv


def _curriculum_callback(base: type, condition: str, budget: int) -> Any:
    class ExposureCallback(base):
        def __init__(self) -> None:
            super().__init__()
            self.pending_stage = 0
            if condition == "mixed":
                self.pending_stage = 3
            self.steps: Counter[int] = Counter()
            self.episodes_started: Counter[int] = Counter()
            self.episodes_completed: Counter[int] = Counter()
            self.terrain_steps: Counter[str] = Counter()
            self.stage_kind_steps: Counter[str] = Counter()
            self.ramp_slope_steps: Counter[str] = Counter()
            self.episode_starts: list[dict[str, Any]] = []

        def _on_step(self) -> bool:
            transitions = zip(self.locals["infos"], self.locals["dones"], strict=True)
            for env_index, (info, done) in enumerate(transitions):
                stage = int(info["curriculum_stage"])
                terrain = info["terrain_config"]
                kind = str(terrain["kind"])
                self.steps[stage] += 1
                self.terrain_steps[kind] += 1
                self.stage_kind_steps[f"{stage}:{kind}"] += 1
                if kind == "ramp":
                    self.ramp_slope_steps[f"{float(terrain['slope_deg']):g}"] += 1
                if info["episode_step"] == 1:
                    self.episodes_started[stage] += 1
                    self.episode_starts.append(
                        {
                            "global_timesteps_after_vector_step": self.num_timesteps,
                            "env_index": env_index,
                            "curriculum_stage": stage,
                            "terrain_config": dict(terrain),
                            "target_velocity_mps": float(info["target_velocity_mps"]),
                        }
                    )
                if done:
                    self.episodes_completed[stage] += 1
            next_stage = (
                curriculum_stage(self.num_timesteps, budget)
                if condition == "curriculum"
                else (3 if condition == "mixed" else 0)
            )
            if next_stage != self.pending_stage:
                self.training_env.env_method("set_curriculum_stage", next_stage)
                self.pending_stage = next_stage
            return True

        def statistics(self) -> dict[str, Any]:
            return {
                "stage_steps": {str(stage): self.steps[stage] for stage in range(4)},
                "stage_episodes_started": {
                    str(stage): self.episodes_started[stage] for stage in range(4)
                },
                "stage_episodes_completed": {
                    str(stage): self.episodes_completed[stage] for stage in range(4)
                },
                "terrain_steps": dict(self.terrain_steps),
                "stage_kind_steps": dict(self.stage_kind_steps),
                "ramp_slope_steps": dict(self.ramp_slope_steps),
                "final_stage_sampled": self.steps[3] > 0 if condition == "curriculum" else None,
            }

    return ExposureCallback()


def validate_checkpoint(
    metadata: dict[str, Any], model: Any, environment: type | None = None
) -> None:
    """Schema and shape are both mandatory; a legacy policy cannot be reused."""

    expected = {
        "observation_schema": OBSERVATION_SCHEMA,
        "action_schema": ACTION_SCHEMA,
    }
    if environment is not None:
        expected = {
            name: getattr(environment, name)
            for name in ("observation_schema", "reward_schema", "action_schema")
        }
        if getattr(environment, "control_schema", None) is not None:
            expected["control_schema"] = environment.control_schema
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(f"checkpoint {name} is incompatible; expected {value}")
    if model.observation_space.shape != (44,) or model.action_space.shape != (2,):
        raise ValueError("checkpoint must have 44 observations and 2 residual actions")


def train_policy(
    args: argparse.Namespace,
    *,
    condition: str,
    seed: int,
    output: Path,
    environment: type,
    dependencies: tuple[Any, ...],
    source_hashes: dict[str, str],
) -> tuple[Any, dict[str, Any]]:
    ppo, base_callback, make_vec_env, configure, dummy_vec, subproc_vec = dependencies
    output.mkdir(parents=True, exist_ok=False)
    budget = actual_budget(args.steps, args.envs)
    callback = _curriculum_callback(base_callback, condition, budget)
    scale = reward_scale(args)
    vector_env = make_vec_env(
        environment,
        n_envs=args.envs,
        seed=seed,
        vec_env_cls=subproc_vec if args.envs > 1 else dummy_vec,
        vec_env_kwargs={"start_method": "spawn"} if args.envs > 1 else None,
        env_kwargs={
            "baseline": "lqr",
            "episode_seconds": args.episode_seconds,
            "training_mode": condition,
            "randomize": False,
            "stage": 3 if condition == "mixed" else 0,
        },
        monitor_dir=str(output / "episodes"),
        wrapper_class=partial(TrainingRewardScale, scale=scale) if scale != 1.0 else None,
    )
    logger = None
    started = time.perf_counter()
    try:
        model = ppo("MlpPolicy", vector_env, **PPO_SETTINGS, seed=seed, device="cpu", verbose=0)
        logger = configure(
            str(output / "learning_metrics"), ["stdout", "csv"] if args.verbose else ["csv"]
        )
        model.set_logger(logger)
        model.learn(total_timesteps=budget, callback=callback)
        # SB3 usually dumps before its PPO update; explicitly flush the final
        # update as another row, rather than silently losing its diagnostics.
        logger.dump(step=model.num_timesteps)
        elapsed = time.perf_counter() - started
        if int(model.num_timesteps) != budget:
            raise RuntimeError("training did not complete the predeclared rollout budget")
        model_path = output / "model.zip"
        model.save(model_path)
        _write_json(output / "episode_starts.json", callback.episode_starts)
        metadata = {
            "condition": condition,
            "profile": args.profile,
            "training_reward_scale": scale,
            "training_seed": seed,
            "requested_timesteps": args.steps,
            "actual_timesteps": int(model.num_timesteps),
            "completed_rollouts": budget // (PPO_SETTINGS["n_steps"] * args.envs),
            "ppo_epochs_completed": int(getattr(model, "_n_updates", 0)),
            "ppo": PPO_SETTINGS,
            "observation_schema": environment.observation_schema,
            "reward_schema": environment.reward_schema,
            "action_schema": environment.action_schema,
            "control_schema": getattr(environment, "control_schema", None),
            "model_sha256": _sha256(model_path),
            "source_sha256": source_hashes,
            "wall_time_s": elapsed,
            "training_transitions_per_wall_second": budget / elapsed,
            "parent_process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "memory_scope": "process lifetime high-water mark; excludes worker-process RSS",
            "exposure": callback.statistics(),
            "episode_starts_json": "episode_starts.json",
            "episode_starts_sha256": _sha256(output / "episode_starts.json"),
            "checkpoint_selection": "final",
        }
        _write_json(output / "metadata.json", metadata)
        # Evaluate the serialized model, not just the still-live trainer.
        loaded = ppo.load(model_path, device="cpu")
        validate_checkpoint(metadata, loaded, environment)
        return loaded, metadata
    finally:
        vector_env.close()
        if logger is not None:
            logger.close()


def evaluate_case(
    environment: type,
    case: dict[str, Any],
    episode_seconds: float,
    *,
    model: Any = None,
    quality_criteria: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Log each complete control transition with no policy gating or reset reuse."""

    env = environment(
        baseline="lqr", episode_seconds=episode_seconds, training_mode="flat", randomize=False
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        observation, reset_info = env.reset(
            seed=case["environment_seed"],
            options={key: case[key] for key in ("terrain", "velocity_mps", "height_m")},
        )
        initial_x_m = float(reset_info["position_x_m"])
        for step in range(round(episode_seconds / CONTROL_DT_S)):
            action = (
                np.zeros(2, dtype=np.float32)
                if model is None
                else np.asarray(model.predict(observation, deterministic=True)[0], dtype=np.float32)
            )
            if action.shape != (2,) or not np.isfinite(action).all():
                raise ValueError("policy returned an invalid action")
            observation, reward, terminated, truncated, info = env.step(action)
            row = {
                "step": step + 1,
                "time_s": (step + 1) * CONTROL_DT_S,
                **{field: info[field] for field in INFO_FIELDS},
                "initial_position_x_m": initial_x_m,
                "action_longitudinal": float(action[0]),
                "action_vertical": float(action[1]),
                "policy_enabled": int(model is not None and info["policy_applied"]),
                "policy_gated": int(model is not None and not info["policy_applied"]),
                "reward": float(reward),
                "terminated": int(terminated),
                "truncated": int(truncated),
                "termination_reason": info["termination_reason"],
            }
            if (
                getattr(environment, "control_schema", None)
                == PROFILE_SCHEMAS["tracking-v2"]["control_schema"]
            ):
                row.update({field: info[field] for field in TRACKING_INFO_FIELDS})
            rows.append(row)
            if terminated or truncated:
                break
    finally:
        env.close()
    metrics = summarize_episode(
        rows,
        quality_criteria=quality_criteria,
        requested_duration_s=episode_seconds,
        target_velocity_mps=case["velocity_mps"],
    )
    metrics["evaluation_wall_time_s"] = time.perf_counter() - started
    return rows, metrics


def summarize_episode(
    rows: list[dict[str, Any]],
    *,
    quality_criteria: dict[str, float] | None = None,
    requested_duration_s: float | None = None,
    target_velocity_mps: float | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty episode")

    def mean(field: str) -> float:
        return float(np.mean([row[field] for row in rows]))

    def rmse(field: str) -> float:
        return float(np.sqrt(np.mean(np.square([row[field] for row in rows]))))

    last = rows[-1]
    progress_m = last["position_x_m"] - rows[0]["initial_position_x_m"]
    commanded_displacement_m = sum(row["command_velocity_mps"] for row in rows) * CONTROL_DT_S
    metrics = {
        "control_steps": len(rows),
        "simulated_duration_s": len(rows) * CONTROL_DT_S,
        "completed": int(bool(last["truncated"]) and not bool(last["terminated"])),
        "terminated": int(last["terminated"]),
        "termination_reason": last["termination_reason"],
        "velocity_rmse_mps": rmse("velocity_error_mps"),
        "clearance_rmse_m": rmse("clearance_error_m"),
        "max_abs_yaw_rad": max(abs(row["yaw_rad"]) for row in rows),
        "final_position_x_m": last["position_x_m"],
        "progress_m": progress_m,
        "commanded_displacement_m": commanded_displacement_m,
        "progress_fraction": (
            progress_m / commanded_displacement_m if abs(commanded_displacement_m) > 1e-12 else None
        ),
        "mean_torque_saturation_fraction": mean("torque_saturation_fraction"),
        "policy_enabled_fraction": mean("policy_enabled"),
        "policy_gated_fraction": mean("policy_gated"),
        "episode_return": float(sum(row["reward"] for row in rows)),
    }
    if quality_criteria is not None:
        criteria = quality_criteria
        tail = rows[-round(criteria["tail_window_seconds"] / CONTROL_DT_S) :]
        tail_rmse = float(np.sqrt(np.mean([row["velocity_error_mps"] ** 2 for row in tail])))
        metrics.update(
            tail_velocity_rmse_mps=tail_rmse,
            tail_velocity_mean_mps=float(np.mean([row["forward_velocity_mps"] for row in tail])),
            tail_command_velocity_mean_mps=float(
                np.mean([row["command_velocity_mps"] for row in tail])
            ),
            tail_observed_seconds=len(tail) * CONTROL_DT_S,
            quality_success=None,
            quality_failure_reasons="not_defined_for_short_or_nonpositive_command",
        )
        if (
            requested_duration_s is not None
            and requested_duration_s >= criteria["minimum_episode_seconds"]
            and target_velocity_mps is not None
            and target_velocity_mps > 0
        ):
            checks = {
                "incomplete_episode": bool(metrics["completed"])
                and len(rows) == round(requested_duration_s / CONTROL_DT_S),
                "progress": metrics["progress_fraction"] is not None
                and criteria["progress_fraction_min"]
                <= metrics["progress_fraction"]
                <= criteria["progress_fraction_max"],
                "tail_velocity": tail_rmse
                <= max(
                    criteria["tail_velocity_rmse_absolute_mps"],
                    criteria["tail_velocity_rmse_target_fraction"] * target_velocity_mps,
                ),
                "clearance": metrics["clearance_rmse_m"] <= criteria["clearance_rmse_max_m"],
                "yaw": metrics["max_abs_yaw_rad"] <= criteria["max_abs_yaw_rad"],
                "torque_saturation": metrics["mean_torque_saturation_fraction"]
                <= criteria["mean_torque_saturation_fraction_max"],
            }
            failures = [name for name, passed in checks.items() if not passed]
            metrics["quality_success"] = int(not failures)
            metrics["quality_failure_reasons"] = "|".join(failures)
    return metrics


def _dependency_versions() -> dict[str, str | None]:
    result = {}
    for name in ("numpy", "scipy", "mujoco", "gymnasium", "torch", "stable-baselines3"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    environment = profile_environment(args.profile)

    dependencies = _load_rl_dependencies() if args.mode == "train" else None
    protocol = build_protocol(args)
    source_hashes = _source_hashes()
    provenance = capture_git_provenance(ROOT)
    args.output.mkdir(parents=True, exist_ok=False)
    _write_json(args.output / "protocol.json", protocol)
    _write_json(
        args.output / "provenance.json",
        {
            **provenance,
            "source_sha256": source_hashes,
            "python_version": platform.python_version(),
            "dependency_versions": _dependency_versions(),
            "protocol_sha256": _sha256(args.output / "protocol.json"),
        },
    )
    policies: list[tuple[str, int | None, Any]] = [("zero_residual", None, None)]
    if dependencies is not None:
        for seed in args.seeds:
            for condition in training_conditions(args):
                print(f"Training {condition}, seed {seed}", flush=True)
                model, _ = train_policy(
                    args,
                    condition=condition,
                    seed=seed,
                    output=args.output / "training" / f"{condition}_seed_{seed}",
                    environment=environment,
                    dependencies=dependencies,
                    source_hashes=source_hashes,
                )
                policies.append((f"{condition}_rl", seed, model))
    evaluations = args.output / "evaluation"
    evaluations.mkdir()
    metrics = []
    for condition, seed, model in policies:
        label = condition if seed is None else f"{condition}_seed_{seed}"
        destination = evaluations / label
        destination.mkdir()
        print(f"Evaluating {label} ({args.evaluation_split})", flush=True)
        for case in protocol["cases"]:
            rows, result = evaluate_case(
                environment,
                case,
                args.episode_seconds,
                model=model,
                quality_criteria=protocol["quality_criteria"],
            )
            telemetry_path = destination / f"{case['case_id']}.csv"
            _write_csv(telemetry_path, rows, tuple(rows[0]))
            metrics.append(
                {
                    "condition": condition,
                    "training_seed": seed,
                    "case_id": case["case_id"],
                    "terrain_kind": case["terrain"]["kind"],
                    "environment_seed": case["environment_seed"],
                    "telemetry_csv": str(telemetry_path.relative_to(args.output)),
                    **result,
                }
            )
    _write_csv(args.output / "metrics.csv", metrics, tuple(metrics[0]))
    source_unchanged = _source_hashes() == source_hashes
    _write_json(
        args.output / "summary.json",
        {
            "episodes": len(metrics),
            "source_unchanged_during_run": source_unchanged,
            "baseline_episodes": sum(item["condition"] == "zero_residual" for item in metrics),
            "training_runs": len(policies) - 1,
            "interpretation": "fixed-budget simulation experiment; no convergence or hardware claim",
        },
    )
    _write_json(
        args.output / "manifest.json",
        {
            "sha256": {
                str(path.relative_to(args.output)): _sha256(path)
                for path in sorted(args.output.rglob("*"))
                if path.is_file()
            },
            "excludes": ["manifest.json", "files added after the run"],
        },
    )
    if not source_unchanged:
        raise RuntimeError(
            "source changed during this run; artifacts retained but run is not frozen"
        )
    return args.output


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except (ValueError, FileExistsError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    print(f"Results: {result}")


if __name__ == "__main__":
    main()
