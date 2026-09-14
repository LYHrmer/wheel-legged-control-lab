#!/usr/bin/env python3
"""Run an independent v3 terrain curriculum experiment or a short pipeline check.

This uses the existing 82-input, eight-residual wheel/leg task. It does not train
jumping, lateral foot placement, or the separate course GUI controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if __package__:
    from .d1_course_curriculum import D1CourseCurriculumEnv, scaled_training_terrain
    from .run_d1_locomotion_experiment import PPO_SETTINGS
else:
    from d1_course_curriculum import D1CourseCurriculumEnv, scaled_training_terrain
    from run_d1_locomotion_experiment import PPO_SETTINGS

from wheel_legged_control.d1.control_loop import D1MotionCommand
from wheel_legged_control.d1.locomotion_checkpoint import (
    load_locomotion_policy,
    write_checkpoint_metadata,
)
from wheel_legged_control.d1.locomotion_env import D1LocomotionEnv

SCHEMA = "d1-course-curriculum-v1"


def forward_command(time_s):
    """The same settle/ramp/forward command is used in every condition."""
    fraction = float(np.clip((time_s - 0.5) / 0.5, 0.0, 1.0))
    return D1MotionCommand(forward_velocity_mps=0.25 * fraction)


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def source_hashes():
    paths = [ROOT / "pyproject.toml", *sorted((ROOT / "src").rglob("*.py"))]
    paths += sorted(
        p
        for p in (ROOT / "src/wheel_legged_control").rglob("*")
        if p.is_file() and p.suffix.lower() in {".xml", ".urdf", ".stl"}
    )
    paths += [
        Path(__file__),
        ROOT / "scripts/d1_course_curriculum.py",
        ROOT / "scripts/run_d1_locomotion_experiment.py",
    ]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def summarize_exposure(records):
    """Scheduled difficulty and actual geometric exposure remain separate."""
    steps, episodes, nonflat = Counter(), Counter(), Counter()
    for record in records:
        level = str(record["level"])
        steps[level] += record["transitions"]
        episodes[level] += 1
        nonflat[level] += (record["last_terrain_exposure"] or {}).get("nonflat_steps", 0)
    return {
        "transitions_by_level": dict(steps),
        "episodes_by_level": dict(episodes),
        "geometric_nonflat_steps_by_level": dict(nonflat),
        "definition": "Geometric exposure is not verified obstacle traversal.",
    }


def train_condition(output, *, condition, seed, steps, episode_seconds, smoke):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure

    torch.set_num_threads(1)
    output.mkdir()
    settings = dict(PPO_SETTINGS)
    if smoke:
        settings.update(n_steps=32, batch_size=32, n_epochs=1)
    env = D1CourseCurriculumEnv(
        condition=condition,
        total_steps=steps,
        seed=seed,
        episode_seconds=episode_seconds,
        command_source=forward_command,
    )
    started = time.monotonic()
    try:
        model = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0, **settings)
        model.set_logger(configure(str(output), ["csv"]))
        # SB3 performs the first seed/reset. The wrapper allows this initial
        # seed while preventing a training stream from being rewound later.
        model.learn(total_timesteps=steps)
        if model.num_timesteps != steps or env.total_transitions != steps:
            raise RuntimeError("PPO and actual environment transition counts differ")
        model_path = output / "model.zip"
        model.save(model_path)
        write_checkpoint_metadata(
            model_path,
            env,
            output / "model.metadata.json",
            extra={
                "experiment_schema": SCHEMA,
                "condition": condition,
                "seed": seed,
                "pipeline_smoke_only": smoke,
                "actual_transitions": env.total_transitions,
            },
        )
        # A separate reset environment prevents the reload check from changing
        # curriculum exposure counts or performing an extra training step.
        check = D1LocomotionEnv(
            baseline="wheel_leg", action_mode="independent8", episode_seconds=0.02
        )
        try:
            obs, _ = check.reset(seed=99173)
            loaded = load_locomotion_policy(model_path, output / "model.metadata.json", check)
            expected = model.predict(obs, deterministic=True)[0]
            actual = loaded.predict(obs, deterministic=True)[0]
            if not np.array_equal(expected, actual):
                raise RuntimeError("saved policy changes its deterministic action after reload")
            new_obs, reward, _, _, _ = check.step(actual)
            if not np.isfinite(new_obs).all() or not np.isfinite(reward):
                raise RuntimeError("reloaded policy produced a nonfinite physical step")
        finally:
            check.close()
        summary = {
            "condition": condition,
            "seed": seed,
            "pipeline_smoke_only": smoke,
            "actual_transitions": env.total_transitions,
            "ppo_updates": int(model._n_updates),
            "ppo_updates_definition": "SB3 completed optimisation epochs (_n_updates).",
            "rollout_train_calls": steps // settings["n_steps"],
            "reloaded_action_exact": True,
            "reloaded_physics_step_finite": True,
            "wall_seconds": time.monotonic() - started,
        }
    finally:
        env.close()
        write_json(output / "episodes.json", env.episode_records)
    summary.update(summarize_exposure(env.episode_records))
    if sum(r["transitions"] for r in env.episode_records) != steps:
        raise RuntimeError("episode exposure records do not conserve actual transitions")
    write_json(output / "summary.json", summary)
    return summary


def probe_levels(output, seed, seconds):
    """Real zero-residual forward runs; no checkpoint selection or holdout use."""
    reports = []
    for level in range(4):
        case = output / f"level{level}"
        case.mkdir()
        terrain = scaled_training_terrain(level, 0)
        env = D1LocomotionEnv(
            baseline="wheel_leg",
            action_mode="independent8",
            terrain=terrain,
            episode_seconds=seconds,
            command_source=lambda _: D1MotionCommand(forward_velocity_mps=0.25),
        )
        rows = []
        try:
            env.reset(seed=seed)
            start_x = float(env.plant.data.qpos[0])
            while True:
                obs, reward, terminated, truncated, info = env.step(np.zeros(8))
                if not np.isfinite(obs).all() or not np.isfinite(reward):
                    raise RuntimeError("nonfinite probe observation or reward")
                rows.append(
                    {
                        **info["metrics"],
                        "reward": reward,
                        "terrain_exposure": info["terrain_exposure"],
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                )
                if terminated or truncated:
                    break
            summary = {
                "level": level,
                "seed": seed,
                "terrain": asdict(terrain),
                "actual_transitions": len(rows),
                "forward_target_mps": 0.25,
                "forward_progress_m": rows[-1]["x_m"] - start_x,
                "terminal_reason": info["terminal_reason"],
                "geometric_exposure": info["terrain_exposure"],
                "velocity_rmse_mps": float(
                    np.sqrt(np.mean([r["velocity_error_mps"] ** 2 for r in rows]))
                ),
                "minimum_body_clearance_m": min(r["clearance_m"] for r in rows),
                "definition": "Forward/terrain exposure probe, not full-road or stair success.",
            }
            write_json(case / "trace.json", rows)
            write_json(case / "summary.json", summary)
            reports.append(summary)
        finally:
            env.close()
    return reports


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke", "train", "probe"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("flat", "mixed", "curriculum"),
        default=["flat", "mixed", "curriculum"],
    )
    parser.add_argument("--seed", type=int, default=47000)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Per condition; smoke defaults to 256, train to 4096.",
    )
    parser.add_argument(
        "--episode-seconds",
        type=float,
        default=None,
        help="Smoke defaults to .32; train/probe to 12.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    smoke = args.mode == "smoke"
    steps = args.steps if args.steps is not None else (256 if smoke else 4096)
    seconds = args.episode_seconds if args.episode_seconds is not None else (0.32 if smoke else 12)
    rollout = 32 if smoke else 128
    if args.seed < 0 or steps <= 0 or steps % rollout:
        raise ValueError(
            f"seed must be nonnegative; steps must be a positive multiple of {rollout}"
        )
    if (
        not np.isfinite(seconds)
        or seconds < 0.01
        or not np.isclose(seconds / 0.01, round(seconds / 0.01), rtol=0, atol=1e-8)
    ):
        raise ValueError("episode seconds must be a positive multiple of .01")
    if len(set(args.conditions)) != len(args.conditions):
        raise ValueError("conditions must be unique")
    before = source_hashes()
    protocol = {
        "schema": SCHEMA,
        "mode": args.mode,
        "conditions": args.conditions,
        "seed": args.seed,
        "steps_per_condition": steps,
        "episode_seconds": seconds,
        "source_sha256": before,
        "source_hash_scope": "Python implementation, pyproject and robot XML/URDF/STL assets.",
        "versions": {
            name: version(name)
            for name in ("numpy", "mujoco", "gymnasium", "stable-baselines3", "torch")
        },
        "ppo_settings": {
            **PPO_SETTINGS,
            **({"n_steps": 32, "batch_size": 32, "n_epochs": 1} if smoke else {}),
        },
        "scope": "Independent basic terrain curriculum; no jump/side-step/GUI skill claim.",
        "difficulty": "Flat, then training terrain amplitudes scaled by 1/3, 2/3, 1.",
        "curriculum_rule": "Budget quartiles using actual transitions, applied only at episode reset.",
        "mixed_rule": "Uniform episode-level difficulty; actual exposure is logged, not assumed equal.",
        "training_command": "Stand .5s, ramp over .5s to .25m/s forward; yaw0, clearance.455m. Smoke ends before the ramp.",
        "probe_command": "Immediate constant .25m/s forward; yaw0, clearance.455m; zero residual.",
        "evaluation": "No development/holdout checkpoint selection; smoke is not a benefit comparison.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", protocol)
    try:
        if args.mode == "probe":
            summaries = probe_levels(args.output, args.seed, seconds)
        else:
            summaries = [
                train_condition(
                    args.output / condition,
                    condition=condition,
                    seed=args.seed,
                    steps=steps,
                    episode_seconds=seconds,
                    smoke=smoke,
                )
                for condition in args.conditions
            ]
        if source_hashes() != before:
            raise RuntimeError("source changed while experiment was running")
        write_json(
            args.output / "summary.json",
            {
                "schema": SCHEMA,
                "mode": args.mode,
                "source_unchanged": True,
                "results": summaries,
                "skill_improvement_established": False,
            },
        )
    except Exception as error:
        write_json(
            args.output / "failure.json", {"type": type(error).__name__, "error": str(error)}
        )
        raise
    print(json.dumps({"output": str(args.output), "mode": args.mode, "cases": len(summaries)}))


if __name__ == "__main__":
    main()
