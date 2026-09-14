"""Train fresh heading-hold PPO models with checkpoints after completed updates.

Formal training is one uninterrupted 65,536-transition learn call per seed.
The user command only trains straight forward motion and zero user yaw rate.
This runner does not perform development/holdout evaluation during learning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.d1_heading_curriculum import CURRICULUM_STEPS, D1HeadingCurriculumEnv
from scripts.run_d1_course_curriculum import forward_command, summarize_exposure
from scripts.run_d1_course_curriculum import source_hashes as curriculum_source_hashes
from scripts.run_d1_locomotion_experiment import PPO_SETTINGS
from wheel_legged_control.d1.locomotion_checkpoint import write_checkpoint_metadata
from wheel_legged_control.d1.ppo_update_audit import parameter_sha256

SCHEMA = "d1-heading-hold-training-study-v1"
TRAIN_SEEDS = (49001, 49002, 49003)
TRAIN_STEPS = 65536
CHECKPOINT_STEPS = (16384, 65536)
EPISODE_SECONDS = 32.0
TASK_SCHEMAS = {
    "task_schema": "d1-heading-reference-task-v1",
    "observation_schema": "d1-proprio-servo82-heading85-v1",
    "reward_schema": "d1-heading-goal-servo-rate-v1",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_manifest(directory):
    write_json(directory / "manifest.json", {
        str(p.relative_to(directory)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p != directory / "manifest.json"
    })


def source_hashes():
    hashes = curriculum_source_hashes()
    for name in ("d1_heading_reference.py", "d1_heading_tracking_env.py",
                 "d1_heading_curriculum.py", "run_d1_heading_study.py"):
        path = ROOT / "scripts" / name
        hashes[str(path.relative_to(ROOT))] = sha256(path)
    return hashes


class AfterUpdatePPO(PPO):
    """Stock PPO with an observer called only after the full train call returns."""

    def __init__(self, *args, after_update=None, **kwargs):
        self._after_update = after_update
        self.completed_train_calls = 0
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), "_after_update"]

    def train(self):
        before_steps, before_updates = self.num_timesteps, self._n_updates
        super().train()
        if self.num_timesteps != before_steps or self._n_updates <= before_updates:
            raise RuntimeError("PPO update did not preserve steps and advance optimization epochs")
        self.completed_train_calls += 1
        if self._after_update is not None:
            self._after_update(self)


class RecordInitialEpisode(BaseCallback):
    """Record the first reset performed by SB3 without another reset or RNG draw."""

    def __init__(self, env, output):
        super().__init__()
        self.training_env_reference, self.output = env, output

    def _on_training_start(self):
        env = self.training_env_reference
        actual = env.unwrapped.episode_metadata
        if any(actual.get(key) != value for key, value in TASK_SCHEMAS.items()):
            raise ValueError("training environment does not implement the declared heading task")
        if env.observation_space.shape != (85,) or env.action_space.shape != (8,):
            raise ValueError("heading training requires the declared 85-observation/eight-action spaces")
        if "heading_task_config" not in actual:
            raise ValueError("actual heading configuration is absent from episode metadata")
        write_json(self.output / "first_episode_metadata.json", actual)

    def _on_step(self):
        return True


def exposure_snapshot(env):
    """Read the live episode without closing, resetting or finalizing it."""
    completed = deepcopy(env.episode_records)
    active = None
    if env.episode_active and env._episode_transitions:
        active = {
            "episode_index": env._episode_counter,
            "condition": env.condition,
            "level": env.current_level,
            "terrain_index": env.current_terrain_index,
            "episode_seed": env.current_episode_seed,
            "terrain_parameters": asdict(env.terrain_config),
            "transitions": env._episode_transitions,
            "start_total_transitions": env._episode_start_transitions,
            "end_total_transitions": env.total_transitions,
            "reward_sum": env._episode_reward_sum,
            "completed": False,
            "terminal_reason": "active_checkpoint_snapshot",
            "last_terrain_exposure": deepcopy(env._last_exposure),
        }
    records = completed + ([] if active is None else [active])
    observed = sum(record["transitions"] for record in records)
    if observed != env.total_transitions:
        raise RuntimeError("live episode snapshot does not conserve training transitions")
    return {
        "actual_transitions": observed, "completed_episodes": completed,
        "active_episode": active, **summarize_exposure(records),
    }


def actual_hyperparameters(model):
    fields = ("n_steps", "batch_size", "n_epochs", "learning_rate", "gamma", "gae_lambda",
              "normalize_advantage", "ent_coef", "vf_coef", "max_grad_norm", "target_kl",
              "use_sde", "sde_sample_freq")
    return {**{name: getattr(model, name) for name in fields},
            "clip_range": float(model.clip_range(1.0)), "num_envs": model.n_envs,
            "policy_kwargs": model.policy_kwargs,
            "optimizer_class": type(model.policy.optimizer).__name__,
            "optimizer_defaults": model.policy.optimizer.defaults,
            "activation_function": model.policy.activation_fn.__name__, "device": str(model.device)}


def verify_saved_checkpoints(model, records, states, output):
    """Run after learn ends; an independent environment cannot alter curriculum exposure."""
    from scripts.d1_heading_tracking_env import D1HeadingTrackingEnv, load_heading_policy

    for record in records:
        budget = record["budget"]
        folder = output / record["directory"]
        check_env = D1HeadingTrackingEnv(episode_seconds=0.02, command_source=forward_command)
        try:
            observation, _ = check_env.reset(seed=99173)
            loaded = load_heading_policy(folder / "model.zip", folder / "model.metadata.json", check_env)
            if loaded.num_timesteps != budget or loaded._n_updates != record["optimization_epochs"]:
                raise RuntimeError("loaded checkpoint lost its post-update counters")
            if parameter_sha256(loaded.policy) != record["parameter_sha256"]:
                raise RuntimeError("loaded checkpoint parameters differ from the completed update")
            # Reconstruct the exact in-memory policy captured at that completed update.
            reference = model.policy_class(**model.policy._get_constructor_parameters()).to(model.device)
            reference.load_state_dict(states[budget])
            expected = reference.predict(observation, deterministic=True)[0]
            actual = loaded.predict(observation, deterministic=True)[0]
            if not np.array_equal(expected, actual):
                raise RuntimeError("saved policy changed its deterministic action")
            new_obs, reward, terminated, truncated, _ = check_env.step(actual)
            if new_obs.shape != (85,) or not np.isfinite(new_obs).all() or not np.isfinite(reward):
                raise RuntimeError("reloaded policy produced an invalid heading-task step")
            write_json(folder / "reload_check.json", {
                "after_training_finished": True, "independent_environment": True,
                "parameter_hash_exact": True, "deterministic_action_exact": True,
                "additional_physical_steps": 1, "finite_85_observation_and_reward": True,
                "terminated": bool(terminated), "truncated": bool(truncated),
            })
        finally:
            check_env.close()
        write_manifest(folder)


def train_seed(output, *, seed, steps, budgets, smoke):
    import torch
    from stable_baselines3.common.logger import configure

    output.mkdir()
    (output / "checkpoints").mkdir()
    settings = deepcopy(PPO_SETTINGS)
    if smoke:
        settings.update(n_steps=32, batch_size=32, n_epochs=1)
    records, parameter_states = [], {}
    env = None
    started = time.monotonic()
    try:
        env = D1HeadingCurriculumEnv(total_steps=steps, seed=seed,
                                    episode_seconds=EPISODE_SECONDS, command_source=forward_command)

        def after_update(learner):
            if learner.num_timesteps not in budgets:
                return
            budget = int(learner.num_timesteps)
            if env.total_transitions != budget:
                raise RuntimeError("physical and PPO transition counts disagree at checkpoint")
            folder = output / "checkpoints" / f"step{budget}"
            folder.mkdir()
            parameter_states[budget] = {name: tensor.detach().cpu().clone()
                                       for name, tensor in learner.policy.state_dict().items()}
            state_hash = parameter_sha256(learner.policy)
            learner.save(folder / "model.zip")
            metadata = write_checkpoint_metadata(folder / "model.zip", env.unwrapped,
                                                folder / "model.metadata.json", extra={
                "experiment_schema": SCHEMA, "seed": seed, "checkpoint_budget": budget,
                "actual_transitions": env.total_transitions,
                "training_budget_steps": steps, "curriculum_steps": CURRICULUM_STEPS,
                "pipeline_smoke_only": smoke, "user_command_training": "straight_forward_yaw_zero",
                "optimization_epochs": int(learner._n_updates),
                "completed_train_calls": learner.completed_train_calls,
                "saved_after_completed_update": True, "parameter_sha256": state_hash,
            })
            record = {
                "budget": budget, "actual_transitions": env.total_transitions,
                "ppo_num_timesteps": int(learner.num_timesteps),
                "optimization_epochs": int(learner._n_updates),
                "completed_train_calls": learner.completed_train_calls,
                "saved_after_completed_update": True, "parameter_sha256": state_hash,
                "model_sha256": metadata["model_sha256"],
                "metadata_sha256": sha256(folder / "model.metadata.json"),
                "directory": str(folder.relative_to(output)),
                "exposure": exposure_snapshot(env),
            }
            write_json(folder / "checkpoint.json", record)
            records.append(record)
            print(json.dumps({"event": "checkpoint_after_update", "seed": seed,
                              "budget": budget, "optimization_epochs": int(learner._n_updates),
                              "model": str(folder / "model.zip")}), flush=True)

        model = AfterUpdatePPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0,
                               after_update=after_update, **settings)
        hp = actual_hyperparameters(model)
        write_json(output / "initial_policy.json", {
            "seed": seed, "num_timesteps": int(model.num_timesteps),
            "parameter_sha256": parameter_sha256(model.policy), "ppo_actual_hyperparameters": hp,
            "observation_dimension": 85, "action_dimension": 8,
        })
        model.set_logger(configure(str(output), ["csv"]))
        model.learn(total_timesteps=steps, callback=RecordInitialEpisode(env, output))
        model.logger.dump(step=model.num_timesteps)
        if model.num_timesteps != steps or env.total_transitions != steps:
            raise RuntimeError("PPO and physical transition budgets were not fulfilled exactly")
        if [record["budget"] for record in records] != list(budgets):
            raise RuntimeError("not all declared post-update checkpoints were saved")
        if not all(torch.isfinite(value).all() for value in model.policy.parameters()):
            raise RuntimeError("training finished with nonfinite parameters")
        verify_saved_checkpoints(model, records, parameter_states, output)
        result = {
            "seed": seed, "pipeline_smoke_only": smoke,
            "single_uninterrupted_learn_call": True, "training_budget_steps": steps,
            "curriculum_steps": CURRICULUM_STEPS, "actual_transitions": env.total_transitions,
            "ppo_num_timesteps": int(model.num_timesteps),
            "optimization_epochs": int(model._n_updates),
            "completed_train_calls": model.completed_train_calls,
            "ppo_actual_hyperparameters": hp, "checkpoints": records,
            "additional_reload_check_physical_steps": len(records),
            "final_parameter_sha256": parameter_sha256(model.policy),
            "wall_seconds": time.monotonic() - started,
        }
    except BaseException as error:
        write_json(output / "failure.json", {"type": type(error).__name__, "message": str(error),
                   "actual_transitions": 0 if env is None else env.total_transitions,
                   "saved_checkpoint_budgets": [record["budget"] for record in records]})
        raise
    finally:
        if env is not None:
            env.close()
            write_json(output / "episodes.json", env.episode_records)
    if sum(record["transitions"] for record in env.episode_records) != steps:
        raise RuntimeError("final episode records do not conserve physical transitions")
    result.update(summarize_exposure(env.episode_records))
    write_json(output / "summary.json", result)
    write_manifest(output)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke", "train"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--steps", type=int, help="Smoke: positive multiple of 32; train: fixed 65536")
    parser.add_argument("--evaluation-protocol", type=Path,
                        help="Optional external frozen development/holdout identity; never evaluated here")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    smoke = args.mode == "smoke"
    seeds = args.seeds if args.seeds is not None else ([49999] if smoke else list(TRAIN_SEEDS))
    steps = args.steps if args.steps is not None else (256 if smoke else TRAIN_STEPS)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be unique nonnegative integers")
    if not smoke and (steps != TRAIN_STEPS or not set(seeds) <= set(TRAIN_SEEDS)):
        raise ValueError("formal training uses only declared seeds and a fixed 65536-step budget")
    if smoke and (steps <= 0 or steps % 32):
        raise ValueError("smoke steps must be a positive multiple of 32")
    budgets = tuple(sorted({32, steps})) if smoke else CHECKPOINT_STEPS
    output = args.output.absolute()
    if output.exists() or any(path.is_symlink() for path in (output, *output.parents)):
        raise FileExistsError("output must be new without symlink ancestors")
    import torch
    torch.set_num_threads(1)
    before = source_hashes()
    evaluation = None
    if args.evaluation_protocol is not None:
        evaluation = {"path": str(args.evaluation_protocol.absolute()),
                      "sha256": sha256(args.evaluation_protocol),
                      "protocol": json.loads(args.evaluation_protocol.read_text())}
    settings = deepcopy(PPO_SETTINGS)
    if smoke:
        settings.update(n_steps=32, batch_size=32, n_epochs=1)
    output.mkdir(parents=True)
    write_json(output / "protocol.json", {
        "schema": SCHEMA, "mode": args.mode, "pipeline_smoke_only": smoke,
        "seeds": seeds, "training_steps_per_seed": steps,
        "training_total_steps": steps * len(seeds), "checkpoint_steps": list(budgets),
        "checkpoint_timing": "after the full PPO.train returns; never on_rollout_end",
        "curriculum_steps": CURRICULUM_STEPS, "training_episode_seconds": EPISODE_SECONDS,
        "curriculum_rule": "At reset min(3,4*actual_transitions//16384); active episodes keep their level",
        "policy": "stock MlpPolicy", "ppo_settings": settings,
        "task_schemas": TASK_SCHEMAS, "observation_dimension": 85, "action_dimension": 8,
        "heading_reward_weight": 0.5, "heading_sigma_rad": float(np.deg2rad(5.0)),
        "heading_outer_loop": {"kp": 2.0, "kd": 0.4, "limit_rps": 1.0},
        "user_command_training": "forward_command: stand .5s, ramp .5s to .25m/s, yaw0, clearance.455m",
        "claim_scope": "Fresh straight heading-hold training; no nonzero user-yaw training samples",
        "evaluation_performed": False, "external_evaluation_identity": evaluation,
        "evaluation_limits": "Task/reward differs from old82; compare common goal metrics, not cross-task returns",
        "source_sha256": before, "torch_threads": torch.get_num_threads(),
        "versions": {name: version(name) for name in ("numpy", "mujoco", "gymnasium", "stable-baselines3", "torch")},
    })
    results = []
    try:
        for seed in seeds:
            results.append(train_seed(output / f"seed{seed}", seed=seed, steps=steps,
                                      budgets=budgets, smoke=smoke))
        unchanged = source_hashes() == before and (
            evaluation is None or sha256(args.evaluation_protocol) == evaluation["sha256"])
        if not unchanged:
            raise RuntimeError("source or frozen evaluation identity changed during training")
        write_json(output / "summary.json", {"schema": SCHEMA, "all_seeds_completed": True,
                   "source_and_evaluation_identity_unchanged": True, "results": results})
    except BaseException as error:
        write_json(output / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        write_manifest(output)
    print(json.dumps({"output": str(output), "mode": args.mode, "seeds": seeds,
                      "all_seeds_completed": True}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
