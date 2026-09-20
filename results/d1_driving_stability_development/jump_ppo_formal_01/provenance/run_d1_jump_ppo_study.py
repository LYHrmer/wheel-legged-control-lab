"""One fresh, fixed-budget PPO study for the native-plane stationary hop task."""
from __future__ import annotations

import argparse
import gzip
import json
import time
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from scripts.d1_jump_ppo_env import D1JumpPPOEnv, load_jump_policy
from scripts.d1_jump_ppo_task import JumpEpisodeSpec
from scripts.probe_d1_jump_readiness import check_frozen, check_hashes, tagged, write_json
from scripts.run_d1_heading_study import AfterUpdatePPO, actual_hyperparameters
from scripts.run_d1_locomotion_experiment import PPO_SETTINGS
from wheel_legged_control.d1.locomotion_checkpoint import write_checkpoint_metadata
from wheel_legged_control.d1.ppo_update_audit import parameter_sha256


def episode_spec(mode, episode_index, total_transitions):
    """Select at reset from actual completed transitions, before VecEnv auto-reset returns."""
    if mode == "smoke":
        return JumpEpisodeSpec(200, .005) if episode_index == 0 else JumpEpisodeSpec(None, 0.)
    if mode != "formal":
        raise ValueError("mode must be smoke or formal")
    if episode_index % 5 == 4:
        return JumpEpisodeSpec(None, 0.)
    jump_index = episode_index - episode_index // 5
    goal = .005 if total_transitions < 32768 else .01 if total_transitions < 65536 else .02
    return JumpEpisodeSpec((150, 200, 250)[jump_index % 3], goal)


class NativeCounter:
    """Count actual calls and failed partial intervals without retaining training native traces."""
    def __init__(self, plant, limit):
        self.plant, self.limit = plant, limit
        self.attempted = self.returned = self.failed = 0
        self.actual_seconds = 0.
        self.original = None

    def __enter__(self):
        self.original = mujoco.mj_step
        mujoco.mj_step = self.call
        return self

    def __exit__(self, *args):
        mujoco.mj_step = self.original

    def call(self, model, data, *args, **kwargs):
        if model is not self.plant.model or data is not self.plant.data:
            raise RuntimeError("undeclared physics outside the training plant")
        if self.attempted >= self.limit:
            raise RuntimeError("native physics budget exhausted")
        before = float(data.time)
        self.attempted += 1
        try:
            result = self.original(model, data, *args, **kwargs)
        except BaseException:
            self.failed += 1
            self.actual_seconds += float(data.time) - before
            raise
        self.returned += 1
        dt = float(data.time) - before
        self.actual_seconds += dt
        if abs(dt - .002) > 1e-10:
            raise RuntimeError("native clock differs from the fixed two-millisecond timestep")
        return result

    def receipt(self):
        return {"attempted_native_calls": self.attempted, "returned_native_calls": self.returned,
                "failed_native_calls": self.failed, "actual_integrated_seconds": self.actual_seconds,
                "maximum_native_calls": self.limit, "full_native_training_archive": False}


class CurriculumLedger(gym.Wrapper):
    """Record transitions before SB3 can reset the just-finished episode."""
    def __init__(self, env, mode, folder, counter):
        super().__init__(env)
        self.mode, self.folder, self.counter = mode, folder, counter
        self.total_transitions = 0
        self.episode_index = -1
        self.episode_transitions = 0
        self.episode_reward = 0.
        self.episode_records = []
        self.last_observation = self.initial_observation = None
        self.last_info = None
        self.episode_finished = True
        self.files = ExitStack()
        self.trace = self.files.enter_context(gzip.open(folder / "training_trace.jsonl.gz", "xt", encoding="utf-8"))  # noqa: SIM115
        self.episodes = self.files.enter_context(gzip.open(folder / "episodes.jsonl.gz", "xt", encoding="utf-8"))  # noqa: SIM115
        self.boundaries = folder / "boundaries"
        self.boundaries.mkdir()

    def snapshot(self):
        return {"total_transitions": self.total_transitions, "episode_index": self.episode_index,
                "episode_transitions": self.episode_transitions,
                "episode_reward": self.episode_reward, "episode_finished": self.episode_finished,
                "spec": self.env.jump_episode_spec.as_dict(),
                "qpos": self.env.plant.data.qpos.copy(), "qvel": self.env.plant.data.qvel.copy(),
                "qacc_warmstart": self.env.plant.data.qacc_warmstart.copy(),
                "time_s": float(self.env.plant.data.time), "observation": self.last_observation,
                "progress": self.env.jump_progress.snapshot.as_dict(),
                "last_info": self.last_info}

    def reset(self, *, seed=None, options=None):
        if not self.episode_finished:
            raise RuntimeError("unexpected reset of an unfinished training episode")
        self.episode_index += 1
        spec = episode_spec(self.mode, self.episode_index, self.total_transitions)
        self.env.configure_next_episode(spec)
        actual_seed = (629990000 if self.mode == "smoke" else 620010000) + self.episode_index
        self.episode_transitions, self.episode_reward = 0, 0.
        obs, info = self.env.reset(seed=actual_seed, options=options)
        self.episode_finished = False
        self.last_observation = np.array(obs, copy=True)
        self.last_info = None
        if self.initial_observation is None:
            self.initial_observation = self.last_observation.copy()
        write_json(self.boundaries / f"episode_{self.episode_index:05d}_reset.json", tagged({
            **self.snapshot(), "actual_seed": actual_seed, "incoming_seed": seed,
            "metadata": info["episode_metadata"]}))
        return obs, info

    def step(self, action):
        before = self.env.heading_decision
        native_before = self.counter.returned
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_transitions += 1
        self.episode_transitions += 1
        self.episode_reward += float(reward)
        self.last_observation = np.array(obs, copy=True)
        self.last_info = info
        if self.counter.returned - native_before != 5:
            raise RuntimeError("completed control interval must own exactly five native calls")
        if before.tick != self.episode_transitions - 1:
            raise RuntimeError("training episode decision counter changed")
        control = self.env._controller.controller.last_result
        row = {"total_transition": self.total_transitions, "episode_index": self.episode_index,
               "executed_tick": before.tick, "action": action, "reward": float(reward),
               "terminated": bool(terminated), "truncated": bool(truncated),
               "terminal_reason": info["terminal_reason"], "raw_command": info["heading_task"]["user_command_before"],
               "servo_command": info["heading_task"]["servo_command_before"],
               "actual_dt_s": info["heading_task"]["actual_dt_s"],
               "native_calls": self.counter.returned - native_before,
               "reward_terms": info["reward_terms"], "parent_reward_terms": info["parent_reward_terms"],
               "geometry": info["jump_geometry"], "endpoint": info["jump_endpoint_metrics"],
               "progress": info["jump_progress"], "progress_event": info["jump_progress_event"],
               "torque_protected_axes": int(np.count_nonzero(control.torque_limited)),
               "rate_limited_axes": int(np.count_nonzero(control.joint_target_rate_limited)),
               "wheel_targets_rad_s": control.wheel_speed_target_rad_s,
               "action_boundary_axes": int(np.count_nonzero(np.abs(action) >= 1.)),
               "pi_memory_nm": control.memory_after.wheel_integral_nm,
               "peak_abs_requested_torque_nm": float(np.max(np.abs(control.requested_torque_nm))),
               "peak_abs_protected_torque_nm": float(np.max(np.abs(control.torque_nm)))}
        self.trace.write(json.dumps(tagged(row), allow_nan=False, separators=(",", ":")) + "\n")
        if terminated or truncated:
            self.episode_finished = True
            record = {"episode_index": self.episode_index, "spec": self.env.jump_episode_spec.as_dict(),
                      "transitions": self.episode_transitions, "return": self.episode_reward,
                      "end_total_transitions": self.total_transitions,
                      "terminated": bool(terminated), "truncated": bool(truncated),
                      "terminal_reason": info["terminal_reason"], "progress": info["jump_progress"]}
            self.episode_records.append(record)
            self.episodes.write(json.dumps(tagged(record), allow_nan=False) + "\n")
            write_json(self.boundaries / f"episode_{self.episode_index:05d}_end.json", tagged(self.snapshot()))
            self.trace.flush()
            self.episodes.flush()
        return obs, reward, terminated, truncated, info

    def close(self):
        self.files.close()
        self.env.close()


class InitialCallback(BaseCallback):
    def __init__(self, ledger, folder):
        super().__init__()
        self.ledger, self.folder = ledger, folder

    def _on_training_start(self):
        obs = self.ledger.initial_observation
        if obs.shape != (95,) or self.model.num_timesteps != 0:
            raise RuntimeError("fresh training must start with a 95-dimensional observation and zero steps")
        write_json(self.folder / "first_episode_metadata.json", self.ledger.unwrapped.episode_metadata)

    def _on_step(self):
        return True


def train_once(folder, mode):
    folder.mkdir()
    torch.set_num_threads(1)
    steps, seed, epochs = (640, 62999, 1) if mode == "smoke" else (131072, 62001, 4)
    env = D1JumpPPOEnv()
    counter = NativeCounter(env.plant, steps * 5)
    ledger = CurriculumLedger(env, mode, folder, counter)
    checkpoints = []
    started = time.monotonic()
    error, model = None, None
    update_file = (folder / "updates.jsonl").open("x")

    def after_update(learner):
        count = int(learner.num_timesteps)
        if (count != ledger.total_transitions or learner.completed_train_calls != count // 128
                or learner._n_updates != learner.completed_train_calls * epochs):
            raise RuntimeError("completed PPO update counters do not conserve work")
        if any(not torch.isfinite(value).all() for value in learner.policy.state_dict().values()):
            raise RuntimeError("PPO parameters became nonfinite")
        stats = {"transitions": count, "train_calls": learner.completed_train_calls,
                 "optimization_epochs": learner._n_updates,
                 "elapsed_s": time.monotonic()-started,
                 "logger": dict(learner.logger.name_to_value)}
        update_file.write(json.dumps(tagged(stats), allow_nan=False) + "\n")
        update_file.flush()
        if count % 4096 == 0 or mode == "smoke":
            print(json.dumps({"mode": mode, "steps": count, "episodes": len(ledger.episode_records),
                              "elapsed_s": stats["elapsed_s"]}), flush=True)
        if mode == "formal" and count in (32768, 65536, 131072):
            target = folder / f"checkpoint_{count}"
            target.mkdir()
            observations = np.stack([ledger.initial_observation, ledger.last_observation])
            actions = learner.predict(observations, deterministic=True)[0]
            record = {"steps": count, "train_calls": learner.completed_train_calls,
                      "optimization_epochs": learner._n_updates,
                      "parameter_sha256": parameter_sha256(learner.policy), "directory": target.name}
            learner.save(target / "model.zip")
            write_checkpoint_metadata(target / "model.zip", env, target / "model.metadata.json", extra=record)
            np.savez_compressed(target / "reload_probe.npz", observations=observations, actions=actions)
            write_json(target / "snapshot.json", tagged(ledger.snapshot()))
            write_json(target / "receipt.json", record)
            checkpoints.append(record)

    try:
        with counter:
            settings = deepcopy(PPO_SETTINGS)
            settings["n_epochs"] = epochs
            model = AfterUpdatePPO("MlpPolicy", ledger, seed=seed, device="cpu", verbose=0,
                                   after_update=after_update, **settings)
            if model.num_timesteps or model._n_updates or model.policy.optimizer.state:
                raise RuntimeError("training must have fresh parameters, optimizer and counters")
            write_json(folder / "initial_training.json", tagged({
                "mode": mode, "seed": seed, "budget": steps,
                "initial_parameter_sha256": parameter_sha256(model.policy),
                "hyperparameters": actual_hyperparameters(model),
                "optimizer_state_entries": len(model.policy.optimizer.state)}))
            model.learn(total_timesteps=steps, callback=InitialCallback(ledger, folder))
            if model.num_timesteps != steps or counter.returned != steps * 5:
                raise RuntimeError("training did not complete the declared budget exactly")
            for record in checkpoints:
                target = folder / record["directory"]
                before_calls = counter.attempted
                loaded = load_jump_policy(target / "model.zip", target / "model.metadata.json", env)
                probe = np.load(target / "reload_probe.npz")
                actions = loaded.predict(probe["observations"], deterministic=True)[0]
                if (not np.array_equal(actions, probe["actions"])
                        or parameter_sha256(loaded.policy) != record["parameter_sha256"]
                        or loaded.num_timesteps != record["steps"]
                        or loaded._n_updates != record["optimization_epochs"]
                        or loaded.completed_train_calls != record["train_calls"]
                        or counter.attempted != before_calls):
                    raise RuntimeError("checkpoint failed the zero-step reload comparison")
                write_json(target / "reload_verification.json", {"passed": True, "additional_physics": 0})
    except BaseException as exc:  # noqa: BLE001 -- no retry; preserve all consumed training work
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        try:
            write_json(folder / "final_snapshot.json", tagged(ledger.snapshot()))
        except BaseException as exc:  # noqa: BLE001 -- retain primary failure and physical counters
            error = error or {"type": type(exc).__name__, "message": str(exc), "phase": "final_snapshot"}
        finally:
            try:
                ledger.close()
            finally:
                update_file.close()
    observed = sum(row["transitions"] for row in ledger.episode_records)
    if not ledger.episode_finished:
        observed += ledger.episode_transitions
    receipt = {"mode": mode, "error": error, "successful_training_transitions": ledger.total_transitions,
               "episode_transition_sum": observed, "native": counter.receipt(),
               "ppo_timesteps": None if model is None else model.num_timesteps,
               "train_calls": None if model is None else model.completed_train_calls,
               "optimization_epochs": None if model is None else model._n_updates,
               "checkpoint_records": checkpoints, "completed_episodes": len(ledger.episode_records),
               "elapsed_s": time.monotonic()-started, "smoke_weights_discarded": mode == "smoke",
               "passed": error is None and ledger.total_transitions == steps and observed == steps}
    write_json(folder / "receipt.json", tagged(receipt))
    print(json.dumps(tagged(receipt), indent=2), flush=True)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    args = parser.parse_args(argv)
    preflight = json.loads(args.preflight.read_text())
    if not all(preflight.get(key) is True for key in ("passed", "root_execution_authorized", "readiness_records_valid")):
        raise ValueError("a reviewed execution preflight and valid readiness records are required")
    inputs = preflight.get("input_sha256", {})
    if str(Path(__file__).resolve()) not in inputs or not check_hashes(inputs) or not check_frozen():
        raise ValueError("source or frozen77 identity changed")
    readiness = Path(preflight["readiness_directory"])
    if json.loads((readiness / "receipt.json").read_text()).get("records_valid") is not True:
        raise ValueError("readiness instrumentation did not qualify")
    if args.mode == "formal":
        smoke = json.loads(Path(preflight["smoke_receipt"]).read_text())
        if smoke.get("passed") is not True or smoke.get("successful_training_transitions") != 640:
            raise ValueError("formal training requires the completed fixed smoke")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    receipt = train_once(args.output, args.mode)
    identity = {"source_unchanged": check_hashes(inputs), "frozen77_unchanged": check_frozen()}
    identity["passed"] = all(identity.values())
    write_json(args.output / "source_after.json", identity)
    return 0 if receipt["passed"] and identity["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
