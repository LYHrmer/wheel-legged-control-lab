"""One bounded 65,536-transition PPO update on the shared-leg box task."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from scripts.d1_rolling_residual_checkpoint import load_final_checkpoint, save_final_checkpoint
from scripts.d1_rolling_residual_env import D1RollingResidualEnv
from scripts.d1_rolling_residual_task import RollingEpisodeSpec, rolling_task_definition
from scripts.d1_single_step_records import jsonable
from scripts.run_d1_heading_study import AfterUpdatePPO, actual_hyperparameters
from scripts.run_d1_locomotion_experiment import PPO_SETTINGS
from wheel_legged_control.d1.ppo_update_audit import parameter_sha256

TRAIN_SEED = 77341
TRAIN_CONTROL = 65536
TRAIN_ROLLOUTS = 512
TRAIN_EPOCHS = 2048
SCENARIOS = tuple(
    RollingEpisodeSpec(speed, onset, onset + 800, True)
    for speed in (.18, .2, .225) for onset in (200, 250)
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(jsonable(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _actor_hash(policy: Any) -> str:
    digest = hashlib.sha256()
    for name, value in (
        ("action_net.weight", policy.action_net.weight),
        ("action_net.bias", policy.action_net.bias),
        ("log_std", policy.log_std),
    ):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def training_protocol(*, library: Path, contract_sha256: str,
                      freeze_path: Path, freeze: dict[str, Any]) -> dict[str, Any]:
    """Record all intended PPO settings before the first model construction."""
    return {
        "schema": "d1-rolling-residual-ppo-protocol-v1",
        "contract_sha256": contract_sha256,
        "library_path": str(Path(library).resolve()),
        "dso_sha256": _sha256(Path(library)),
        "freeze_path": str(Path(freeze_path).resolve()),
        "freeze_sha256": _sha256(Path(freeze_path)),
        "task_definition": rolling_task_definition(),
        "training_scenarios_ordered": [spec.as_dict() for spec in SCENARIOS],
        "training_seed_actual_each_reset": TRAIN_SEED,
        "training_seed_sb3": TRAIN_SEED,
        "training_control_budget": TRAIN_CONTROL,
        "ppo_settings_exact": deepcopy(PPO_SETTINGS),
        "actor_initialization": {"action_net_weight": 0.0,
                                 "action_net_bias": 0.0, "log_std": -1.5},
        "cpu_torch_threads": 1,
        "evaluation_order": [
            {"speed_mps": speed, "terrain": terrain, "actor": actor,
             "onset_tick": 175, "stop_tick": 975, "seed": 77351}
            for speed in (.2, .25) for terrain in ("plane", "box")
            for actor in ("zero", "final_policy")
        ],
        "frozen_source_count": len(freeze["files"]),
        "one_final_checkpoint_no_selection": True,
        "oracle_terrain_provider_default_disclosed": True,
    }


class _TrainingStepWrapper(gym.Wrapper):
    """SB3-facing Gym wrapper; only the root runtime may delegate its step."""

    def __init__(self, env: D1RollingResidualEnv, runtime: Any, stream: Any) -> None:
        super().__init__(env)
        self.runtime = runtime
        self.stream = stream
        self.episode_index = -1
        self.episode_tick = 0
        self.transition_count = 0
        self.episode_reward = 0.0
        self.initial_observation: np.ndarray | None = None
        self.last_observation: np.ndarray | None = None
        self.episode_summaries: list[dict[str, Any]] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if options:
            raise ValueError("training reset options are not part of the frozen protocol")
        self.episode_index += 1
        self.episode_tick = 0
        self.episode_reward = 0.0
        spec = SCENARIOS[self.episode_index % len(SCENARIOS)]
        self.env.configure_next_episode(spec)
        observation, info = self.env.reset(seed=TRAIN_SEED)
        if self.initial_observation is None:
            self.initial_observation = np.asarray(observation).copy()
        self.last_observation = np.asarray(observation).copy()
        self.stream.write(json.dumps({"event": "episode_reset", "episode_index": self.episode_index,
                                      "spec": spec.as_dict(), "requested_sb3_seed": seed,
                                      "actual_env_seed": TRAIN_SEED,
                                      "metadata": jsonable(info["episode_metadata"])},
                                     allow_nan=False) + "\n")
        self.stream.flush()
        return observation, info

    def step(self, action: np.ndarray):
        raw_command = dict(self.env.unwrapped.command_records[-1])
        observation, reward, terminated, truncated, info = self.runtime.control_step(self.env, action)
        self.transition_count += 1
        self.episode_tick += 1
        self.episode_reward += float(reward)
        self.last_observation = np.asarray(observation).copy()
        reward_terms = info["reward_terms"]
        if (not isinstance(reward_terms, dict)
                or not all(np.isfinite(float(value)) for value in reward_terms.values())
                or abs(sum(float(value) for value in reward_terms.values()) - float(reward)) > 1e-10):
            raise RuntimeError("Heading reward decomposition differs from actual reward")
        row = {
            "event": "control_return", "transition": self.transition_count,
            "episode_index": self.episode_index, "episode_tick": self.episode_tick,
            "raw_command": raw_command, "env_received_policy_action": np.asarray(action).copy(),
            "rolling_action": info["rolling_action"],
            "applied_action": np.asarray(info["applied_action"]).copy(),
            "reward": float(reward), "reward_terms": reward_terms,
            "servo_reward_terms": info.get("servo_reward_terms"),
            "reward_metrics": info.get("metrics"),
            "terminated": bool(terminated), "truncated": bool(truncated),
            "terminal_reason": info.get("terminal_reason"),
            "base_qpos": self.env.unwrapped.plant.data.qpos[:7].copy(),
            "base_qvel": self.env.unwrapped.plant.data.qvel[:6].copy(),
        }
        self.stream.write(json.dumps(jsonable(row), allow_nan=False) + "\n")
        if self.transition_count % 128 == 0 or terminated or truncated:
            self.stream.flush()
        if terminated or truncated:
            self.episode_summaries.append({
                "episode_index": self.episode_index,
                "spec": self.env.rolling_spec.as_dict(),
                "completed_controls": self.episode_tick,
                "reward_sum": self.episode_reward,
                "terminated": bool(terminated), "truncated": bool(truncated),
                "terminal_reason": info.get("terminal_reason"),
            })
        return observation, reward, terminated, truncated, info


class _GaussianSampleAudit(BaseCallback):
    """Capture SB3's preclip Gaussian sample and its separate env input."""

    def __init__(self, adapter: _TrainingStepWrapper, stream: Any) -> None:
        super().__init__()
        self.adapter = adapter
        self.stream = stream
        self.count = 0

    def _on_step(self) -> bool:
        raw = np.asarray(self.locals["actions"], dtype=np.float64)
        clipped = np.asarray(self.locals["clipped_actions"], dtype=np.float64)
        infos = self.locals["infos"]
        if raw.shape != (1, 1) or clipped.shape != (1, 1) or len(infos) != 1:
            raise RuntimeError("unexpected SB3 one-environment scalar action batch")
        if not np.isfinite(raw).all() or not np.isfinite(clipped).all():
            raise RuntimeError("nonfinite PPO sampled or clipped action")
        receipt = infos[0]["rolling_action"]
        if (float(clipped[0, 0]) != receipt["policy_scalar"]
                or float(np.clip(raw[0, 0], -1., 1.)) != float(clipped[0, 0])):
            raise RuntimeError("PPO Gaussian sample/clip differs from the actual env input")
        self.count += 1
        if self.count != self.adapter.transition_count:
            raise RuntimeError("PPO callback and physical transition sequence diverged")
        self.stream.write(json.dumps({
            "transition": self.count,
            "preclip_gaussian_action": float(raw[0, 0]),
            "clipped_env_input": float(clipped[0, 0]),
            "actual_policy_action": receipt["policy_scalar"],
            "applied_physical_action": receipt["physical_action"],
        }, allow_nan=False) + "\n")
        if self.count % 128 == 0:
            self.stream.flush()
        return True


def run_training(runtime: Any, output: Path, *, dso_sha256: str,
                 training_protocol_sha256: str) -> tuple[dict[str, Any], D1RollingResidualEnv, Any]:
    """Train exactly once, archive one final checkpoint, reload it without physics."""
    output = Path(output)
    output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    settings = deepcopy(PPO_SETTINGS)
    if (settings["n_steps"], settings["batch_size"], settings["n_epochs"]) != (128, 128, 4):
        raise RuntimeError("frozen PPO rollout/update settings changed")
    env = runtime.construct(lambda: D1RollingResidualEnv(SCENARIOS[0]))
    runtime.start_segment("training", TRAIN_CONTROL)
    runtime.bind(env)
    updates: list[dict[str, Any]] = []
    with (output / "controls.jsonl").open("x", encoding="utf-8") as controls, (
        output / "gaussian_samples.jsonl"
    ).open("x", encoding="utf-8") as samples, (
        output / "completed_updates.jsonl"
    ).open("x", encoding="utf-8") as updates_stream:
        adapter = _TrainingStepWrapper(env, runtime, controls)

        def after_update(learner: Any) -> None:
            if (learner.num_timesteps != adapter.transition_count
                    or learner.completed_train_calls != len(updates) + 1
                    or learner._n_updates != learner.completed_train_calls * settings["n_epochs"]):
                raise RuntimeError("completed PPO update disagrees with physical controls")
            record = {
                "train_call": learner.completed_train_calls,
                "completed_control": adapter.transition_count,
                "optimization_epochs": learner._n_updates,
                "policy_sha256": parameter_sha256(learner.policy),
                "actor_sha256": _actor_hash(learner.policy),
                "optimizer_state_entries": len(learner.policy.optimizer.state),
            }
            updates.append(record)
            updates_stream.write(json.dumps(record, allow_nan=False) + "\n")
            updates_stream.flush()
            if learner.completed_train_calls % 16 == 0:
                print(json.dumps({"event": "completed_ppo_update",
                                  "completed_controls": adapter.transition_count,
                                  "train_calls": learner.completed_train_calls,
                                  "optimization_epochs": learner._n_updates}), flush=True)

        model = AfterUpdatePPO("MlpPolicy", adapter, seed=TRAIN_SEED, device="cpu",
                               verbose=0, after_update=after_update, **settings)
        if model.num_timesteps or model._n_updates or model.policy.optimizer.state:
            raise RuntimeError("PPO model must be fresh before its sole learn call")
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
            model.policy.log_std.fill_(-1.5)
        initial_hash = parameter_sha256(model.policy)
        initial_actor_hash = _actor_hash(model.policy)
        _write_json(output / "initial_training.json", {
            "seed": TRAIN_SEED, "training_control_budget": TRAIN_CONTROL,
            "ppo_settings": settings, "actual_hyperparameters": actual_hyperparameters(model),
            "initial_parameter_sha256": initial_hash,
            "initial_actor_sha256": initial_actor_hash,
            "optimizer_state_entries": len(model.policy.optimizer.state),
            "actor_mean_weight_bias_zero": True, "actor_log_std": -1.5,
        })
        audit = _GaussianSampleAudit(adapter, samples)
        model.learn(total_timesteps=TRAIN_CONTROL, callback=audit)
        controls.flush()
        samples.flush()
        updates_stream.flush()
    final_hash = parameter_sha256(model.policy)
    final_actor_hash = _actor_hash(model.policy)
    if (adapter.transition_count != TRAIN_CONTROL or audit.count != TRAIN_CONTROL
            or model.num_timesteps != TRAIN_CONTROL
            or model.completed_train_calls != TRAIN_ROLLOUTS
            or model._n_updates != TRAIN_EPOCHS or len(updates) != TRAIN_ROLLOUTS
            or len(model.policy.optimizer.state) == 0 or initial_hash == final_hash
            or not all(torch.isfinite(value).all() for value in model.policy.parameters())):
        raise RuntimeError("frozen PPO completion, optimizer or finite-parameter gate failed")
    if adapter.initial_observation is None or adapter.last_observation is None:
        raise RuntimeError("PPO checkpoint observations are unavailable")
    probes = np.stack((adapter.initial_observation, adapter.last_observation))
    checkpoint_dir = output / "final_checkpoint"
    sidecar = save_final_checkpoint(
        model, env, checkpoint_dir, initial_parameter_sha256=initial_hash,
        final_parameter_sha256=final_hash, probe_observations=probes,
        dso_sha256=dso_sha256, training_protocol_sha256=training_protocol_sha256,
    )
    native_before_reload = runtime.ledger_receipt()["native_returned"]
    loaded = load_final_checkpoint(
        checkpoint_dir / "model.zip", checkpoint_dir / "model.metadata.json", env,
        expected_dso_sha256=dso_sha256,
        expected_training_protocol_sha256=training_protocol_sha256,
    )
    if runtime.ledger_receipt()["native_returned"] != native_before_reload:
        raise RuntimeError("checkpoint reload consumed a physical step")
    deterministic, _ = loaded.predict(probes, deterministic=True)
    deterministic = np.asarray(deterministic, dtype=np.float64)
    active = np.array([0.25 * float(np.clip(value, -1., 1.))
                       for value in deterministic[:, 0]])
    report = {
        "passed": True, "num_timesteps": model.num_timesteps,
        "physical_control_transitions": adapter.transition_count,
        "completed_train_calls": model.completed_train_calls,
        "optimization_epochs": model._n_updates,
        "completed_update_records": len(updates),
        "initial_parameter_sha256": initial_hash,
        "final_parameter_sha256": final_hash,
        "initial_actor_sha256": initial_actor_hash,
        "final_actor_sha256": final_actor_hash,
        "actor_parameters_changed": initial_actor_hash != final_actor_hash,
        "optimizer_state_entries": len(model.policy.optimizer.state),
        "episodes_completed": adapter.episode_summaries,
        "active_episode_index": adapter.episode_index,
        "active_episode_completed_controls": adapter.episode_tick,
        "actual_hyperparameters": actual_hyperparameters(model),
        "model_sha256": sidecar["model_sha256"],
        "checkpoint_metadata_sha256": _sha256(checkpoint_dir / "model.metadata.json"),
        "checkpoint_reload_zero_physics": True,
        "deterministic_probe_policy_scalar": deterministic[:, 0].tolist(),
        "deterministic_probe_counterfactual_leg_residual_normalized_if_forward_gate_active": active.tolist(),
        "deterministic_probe_counterfactual_leg_residual_m_if_forward_gate_active": (active * .04).tolist(),
        "probe_residuals_are_not_actual_applied_actions": True,
        "preclip_gaussian_and_clipped_inputs_logged_separately": True,
    }
    _write_json(output / "training_receipt.json", report)
    runtime.unbind()
    return report, env, loaded
