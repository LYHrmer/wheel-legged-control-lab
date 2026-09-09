"""Read-only PPO update samples, with collection/action likelihood semantics.

Core capture/evaluation structure drafted with Claude Opus, then adapted to the
installed SB3 implementation. The audit copies the first 128 env-major samples;
this is a diagnostic trajectory fragment, not an unbiased rollout statistic.
Only standard MLP diagonal-Gaussian policies are supported by this experiment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.distributions import DiagGaussianDistribution


def parameter_sha256(policy):
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        array = value.detach().cpu().numpy()
        digest.update(name.encode())
        digest.update(str((array.dtype, array.shape)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _head(array, count):
    return (
        array.swapaxes(0, 1)
        .reshape(array.shape[0] * array.shape[1], *array.shape[2:])[:count]
        .copy()
    )


class AuditedRolloutBuffer(RolloutBuffer):
    """Retain the exact GAE boundary arguments, without changing recursion."""

    def compute_returns_and_advantage(self, last_values, dones):
        self.audit_last_values = last_values.detach().cpu().numpy().reshape(-1).copy()
        self.audit_last_dones = np.asarray(dones).copy()
        super().compute_returns_and_advantage(last_values, dones)


class AuditedPPO(PPO):
    """One record per PPO train call (which contains multiple optimizer epochs)."""

    def __init__(self, *args, audit_directory=None, **kwargs):
        if audit_directory is not None:
            if kwargs.get("rollout_buffer_class") not in (None, AuditedRolloutBuffer):
                raise ValueError("update auditing requires the standard audited rollout buffer")
            kwargs["rollout_buffer_class"] = AuditedRolloutBuffer
        super().__init__(*args, **kwargs)
        self._audit_directory = None if audit_directory is None else Path(audit_directory)
        self._audit_index = 0
        if self._audit_directory is not None:
            self._check_supported()
            if self._audit_directory.exists() and any(self._audit_directory.iterdir()):
                raise ValueError("audit directory must be new or empty")
            self._audit_directory.mkdir(parents=True, exist_ok=True)

    def _check_supported(self):
        if (
            self.use_sde
            or not isinstance(self.action_space, spaces.Box)
            or not isinstance(self.observation_space, spaces.Box)
        ):
            raise ValueError("audit requires Box observation/actions without gSDE")
        if (
            type(self.policy.action_dist) is not DiagGaussianDistribution
            or self.policy.squash_output
        ):
            raise ValueError("audit currently supports a plain diagonal Gaussian")
        # Switching train/eval must not change sampling or running statistics.
        forbidden = (torch.nn.modules.batchnorm._BatchNorm, torch.nn.modules.dropout._DropoutNd)
        if any(isinstance(module, forbidden) for module in self.policy.modules()):
            raise ValueError("audit does not support dropout or batch normalization")

    def _evaluate(self, sample):
        was_training = self.policy.training
        self.policy.set_training_mode(False)
        try:
            with torch.no_grad():
                obs = torch.as_tensor(sample["observations"], device=self.device)
                action = torch.as_tensor(sample["actions_raw"], device=self.device)
                value, log_prob, _ = self.policy.evaluate_actions(obs, action)
                normal = self.policy.get_distribution(obs).distribution
                return {
                    "log_prob": log_prob.cpu().numpy().reshape(-1).astype(np.float64),
                    "values": value.cpu().numpy().reshape(-1).astype(np.float64),
                    "action_mean": normal.mean.cpu().numpy().copy(),
                    "action_std": normal.stddev.cpu().numpy().copy(),
                }
        finally:
            self.policy.set_training_mode(was_training)

    def train(self):
        if self._audit_directory is None:
            return super().train()
        self._check_supported()
        buffer = self.rollout_buffer
        if not buffer.full or buffer.generator_ready:
            raise ValueError("audit requires a fresh, complete unflattened rollout")
        count = min(128, buffer.buffer_size * buffer.n_envs)
        sample = {
            name: _head(getattr(buffer, field), count)
            for name, field in (
                ("observations", "observations"),
                ("actions_raw", "actions"),
                ("old_log_prob", "log_probs"),
                ("old_values", "values"),
                ("advantages", "advantages"),
                ("returns", "returns"),
            )
        }
        # Full time-major arrays are needed for independent backward GAE.
        # Rewards already include SB3 timeout bootstraps, so may differ from
        # raw environment rewards recorded in the on_step callback.
        sample.update(
            {
                "gae_rewards": buffer.rewards.copy(),
                "gae_values": buffer.values.copy(),
                "gae_episode_starts": buffer.episode_starts.copy(),
                "gae_last_values": buffer.audit_last_values.copy(),
                "gae_last_dones": buffer.audit_last_dones.copy(),
                "gae_gamma": np.asarray(buffer.gamma),
                "gae_lambda": np.asarray(buffer.gae_lambda),
            }
        )
        before, hash_before = self._evaluate(sample), parameter_sha256(self.policy)
        super().train()
        after, hash_after = self._evaluate(sample), parameter_sha256(self.policy)
        self._write(sample, before, after, hash_before, hash_after)

    def _write(self, sample, before, after, hash_before, hash_after):
        arrays = dict(sample)
        clip_range = float(self.clip_range(self._current_progress_remaining))
        metrics = {}
        for name, result in (("before", before), ("after", after)):
            arrays.update({f"{key}_{name}": value for key, value in result.items()})
            logratio = result["log_prob"] - sample["old_log_prob"].astype(np.float64)
            ratio = np.exp(logratio)
            arrays[f"ratio_{name}"] = ratio
            metrics[f"approx_reverse_kl_{name}"] = float(np.mean(np.expm1(logratio) - logratio))
            metrics[f"ratio_clip_fraction_{name}"] = float(np.mean(np.abs(ratio - 1) > clip_range))
        if not all(np.isfinite(array).all() for array in arrays.values()):
            raise FloatingPointError("PPO update audit contains nonfinite samples or likelihoods")
        variance = float(np.var(sample["returns"]))
        explained = (
            1 - float(np.var(sample["returns"] - sample["old_values"])) / variance
            if variance > 0
            else None
        )
        filename = f"sample_{self._audit_index:06d}.npz"
        np.savez_compressed(self._audit_directory / filename, **arrays)
        learner_metrics = {
            key: float(value) if np.isfinite(value) else None
            for key, value in self.logger.name_to_value.items()
            if key.startswith("train/") and isinstance(value, (int, float, np.integer, np.floating))
        }
        row = {
            "audit_index": self._audit_index,
            "npz": filename,
            "num_timesteps": self.num_timesteps,
            "n_updates": self._n_updates,
            "n_audit_samples": len(sample["old_values"]),
            "sample_selection": "first min(128, rollout_size) in env-major order, not a random sample",
            "clip_range": clip_range,
            "advantage_mean": float(sample["advantages"].mean()),
            "advantage_std": float(sample["advantages"].std()),
            "explained_variance_old_values": explained,
            "param_sha256_before": hash_before,
            "param_sha256_after": hash_after,
            "parameters_changed": hash_before != hash_after,
            "logger_train_metrics": learner_metrics,
            **metrics,
        }
        with (self._audit_directory / "updates.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        self._audit_index += 1

    def _excluded_save_params(self):
        # Reloading a checkpoint must never reopen its training process's files.
        return super()._excluded_save_params() + ["_audit_directory", "_audit_index"]
