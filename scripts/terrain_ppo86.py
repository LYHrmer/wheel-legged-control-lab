"""One on-policy CPU PPO update on two fresh 64-tick segments per seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from wheel_legged_control.ppo_learning import generalized_advantage_estimate
from wheel_legged_control.ppo_update import FrozenBatch, UpdateConfig, one_sgd_update

SEEDS = (8603, 8611, 8623)
SEGMENT_TICKS = 64
STD = 0.25
GAMMA = 0.99
GAE_LAMBDA = 0.95
CONFIG = UpdateConfig(learning_rate=0.0003, clip_range=0.2, vf_coef=0.5, ent_coef=0.0)


def _write(path, value):
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, allow_nan=False, indent=2)
        target.write("\n")


def _vector(value, size=None):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or array.ndim != 1 or array.size == 0:
        raise ValueError("a nonempty real vector is required")
    array = array.astype(np.float64, copy=True)
    if (size is not None and array.size != size) or not np.isfinite(array).all():
        raise ValueError("nonfinite vector or unexpected dimension")
    return array


def _scalar(value):
    a = np.asarray(value)
    if a.ndim != 0 or a.dtype.kind not in "fiu" or not math.isfinite(float(a)):
        raise ValueError("finite real scalar required")
    return float(a)


def _reset(env, seed, size=None):
    result = env.reset(seed=seed)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Mapping):
        result = result[0]
    return _vector(result, size)


class _MeanActor(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size, dtype=torch.float64, device="cpu"))
        self.bias = nn.Parameter(torch.zeros((), dtype=torch.float64, device="cpu"))

    def forward(self, observation):
        return Normal((observation @ self.weight + self.bias).unsqueeze(-1), STD)


class _Critic(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size, dtype=torch.float64, device="cpu"))
        self.bias = nn.Parameter(torch.zeros((), dtype=torch.float64, device="cpu"))

    def forward(self, observation):
        return observation @ self.weight + self.bias


def _state(module):
    return {name: value.tolist() for name, value in module.state_dict().items()}


def _digest(state):
    text = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _evidence(info, physical):
    fields = {k: _scalar(info[k]) for k in ("requested_transfer_n", "allocated_transfer_n")}
    if physical or "applied_torque_nm" in info:
        fields["applied_torque_nm"] = _vector(info["applied_torque_nm"], 16).tolist()
    if "reward_terms" in info:
        fields["reward_terms"] = {name: _scalar(value) for name, value in info["reward_terms"].items()}
    if info.get("termination_reason") is not None:
        fields["termination_reason"] = str(info["termination_reason"])
    return fields


def _collect(env, seed, obs, actor, critic, physical, stream, counts):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    with torch.no_grad():
        for segment in range(2):
            if segment:
                obs = _reset(env, seed, len(obs))
                counts["resets_completed"] += 1
            for tick in range(1, SEGMENT_TICKS + 1):
                tensor = torch.from_numpy(obs)
                dist = actor(tensor)
                noise = torch.randn((1,), dtype=torch.float64, device="cpu", generator=rng)
                sample = dist.loc + dist.scale * noise
                raw = _scalar(sample.item())
                bounded = max(-1.0, min(raw, 1.0))
                counts["step_calls"] += 1
                following, reward, terminal, time_limit, info = env.step(bounded)
                counts["steps_returned"] += 1
                if not isinstance(terminal, (bool, np.bool_)) or not isinstance(time_limit, (bool, np.bool_)):
                    raise TypeError("termination and truncation must be booleans")
                following = _vector(following, len(obs))
                row = {
                    "segment": segment + 1, "segment_tick": tick,
                    "observation": obs.tolist(), "next_observation": following.tolist(),
                    "raw_action": raw, "env_action": bounded, "old_mean": _scalar(dist.loc.item()),
                    "old_log_prob": _scalar(dist.log_prob(sample).item()),
                    "old_value": _scalar(critic(tensor).item()),
                    "next_value": _scalar(critic(torch.from_numpy(following)).item()),
                    "reward": _scalar(reward), "terminated": bool(terminal),
                    "env_truncated": bool(time_limit),
                    "truncated": bool(time_limit) or tick == SEGMENT_TICKS,
                    **_evidence(info, physical),
                }
                stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                stream.flush()
                rows.append(row)
                counts["recorded_transitions"] += 1
                if terminal or (time_limit and tick < SEGMENT_TICKS):
                    return rows, f"segment {segment + 1} ended at tick {tick}"
                obs = following
    return rows, None


def _batch(rows):
    def column(name):
        return np.asarray([r[name] for r in rows])

    advantages, returns = generalized_advantage_estimate(
        column("reward"), column("old_value"), column("next_value"),
        column("terminated"), column("truncated"), gamma=GAMMA, gae_lambda=GAE_LAMBDA,
    )
    return FrozenBatch(column("observation"), column("raw_action")[:, None], column("old_log_prob"),
                       advantages, returns)


def _checkpoint(output, actor, critic, rows, seed, task):
    size = len(rows[0]["observation"])
    actor_state, critic_state = _state(actor), _state(critic)
    case = {"schema": "round86-smoke-linear-parameters-v1", "task": task, "seed": seed,
            "observation_size": size, "std": STD, "deployment_eligible": False,
            "actor": actor_state, "critic": critic_state}
    path = output / "diagnostic_parameters.json"
    _write(path, case)
    recovered = json.loads(path.read_text(encoding="utf-8"))
    if recovered != case:
        raise ValueError("saved diagnostic parameters differ after reading")
    probe_indices = (0, SEGMENT_TICKS - 1, SEGMENT_TICKS, 2 * SEGMENT_TICKS - 1)
    probes = torch.tensor([rows[i]["observation"] for i in probe_indices], dtype=torch.float64, device="cpu")
    new_actor, new_critic = _MeanActor(size), _Critic(size)
    for label, module in (("actor", new_actor), ("critic", new_critic)):
        module.load_state_dict({k: torch.tensor(v, dtype=torch.float64, device="cpu")
                                for k, v in recovered[label].items()}, strict=True)
    with torch.no_grad():
        before, readback = actor(probes).loc, new_actor(probes).loc
        values, readback_values = critic(probes), new_critic(probes)
    if not (torch.equal(before, readback) and torch.equal(values, readback_values)
            and torch.equal(before.clamp(-1, 1), readback.clamp(-1, 1))):
        raise ValueError("fixed probes changed after reconstructing the policy or critic")
    _write(output / "reload_probes.json", {
        "zero_based_rows": list(probe_indices), "observations": probes.tolist(), "mean": before.tolist(),
        "reloaded_mean": readback.tolist(), "value": values.tolist(), "reloaded_value": readback_values.tolist(),
    })
    return _digest(_state(new_actor)), float(before.abs().max().item())


def run_case(env, seed, output, *, task, physical_gate_passed=False):
    """Use a fresh env for one seed; its reset owns any warmup and terrain change."""
    if seed not in SEEDS or task not in ("synthetic", "physical"):
        raise ValueError("unregistered seed or task")
    physical = task == "physical"
    if physical and (physical_gate_passed is not True or isinstance(env, LagFixture)):
        raise ValueError("physical PPO requires an admitted live D1 environment")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    counts = dict.fromkeys(("resets_completed", "step_calls", "steps_returned", "recorded_transitions",
                            "updates_attempted", "optimizer_steps"), 0)
    scope = {"seed": seed, "task": task, "deployment_eligible": False}
    _write(output / "started.json", {**scope, "status_until_summary": "PARTIAL", "physical_gate_passed": physical,
                                     "segments": 2, "ticks_each": SEGMENT_TICKS, "std": STD,
                                     "gamma": GAMMA, "gae_lambda": GAE_LAMBDA, **vars(CONFIG)})
    try:
        if torch.get_num_threads() != 1:
            torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        obs = _reset(env, seed)
        counts["resets_completed"] += 1
        actor, critic = _MeanActor(len(obs)), _Critic(len(obs))
        initial = _digest(_state(actor))
        with (output / "transaction.jsonl").open("x", encoding="utf-8") as stream:
            rows, stopped = _collect(env, seed, obs, actor, critic, physical, stream, counts)
        if stopped:
            partial = {**scope, **counts, "status": "PARTIAL", "passed": False, "reason": stopped}
            _write(output / "partial.json", partial)
            return partial
        batch = _batch(rows)
        counts["updates_attempted"] += 1
        update = one_sgd_update(actor, critic, batch, CONFIG)
        counts["optimizer_steps"] = update["optimizer_steps"]
        _write(output / "ppo_update.json", {**update, "advantages": batch.advantages.tolist(),
                                           "returns": batch.returns.tolist()})
        final_hash, probe_change = _checkpoint(output, actor, critic, rows, seed, task)
        weight_change = max(abs(r["delta"]) for r in update["parameters"] if r["parameter"] == "actor.weight")
        max_ratio_error = max(abs(x - 1.0) for x in update["before"]["ratio"])
        if (counts["optimizer_steps"] != 1 or max_ratio_error > 1e-12
                or any(r["old_mean"] != 0.0 for r in rows)
                or not all(math.isfinite(v) and v > 0 for v in update["gradient_norms"].values())
                or final_hash == initial or (weight_change == 0 and probe_change == 0)):
            raise ValueError("on-policy likelihood, gradient, actor change or optimizer count check failed")
        result = {
            **scope, **counts, "status": "COMPLETE", "passed": True, "observation_size": len(obs),
            "actor_sha256_before": initial, "actor_sha256_after": final_hash, "fixed_probe_reload_exact": True,
            "actor_mean_weight_max_abs_delta": weight_change, "max_abs_fixed_probe_mean_change": probe_change,
            "gradient_norms": update["gradient_norms"], "requested_nonzero_ticks": sum(
                r["requested_transfer_n"] != 0 for r in rows), "allocated_nonzero_ticks": sum(
                r["allocated_transfer_n"] != 0 for r in rows),
            "max_abs_initial_ratio_error": max_ratio_error,
            "max_abs_requested_minus_allocated_n": max(abs(r["requested_transfer_n"] - r["allocated_transfer_n"])
                                                       for r in rows),
            "sampling_envs": 1, "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "synthetic_optimizer_steps": int(not physical), "d1_optimizer_steps": int(physical),
        }
        _write(output / "summary.json", result)
        return result
    except BaseException as error:
        _write(output / "partial.json", {**scope, **counts, "status": "PARTIAL", "passed": False,
                                         "reason": repr(error)})
        raise


class LagFixture:
    """Deterministic toy actuator with one-tick delay and bounded force in nominal newtons."""

    def __init__(self):
        self.segments = 0

    def reset(self, *, seed):
        self.segments += 1
        if self.segments > 2:
            raise ValueError("only two fixture segments are registered")
        self.target = 0.5 if self.segments == 1 else -0.5
        self.position = self.pending = 0.0
        self.tick = 0
        return (self.position, self.target, self.pending)

    def step(self, action):
        self.tick += 1
        requested = 8 * action
        applied = max(-1.5, min(self.pending, 1.5))
        self.position = 0.8 * self.position + 0.1 * applied
        self.pending = requested
        obs = (self.position, self.target, self.pending)
        reward = -(self.position - self.target) ** 2 - 0.01 * applied**2
        return obs, reward, False, self.tick == SEGMENT_TICKS, {
            "requested_transfer_n": requested, "allocated_transfer_n": applied,
        }


def run_fixture(output):
    """Three technical optimizer checks, with zero robot controls and zero D1 policy updates."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    _write(output / "started.json", {"task": "synthetic", "seeds": SEEDS, "status_until_summary": "PARTIAL"})
    try:
        for seed in SEEDS:
            record = run_case(LagFixture(), seed, output / f"seed_{seed}", task="synthetic")
            cases.append(record)
        complete = all(c["passed"] for c in cases)
        result = {"task": "synthetic", "status": "COMPLETE" if complete else "PARTIAL", "passed": complete,
                  "cases": cases, "d1_controls": 0, "d1_optimizer_steps": 0, "deployment_eligible": False,
                  "synthetic_transitions": sum(c["recorded_transitions"] for c in cases),
                  "synthetic_optimizer_steps": sum(c["optimizer_steps"] for c in cases)}
        _write(output / ("summary.json" if complete else "partial.json"), result)
        return result
    except BaseException as error:
        _write(output / "partial.json", {"status": "PARTIAL", "cases": cases, "reason": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=run_fixture.__doc__)
    parser.add_argument("--output", type=Path, required=True)
    summary = run_fixture(parser.parse_args().output)
    print(json.dumps({k: summary[k] for k in ("status", "synthetic_optimizer_steps", "d1_optimizer_steps")}))
