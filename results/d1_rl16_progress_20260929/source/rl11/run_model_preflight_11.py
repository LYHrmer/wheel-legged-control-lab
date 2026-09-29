"""Root-only synthetic PPO interface check: no robot or engine imports allowed."""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

W = Path(__file__).resolve().parent
R = Path("/home/lyh/wheel-legged-control-lab")
SITE = Path("/home/lyh/.local/lib/python3.10/site-packages")
CONTRACT = W / "model_preflight_contract_11.md"
PPO_SEED = 88612
CAP = 1024
blocked = []
optional_disabled = []


class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "cv2":
            optional_disabled.append(fullname)
            raise ModuleNotFoundError("optional Atari image dependency disabled for Box99 preflight")
        if fullname.split(".")[0] in {
            "mujoco",
            "glfw",
            "pybullet",
            "dm_control",
            "wheel_legged_control",
            "wheel_legged_control_lab",
            "course_plant_08",
            "full_drive_env_08",
            "course_native_guard_08",
            "d1_rolling_engine_runtime",
            "engine_binding",
            "world_upright_course_11",
        }:
            blocked.append(fullname)
            raise RuntimeError("synthetic preflight forbids engine import: " + fullname)


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def sources():
    paths = {
        Path(__file__).resolve(),
        CONTRACT,
        W / "model_preflight_count_clarification_11.md",
        W.parent / "upright11/world_upright_course_11.py",
        Path(sys.executable).resolve(),
        W / "rl16_learning_11.py",
        W.parent / "course_impl08/residual16_math_08.py",
        W / "budget_spec_11.py",
    }
    roots = [SITE / "torch", R / ".local-deps/stable_baselines3"]
    # Capture all Python and ELF implementation, before importing either library.
    for root in roots:
        if not root.is_dir():
            raise RuntimeError("missing dependency source: " + str(root))
        paths.update(
            p
            for p in root.rglob("*")
            if p.is_file() and (p.suffix in (".py", ".so") or ".so." in p.name)
        )
    for name in ("numpy", "gymnasium"):
        root = next(
            (
                base / name
                for base in (SITE, R / ".local-deps")
                if (base / name / "__init__.py").is_file()
            ),
            None,
        )
        if root is None:
            raise RuntimeError("missing dependency: " + name)
        paths.update(
            p
            for p in root.rglob("*")
            if p.is_file() and (p.suffix in (".py", ".so") or ".so." in p.name)
        )
        vendor = root.with_name(root.name + ".libs")
        if vendor.is_dir():
            paths.update(p for p in vendor.rglob("*") if p.is_file())
    # This CPU run uses the existing CUDA-enabled Torch distribution.
    # Freeze adjacent vendor implementation without importing/probing a model.
    for base in (SITE, R / ".local-deps"):
        vendor = base / "nvidia"
        if vendor.is_dir():
            paths.update(
                p for p in vendor.rglob("*")
                if p.is_file() and (p.suffix in (".py", ".so") or ".so." in p.name)
            )
    return {str(p): identity(p) for p in sorted(paths)}


class StopSyntheticBudget(RuntimeError):
    pass


def run(out, result):
    sys.meta_path.insert(0, NoEngine())
    sys.path[:0] = [str(SITE), str(R / ".local-deps"), str(W.parent / "course_impl08"), str(W)]
    import gymnasium as gym
    import numpy as np
    import rl16_learning_11 as learning
    import torch
    from budget_spec_11 import BudgetSpec

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    result["actual_modules"] = {
        mod.__name__: str(Path(mod.__file__).resolve())
        for mod in (np, gym, torch, learning)
    }
    result["versions"] = {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "gymnasium": gym.__version__,
    }
    for path in result["actual_modules"].values():
        if path not in result["source_hashes_before"]:
            raise RuntimeError("imported origin not frozen: " + path)

    class Synthetic99x16(gym.Env):
        """Finite analytic signal; these are never robot states or rewards."""

        observation_space = gym.spaces.Box(-5.0, 5.0, (99,), dtype=np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (16,), dtype=np.float32)

        def __init__(self):
            self.success = 0
            self.attempts = 0
            self.reset_count = 0
            self.episode_tick = 0
            self.rows = []
            self.terminal_observations = []
            result["synthetic_live_counts"] = {
                "successful_transitions": 0, "attempts": 0,
                "resets": 0, "truncations": 0, "callbacks": 0,
            }

        def observation(self):
            # Local episode position and global success count are distinguishable.
            x = np.arange(99, dtype=np.float64)
            return (
                0.25 * np.sin(x * 0.031 + self.episode_tick * 0.013)
                + 0.1 * np.cos(x * 0.017 + self.success * 0.007)
            ).astype(np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.reset_count += 1
            result["synthetic_live_counts"]["resets"] = self.reset_count
            self.episode_tick = 0
            return self.observation(), {"synthetic_unqualified": True}

        def step(self, action):
            self.attempts += 1
            result["synthetic_live_counts"]["attempts"] = self.attempts
            if self.success == CAP:
                raise StopSyntheticBudget(
                    "attempt 1025 refused before advancing synthetic state"
                )
            handed = np.asarray(action, dtype=np.float32).copy()
            if handed.shape != (16,) or not np.isfinite(handed).all():
                raise RuntimeError("invalid synthetic action")
            if np.any(np.abs(handed) > 1):
                raise RuntimeError("SB3 failed to clip synthetic action")
            effective = handed.astype(np.float64)
            if self.success % 7 == 0:
                effective[:] = 0
            self.success += 1
            result["synthetic_live_counts"]["successful_transitions"] = self.success
            self.episode_tick += 1
            obs = self.observation()
            reward = float(
                1.0
                - np.mean((effective - 0.2 * np.sin(self.success * 0.01)) ** 2)
                + 0.03 * np.cos(self.success * 0.037)
            )
            truncated = self.episode_tick == 512
            if truncated:
                self.terminal_observations.append(obs.copy())
                result["synthetic_live_counts"]["truncations"] += 1
            self.rows.append(
                {
                    "handed": handed.copy(),
                    "effective": effective.copy(),
                    "reward": float(np.float32(reward)),
                    "done": truncated,
                }
            )
            return (
                obs,
                reward,
                False,
                truncated,
                {
                    "policy_input_action": handed.copy(),
                    "policy_clipped_action": handed.copy(),
                    "applied_action": effective.copy(),
                    "completed_control_intervals": self.success,
                    "terminal_reason": "synthetic_512_truncation"
                    if truncated
                    else None,
                    "synthetic_unqualified": True,
                },
            )

    env = Synthetic99x16()
    budget = BudgetSpec(1024, 1024, 256, 4)
    model, audit = learning.build_audited_ppo(env, budget=budget, ppo_seed=PPO_SEED)
    probe = np.stack([env.observation(), np.zeros(99, dtype=np.float32)])
    if (torch.count_nonzero(model.policy.action_net.weight).item()
            or torch.count_nonzero(model.policy.action_net.bias).item()):
        raise RuntimeError("initial actor affine output head is not exactly zero")
    result["initial_actor_zero_affine_head_verified_without_predict"] = True
    model_calls = {"save": 0, "load": 0, "predict": 0}
    prediction_rows = []
    result["actual_model_calls"] = model_calls
    result["actual_deterministic_probe_batches"] = prediction_rows

    def count_model_calls(frame, event, arg):
        if (event == "call" and frame.f_code.co_name in model_calls
                and frame.f_code.co_filename.endswith("stable_baselines3/common/base_class.py")):
            name = frame.f_code.co_name
            model_calls[name] += 1
            if name == "predict":
                observed = np.asarray(frame.f_locals["observation"])
                prediction_rows.append({
                    "original_model": frame.f_locals.get("self") is model,
                    "shape": list(observed.shape),
                    "deterministic": frame.f_locals.get("deterministic"),
                })
    sys.setprofile(count_model_calls)
    if not np.array_equal(
        model.policy.log_std.detach().cpu().numpy(), np.full(16, -2, np.float32)
    ):
        raise RuntimeError("initial log_std differs from -2")
    callbacks = []

    def transition(record):
        index = len(callbacks)
        row = env.rows[index]
        if record["num_timesteps"] != index + 1:
            raise RuntimeError("callback timestep mismatch")
        if not np.array_equal(record["sb3_clipped_action"], row["handed"]):
            raise RuntimeError("callback handed action mismatch")
        if not np.array_equal(record["effective_applied_action"], row["effective"]):
            raise RuntimeError("callback effective action mismatch")
        if (
            record["reward_before_bootstrap"] != row["reward"]
            or record["done"] != row["done"]
            or record["truncated"] != row["done"]
        ):
            raise RuntimeError("callback reward/done mismatch")
        callbacks.append(record)
        result["synthetic_live_counts"]["callbacks"] = len(callbacks)

    callback = learning.make_rollout_callback(audit, on_transition=transition)
    model.learn(total_timesteps=budget.total_controls, callback=callback, progress_bar=False)
    result["synthetic_actual"] = {
        "successful_transitions": env.success,
        "attempts": env.attempts,
        "resets": env.reset_count,
        "truncations": len(env.terminal_observations),
        "callbacks": len(callbacks),
    }
    if result["synthetic_actual"] != {
        "successful_transitions": 1024,
        "attempts": 1024,
        "resets": 3,
        "truncations": 2,
        "callbacks": 1024,
    }:
        raise RuntimeError("synthetic exact counts failed")
    # The actual VecEnv retains the terminal observation at the last truncation.
    terminal = model.get_env().buf_infos[0]["terminal_observation"]
    if not np.array_equal(terminal, env.terminal_observations[-1]):
        raise RuntimeError("VecEnv did not preserve actual terminal observation")
    result["terminal_observation_verified"] = True
    receipt = learning.training_receipt(model, audit)
    result["training_receipt"] = receipt
    if receipt["actual"] != {
        "num_timesteps": 1024,
        "train_calls": 1,
        "epochs": 4,
        "optimizer_steps": 16,
        "rollouts": 1,
        "transitions": 1024,
    }:
        raise RuntimeError("actual PPO update counts differ")
    if audit.violations or audit.failures:
        raise RuntimeError("actual learning audit recorded a violation")
    if receipt["status"] != "complete" or not receipt["qualified_for_final_checkpoint"]:
        raise RuntimeError("new explicit synthetic budget did not complete")
    if (receipt["budget_spec"] != budget.as_dict()
            or receipt["budget_spec_sha256"] != budget.canonical_sha256()
            or receipt["ppo_seed"] != PPO_SEED):
        raise RuntimeError("new explicit budget/seed provenance differs")
    for name in (
        "policy_state_changed",
        "actor_mean_state_changed",
        "log_std_state_changed",
        "optimizer_state_changed",
    ):
        if receipt[name] is not True:
            raise RuntimeError("actual update did not change " + name)
    if not all(
        x["nonfinite_grad_steps"] == 0 and x["optimizer_steps"] == 16
        for x in receipt["updates"]
    ):
        raise RuntimeError("gradient/update hooks failed")
    from copy import deepcopy
    from datetime import datetime, timezone

    metadata = {
        "run_id": "synthetic_budget11_interface_only",
        "wall_clock_utc": datetime.now(timezone.utc).isoformat(),
        "contract_documents": {str(CONTRACT): identity(CONTRACT)},
        "source_files": result["source_hashes_before"],
        "engine_elf": None,
        "dependency_hashes": {key: value for key, value in result["source_hashes_before"].items()
                              if "/site-packages/" in key or "/.local-deps/" in key},
        "task_schema": learning.WORLD_UPRIGHT_TASK_SCHEMA,
        "reward_schema": learning.WORLD_UPRIGHT_REWARD_SCHEMA,
        "reference_schema": learning.WORLD_UPRIGHT_REFERENCE_SCHEMA,
        "reference_definition": learning.WORLD_UPRIGHT_REFERENCE_DEFINITION,
        "observation_schema": learning.WORLD_UPRIGHT_OBSERVATION_SCHEMA,
        "controller_schema": "synthetic-effective-action-gate-no-controller",
        "control_loop_schema": learning.WORLD_UPRIGHT_LOOP_SCHEMA,
        "control_dt_s": .01,
        "physics_dt_s": .002,
        "physics_steps_per_control": 5,
        "curriculum": "analytic signals only; no physical steps",
        "command_seed": PPO_SEED,
        "measurement_seed_stream": "synthetic only",
        "spawn_position_m": None,
        "qualified_caps": {},
        "budget_ledger": {"synthetic_transitions": env.success, "robot_controls": 0,
                         "native_calls": 0},
        "residual_permission_semantics": "synthetic every seventh action zeroed",
        "baseline_credit_note": "No physical qualification or learning benefit is established.",
        "synthetic_only": True,
        "qualified_for_robot_use": False,
        "timing_fields_are_interface_constants_not_executed_physics": True,
    }
    final_path = out / "SYNTHETIC_ONLY_NOT_FOR_ROBOT_FINAL"
    manifest = learning.save_final_and_verify(
        model, str(final_path), metadata, probe, audit=audit, receipt=receipt,
    )
    save(out / "synthetic_final_manifest.json", manifest)
    result["synthetic_final_manifest"] = manifest
    result["complete_budget_final_and_reload_verified"] = True
    result["loader_rejections"] = []
    # These malformed manifests must reject before reaching the load site.
    # No second model load or any altered on-disk sidecar is necessary.
    final_bytes = {str(path): identity(path) for path in sorted(final_path.iterdir())}
    def different_budget(doc):
        other = BudgetSpec(65536, 1024, 256, 4)
        doc["budget_spec"] = other.as_dict()
        doc["budget_spec_sha256"] = other.canonical_sha256()
        doc["trusted_fields"]["budget_spec"] = other.as_dict()
        doc["trusted_fields"]["budget_spec_sha256"] = other.canonical_sha256()

    for label, mutate in (
        ("missing_reference", lambda doc: doc["trusted_fields"].pop("reference_schema")),
        ("different_budget", different_budget),
        ("different_seed", lambda doc: doc["trusted_fields"].update(ppo_seed=PPO_SEED + 1)),
    ):
        invalid = deepcopy(manifest)
        mutate(invalid)
        calls_before = dict(model_calls)
        try:
            learning.load_and_verify_final(str(final_path), invalid)
        except ValueError as error:
            result["loader_rejections"].append({
                "case": label, "message": str(error), "calls_before": calls_before,
                "calls_after": dict(model_calls),
            })
        else:
            raise RuntimeError("strict loader accepted " + label)
        if model_calls != calls_before:
            raise RuntimeError("negative manifest reached a model load/predict/save")
    sys.setprofile(None)
    result["actual_model_calls"] = model_calls
    result["actual_deterministic_probe_batches"] = prediction_rows
    if model_calls != {"save": 1, "load": 1, "predict": 3}:
        raise RuntimeError("actual new model save/load/probe counts differ")
    if (sum(row["original_model"] for row in prediction_rows) != 2
            or any(row["shape"] != [2, 99] or not row["deterministic"]
                   for row in prediction_rows)):
        raise RuntimeError("actual original/reloaded probe batches differ")
    if any(identity(Path(path)) != record for path, record in final_bytes.items()):
        raise RuntimeError("negative manifest checks changed correct checkpoint bytes")
    with (out / "synthetic_callback_trace.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            raw=np.stack([x["raw_gaussian_action"] for x in callbacks]),
            clipped=np.stack([x["sb3_clipped_action"] for x in callbacks]),
            effective=np.stack([x["effective_applied_action"] for x in callbacks]),
            rewards=np.array([x["reward_before_bootstrap"] for x in callbacks]),
            dones=np.array([x["done"] for x in callbacks]),
            terminal_observations=np.stack(env.terminal_observations),
        )
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, choices=(1,), required=True)
    args = parser.parse_args()
    out = W / f"model_preflight_{args.round:02d}"
    if out.exists():
        raise RuntimeError("exclusive preflight directory already exists")
    go = json.loads(
        (W / f"astra_model_preflight_go_11_{args.round:02d}.json").read_text()
    )
    if (
        go.get("decision") != "GO"
        or go.get("runner_sha256") != identity(Path(__file__))["sha256"]
    ):
        raise RuntimeError("missing exact preflight source GO")
    for key, path in (
        ("learning_sha256", W / "rl16_learning_11.py"),
        ("budget_sha256", W / "budget_spec_11.py"),
        ("contract_sha256", CONTRACT),
    ):
        if go.get(key) != identity(path)["sha256"]:
            raise RuntimeError("new interface GO source differs: " + key)
    if (go.get("synthetic_transition_limit") != CAP or go.get("ppo_seed") != PPO_SEED
            or go.get("robot_control_limit") != 0 or go.get("native_limit") != 0
            or go.get("wallclock_limit_s") != 120):
        raise RuntimeError("new interface GO exact budget differs")
    before = sources()
    out.mkdir()
    started = time.monotonic()
    save(
        out / "reservation.json",
        {
            "synthetic_only": True,
            "round": args.round,
            "successful_transition_cap": CAP,
            "refusal_attempt_cap": 0,
            "robot_control_steps": 0,
            "native_steps": 0,
            "wallclock_limit_s": 120,
            "source_hashes": before,
        },
    )
    result = {
        "synthetic_only": True,
        "qualified_for_robot_use": False,
        "passed": False,
        "robot_control_steps": 0,
        "native_steps": 0,
        "source_hashes_before": before,
    }

    def timeout(_sig, _frame):
        raise TimeoutError("120s synthetic preflight limit")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(120)
    try:
        run(out, result)
        result["passed"] = True
    except BaseException as error:  # noqa: BLE001 -- durable failure receipt, never a retry
        result["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        sys.setprofile(None)
        signal.alarm(0)
        result["wallclock_s"] = time.monotonic() - started
        result["blocked_engine_imports"] = blocked
        result["optional_dependency_imports_disabled"] = optional_disabled
        result["source_hashes_changed"] = [
            p
            for p, value in before.items()
            if not Path(p).is_file() or identity(Path(p)) != value
        ]
        if (blocked or result["source_hashes_changed"] or result["wallclock_s"] > 120
                or result.get("synthetic_actual", {}).get("attempts") != CAP):
            result["passed"] = False
        save(out / "receipt.json", result)
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in ("passed", "synthetic_actual", "error", "wallclock_s")
            }
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
