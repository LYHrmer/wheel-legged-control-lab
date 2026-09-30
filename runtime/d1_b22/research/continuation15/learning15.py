"""Warm-start PPO with one scoped gradient-clip intervention per arm.

No model, Torch, engine, or environment is imported or executed at module import.
Root owns env creation, exactly one learn call, persistence, and all physical steps.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import json

import numpy as np


PARENT_MODEL_SHA256 = "6cf2db80be7b990efc8be40eff307e58351eae839970b0c78ce5c9b5193f8e70"
SEED = 151001


def file_identity(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "bytes": size}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parameter_groups(policy):
    """Return disjoint actor/critic named parameters covering policy and Adam."""
    named = dict(policy.named_parameters(remove_duplicate=False))
    critic = {name: p for name, p in named.items()
              if name.startswith(("mlp_extractor.value_net.", "value_net."))}
    actor = {name: p for name, p in named.items() if name not in critic}
    require(len(named) == 13 and len(actor) == 7 and len(critic) == 6,
            "unexpected policy parameter groups")
    require("log_std" in actor and "value_net.bias" in critic,
            "log_std or value head is absent")
    ids = [id(p) for p in named.values()]
    require(len(set(ids)) == len(ids), "shared trainable parameter detected")
    optimizer_ids = [id(p) for group in policy.optimizer.param_groups for p in group["params"]]
    require(len(optimizer_ids) == len(ids) and set(optimizer_ids) == set(ids),
            "optimizer does not cover policy parameters exactly once")
    return actor, critic


def gradient_norm(parameters):
    """Read actual CPU gradients in float64; fail before a nonfinite optimizer step."""
    squared = 0.0
    for name, parameter in parameters.items():
        gradient = parameter.grad
        require(gradient is not None, f"missing gradient: {name}")
        array = gradient.detach().cpu().numpy().astype(np.float64)
        require(np.isfinite(array).all(), f"nonfinite gradient: {name}")
        squared += float(np.sum(array * array))
    result = math.sqrt(squared)
    require(math.isfinite(result), "nonfinite gradient norm")
    return result


def gradient_hash(parameters):
    """Stable byte digest of named, finite gradients at this actual minibatch."""
    digest = hashlib.sha256()
    for name, parameter in sorted(parameters.items()):
        gradient = parameter.grad
        require(gradient is not None, f"missing gradient for hash: {name}")
        array = np.ascontiguousarray(gradient.detach().cpu().numpy())
        require(np.isfinite(array).all(), f"nonfinite gradient for hash: {name}")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(repr(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def gradient_snapshot(parameters):
    return {name: p.grad.detach().cpu().numpy().copy() for name, p in parameters.items()}


def parameter_group_norm(parameters):
    squared = 0.0
    for name, parameter in parameters.items():
        array = parameter.detach().cpu().numpy().astype(np.float64)
        require(np.isfinite(array).all(), f"nonfinite parameter: {name}")
        squared += float(np.sum(array * array))
    return math.sqrt(squared)


def parameter_snapshot(parameters):
    return {name: p.detach().cpu().numpy().astype(np.float64, copy=True)
            for name, p in parameters.items()}


def parameter_delta_norm(parameters, before):
    squared = 0.0
    for name, p in parameters.items():
        after = p.detach().cpu().numpy().astype(np.float64)
        require(np.isfinite(after).all(), f"nonfinite parameter after Adam step: {name}")
        squared += float(np.sum((after - before[name]) ** 2))
    return math.sqrt(squared)


class ClipAudit:
    """Bounded actual clip/Adam-step ledger, kept outside the saved PPO object."""

    def __init__(self, model, arm: str, max_steps: int):
        require(arm in ("global", "grouped"), "arm must be global or grouped")
        require(type(max_steps) is int and max_steps == 256, "arm budget must be 256 steps")
        self.arm = arm
        self.max_steps = max_steps
        self.actor, self.critic = parameter_groups(model.policy)
        self.events = []
        self.clip_attempted = 0
        self.clip_returned = 0
        self.native_clip_attempted = 0
        self.native_clip_returned = 0
        self.optimizer_attempted = 0
        self.optimizer_returned = 0
        self.on_step = None
        self.gradient_writer = None
        self.gradient_sample_records = []
        self._before = None

    def set_gradient_writer(self, writer):
        """writer(step, pre, post, ids) must persist arrays and return a small receipt."""
        require(callable(writer) and self.gradient_writer is None and not self.events,
                "gradient writer must be set once before learning")
        self.gradient_writer = writer

    def set_gradient_directory(self, directory):
        """Built-in exclusive NPZ persistence when root has no archive writer yet."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)

        def write(step, pre, post, ids):
            path = directory / f"gradient_step_{step:04d}.npz"
            arrays = {f"pre::{name}": array for name, array in pre.items()}
            arrays.update({f"post::{name}": array for name, array in post.items()})
            with path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            return {"file": str(path), "identity": file_identity(path), "ids": ids}

        self.set_gradient_writer(write)

    def intercept(self, original, parameters, max_norm, norm_type=2.0,
                  error_if_nonfinite=False, foreach=None):
        require(len(self.events) == self.optimizer_attempted and
                self.clip_attempted == self.optimizer_attempted and
                self.clip_attempted < self.max_steps, "clip count exceeds step budget")
        self.clip_attempted += 1
        require(float(max_norm) == .5 and float(norm_type) == 2.0 and
                not error_if_nonfinite, "original PPO clip arguments differ")
        ordered = list(parameters)
        expected = list(self.actor.values()) + list(self.critic.values())
        require(len(ordered) == len(expected) and
                {id(p) for p in ordered} == {id(p) for p in expected},
                "clip did not receive exactly all policy parameters")
        pre_actor = gradient_norm(self.actor)
        pre_critic = gradient_norm(self.critic)
        pre_joint = math.hypot(pre_actor, pre_critic)
        step = len(self.events)
        ids = {"global_step": step, "rollout": step // 16,
               "epoch": (step % 16) // 4, "minibatch": step % 4}
        keep_full = step % 16 in (0, 15)
        pre_hashes = {"actor": gradient_hash(self.actor),
                      "critic": gradient_hash(self.critic)}
        pre_full = ({"actor::" + k: v for k, v in gradient_snapshot(self.actor).items()}
                    | {"critic::" + k: v for k, v in gradient_snapshot(self.critic).items()}) if keep_full else None
        kwargs = {"norm_type": norm_type, "error_if_nonfinite": error_if_nonfinite}
        if foreach is not None:
            kwargs["foreach"] = foreach
        if self.arm == "global":
            self.native_clip_attempted += 1
            returned = original(ordered, max_norm, **kwargs)
            self.native_clip_returned += 1
        else:
            self.native_clip_attempted += 1
            original(list(self.actor.values()), max_norm, **kwargs)
            self.native_clip_returned += 1
            self.native_clip_attempted += 1
            original(list(self.critic.values()), max_norm, **kwargs)
            self.native_clip_returned += 1
            returned = None  # PPO does not consume clip_grad_norm_'s return.
        post_actor = gradient_norm(self.actor)
        post_critic = gradient_norm(self.critic)
        post_joint = math.hypot(post_actor, post_critic)
        post_hashes = {"actor": gradient_hash(self.actor),
                       "critic": gradient_hash(self.critic)}
        if keep_full:
            require(self.gradient_writer is not None,
                    "fixed first/last minibatch gradient writer is missing")
            post_full = ({"actor::" + k: v for k, v in gradient_snapshot(self.actor).items()}
                         | {"critic::" + k: v for k, v in gradient_snapshot(self.critic).items()})
            receipt = self.gradient_writer(step, pre_full, post_full, ids)
            require(isinstance(receipt, dict) and receipt,
                    "gradient writer did not return a persistence receipt")
            self.gradient_sample_records.append(receipt)
        alpha_joint = min(1., .5 / (pre_joint + 1e-6))
        alpha_actor = min(1., .5 / (pre_actor + 1e-6))
        alpha_critic = min(1., .5 / (pre_critic + 1e-6))
        if self.arm == "global":
            require(post_joint <= .50001 and
                    math.isclose(post_joint, pre_joint * alpha_joint, rel_tol=2e-5, abs_tol=1e-5),
                    "global clip did not scale the joint gradient")
        else:
            require(post_actor <= .50001 and post_critic <= .50001 and
                    math.isclose(post_actor, pre_actor * alpha_actor, rel_tol=2e-5, abs_tol=1e-5)
                    and math.isclose(post_critic, pre_critic * alpha_critic,
                                     rel_tol=2e-5, abs_tol=1e-5),
                    "grouped clip did not cap each gradient group")
        self.events.append({
            "step": step, "arm": self.arm, **ids,
            "pre_actor_l2": pre_actor, "pre_critic_l2": pre_critic,
            "pre_joint_l2": pre_joint, "post_actor_l2": post_actor,
            "post_critic_l2": post_critic, "post_joint_l2": post_joint,
            "expected_global_alpha": alpha_joint,
            "hypothetical_global_alpha_on_same_pregradient": alpha_joint,
            "expected_actor_alpha": alpha_joint if self.arm == "global" else alpha_actor,
            "expected_critic_alpha": alpha_joint if self.arm == "global" else alpha_critic,
            "actual_actor_scale": post_actor / pre_actor if pre_actor else 1.,
            "actual_critic_scale": post_critic / pre_critic if pre_critic else 1.,
            "pre_gradient_sha256": pre_hashes, "post_gradient_sha256": post_hashes,
            "full_gradient_saved": keep_full,
            "missing_gradients": 0, "nonfinite_gradients": 0,
            "optimizer_attempted": False, "optimizer_returned": False,
        })
        self.clip_returned += 1
        return returned

    def before_optimizer(self):
        require(self.optimizer_attempted < self.max_steps and
                len(self.events) == self.optimizer_attempted + 1,
                "optimizer step without exactly one preceding clip")
        event = self.events[-1]
        require(not event["optimizer_attempted"], "duplicate optimizer pre-hook")
        pre_actor = gradient_norm(self.actor)
        pre_critic = gradient_norm(self.critic)
        require(math.isclose(pre_actor, event["post_actor_l2"], rel_tol=1e-7, abs_tol=1e-9)
                and math.isclose(pre_critic, event["post_critic_l2"], rel_tol=1e-7, abs_tol=1e-9),
                "gradients changed between clip and optimizer")
        self.optimizer_attempted += 1
        event["optimizer_attempted"] = True
        self._before = (parameter_snapshot(self.actor), parameter_snapshot(self.critic),
                        parameter_group_norm(self.actor), parameter_group_norm(self.critic))

    def after_optimizer(self):
        require(self._before is not None and self.optimizer_returned + 1 ==
                self.optimizer_attempted, "optimizer post-hook has no matching pre-hook")
        event = self.events[-1]
        event["optimizer_returned"] = True
        self.optimizer_returned += 1
        actor_before, critic_before, actor_norm, critic_norm = self._before
        event["adam_actor_parameter_delta_l2"] = parameter_delta_norm(self.actor, actor_before)
        event["adam_critic_parameter_delta_l2"] = parameter_delta_norm(self.critic, critic_before)
        event["adam_joint_parameter_delta_l2"] = math.hypot(
            event["adam_actor_parameter_delta_l2"], event["adam_critic_parameter_delta_l2"])
        event["adam_actor_delta_over_parameter_norm"] = (
            event["adam_actor_parameter_delta_l2"] / max(actor_norm, 1e-12))
        event["adam_critic_delta_over_parameter_norm"] = (
            event["adam_critic_parameter_delta_l2"] / max(critic_norm, 1e-12))
        self._before = None
        if self.on_step is not None:
            self.on_step(dict(event))

    def as_dict(self):
        return {"schema": "d1-warmstart-clip-audit-15-v1", "arm": self.arm,
                "max_optimizer_steps": self.max_steps,
                "clip_attempted": self.clip_attempted,
                "clip_returned": self.clip_returned,
                "clip_interceptions": self.clip_attempted,
                "completed_clip_events": len(self.events),
                "native_clip_attempted": self.native_clip_attempted,
                "native_clip_returned": self.native_clip_returned,
                "optimizer_attempted": self.optimizer_attempted,
                "optimizer_returned": self.optimizer_returned,
                "actor_parameter_names": list(self.actor),
                "critic_parameter_names": list(self.critic),
                "gradient_sample_records": self.gradient_sample_records,
                "events": self.events}


def load_warm_start(env, budget, seed, arm, parentpaths):
    """Load old final once, verify full optimizer parentage, then attach a new audit."""
    from budget_spec_11 import BudgetSpec
    from rl16_learning_11 import (LearningAudit, _audited_ppo_class, hash_state,
                                  LEARNING_RATE)

    require(isinstance(budget, BudgetSpec) and budget.as_dict() ==
            BudgetSpec(16384, 1024, 256, 4).as_dict(), "arm budget differs")
    require(type(seed) is int and seed == SEED, "shared stage seed differs")
    require(arm in ("global", "grouped"), "unknown arm")
    model_path = Path(parentpaths["model"]).resolve()
    metadata_path = Path(parentpaths["metadata"]).resolve()
    expected_zip = parentpaths.get("model_sha256", PARENT_MODEL_SHA256)
    require(expected_zip == PARENT_MODEL_SHA256 and
            file_identity(model_path)["sha256"] == expected_zip,
            "old final ZIP identity differs")
    metadata = json.loads(metadata_path.read_text())
    expected = metadata["training_receipt"]["final_hashes"]
    require(metadata["training_receipt"]["actual"] == {
        "num_timesteps": 65536, "train_calls": 64, "epochs": 256,
        "optimizer_steps": 1024, "rollouts": 64, "transitions": 65536},
        "old training receipt differs")
    model = _audited_ppo_class().load(str(model_path), env=env, device="cpu",
                                      force_reset=True)
    policy = model.policy
    parent_hashes = {
        "policy_state": hash_state("policy", policy.state_dict()),
        "optimizer_state": hash_state("optimizer", policy.optimizer.state_dict()),
        "actor_mean_state": hash_state("actor_mean", {
            "policy_net": policy.mlp_extractor.policy_net.state_dict(),
            "action_net": policy.action_net.state_dict()}),
        "log_std_state": hash_state("log_std", policy.log_std.detach()),
    }
    require(parent_hashes == expected, "warm-start policy/Adam state differs from old final")
    require(int(model.num_timesteps) == 65536 and int(model._n_updates) == 256,
            "old PPO counters differ")
    steps = {int(float(state["step"])) for state in policy.optimizer.state.values()}
    require(steps == {1024}, "Adam moment step counters differ from parent")
    require(model.n_envs == 1 and model.device.type == "cpu" and
            model.get_vec_normalize_env() is None and model.gamma == .99 and
            model.gae_lambda == .95 and model.vf_coef == .5 and
            model.max_grad_norm == .5 and model.target_kl is None and
            model.n_steps == 1024 and model.batch_size == 256 and
            model.n_epochs == 4 and model.clip_range_vf is None,
            "loaded PPO learning contract differs")
    require(float(policy.optimizer.param_groups[0]["lr"]) == LEARNING_RATE,
            "loaded Adam learning rate differs")
    require(all(float(model.lr_schedule(progress)) == LEARNING_RATE
                for progress in (1., .5, 0.)),
            "parent learning-rate schedule is not constant")
    parameter_groups(policy)
    parentage = {"parent_model_identity": file_identity(model_path),
                 "parent_metadata_identity": file_identity(metadata_path),
                 "parent_hashes": parent_hashes, "parent_num_timesteps": 65536,
                 "parent_n_updates": 256, "parent_optimizer_steps": 1024,
                 "arm": arm, "stage_seed": seed,
                 "actor_reinitialized": False, "optimizer_reinitialized": False}
    model.set_random_seed(seed)
    model.seed = seed
    require({
        "policy_state": hash_state("policy", policy.state_dict()),
        "optimizer_state": hash_state("optimizer", policy.optimizer.state_dict()),
        "actor_mean_state": hash_state("actor_mean", {
            "policy_net": policy.mlp_extractor.policy_net.state_dict(),
            "action_net": policy.action_net.state_dict()}),
        "log_std_state": hash_state("log_std", policy.log_std.detach()),
    } == parent_hashes, "setting stage seed changed parent parameters or Adam moments")
    model.num_timesteps = 0
    model._n_updates = 0
    audit = LearningAudit(budget, seed)
    audit.construction = {
        "warm_start": True, "old_final_zip_sha256": expected_zip,
        "parent_hashes": parent_hashes, "actor_reinitialized": False,
        "optimizer_reinitialized": False, "stage_counters_reset_only": True,
        "arm": arm, "stage_seed": seed, "budget_spec": budget.as_dict(),
        "optimizer_lr": float(policy.optimizer.param_groups[0]["lr"]),
    }
    model.learning_audit = audit
    audit.snapshot_initial(model)
    require(audit.initial_hashes == parent_hashes,
            "stage initial hashes are not bitwise parent hashes")
    clip_audit = ClipAudit(model, arm, budget.optimizer_steps)
    return model, audit, clip_audit, parentage


@contextmanager
def scoped_clip_audit(model, clip_audit: ClipAudit, *, soft_stop=None):
    """Patch only inside each original AuditedPPO.train; restore before save."""
    import torch

    require(model.policy is not None and isinstance(clip_audit, ClipAudit),
            "scoped clip audit is not bound to a PPO policy")
    optimizer = model.policy.optimizer
    original_train = model.train
    original_clip = torch.nn.utils.clip_grad_norm_
    had_instance_train = "train" in model.__dict__
    instance_train = model.__dict__.get("train")
    require(not had_instance_train, "model train method was already overridden")
    active = False

    def pre_hook(opt, args, kwargs):
        require(opt is optimizer, "another optimizer stepped")
        clip_audit.before_optimizer()

    def post_hook(opt, args, kwargs):
        require(opt is optimizer, "another optimizer stepped")
        clip_audit.after_optimizer()

    def train_once(*args, **kwargs):
        nonlocal active
        require(not active, "nested train call")
        active = True
        policy = model.policy
        original_evaluate = policy.evaluate_actions
        had_instance_evaluate = "evaluate_actions" in policy.__dict__
        instance_evaluate = policy.__dict__.get("evaluate_actions")
        handles = []

        def ensure_hooks_after_old_audit():
            if not handles:
                # AuditedPPO.train installs its hooks before calling super().train.
                # Register here so its post-hook books the completed Adam step first.
                handles.extend((optimizer.register_step_pre_hook(pre_hook),
                                optimizer.register_step_post_hook(post_hook)))

        def evaluate_at_minibatch_boundary(*call_args, **call_kwargs):
            if soft_stop is not None and soft_stop():
                raise TimeoutError("soft stop latched before next PPO minibatch")
            return original_evaluate(*call_args, **call_kwargs)

        def intercept(parameters, max_norm, norm_type=2.0,
                      error_if_nonfinite=False, foreach=None):
            ensure_hooks_after_old_audit()
            return clip_audit.intercept(original_clip, parameters, max_norm,
                                        norm_type, error_if_nonfinite, foreach)

        require(torch.nn.utils.clip_grad_norm_ is original_clip,
                "clip function was already replaced")
        torch.nn.utils.clip_grad_norm_ = intercept
        policy.evaluate_actions = evaluate_at_minibatch_boundary
        try:
            return original_train(*args, **kwargs)
        finally:
            torch.nn.utils.clip_grad_norm_ = original_clip
            for handle in handles:
                handle.remove()
            if had_instance_evaluate:
                policy.evaluate_actions = instance_evaluate
            else:
                del policy.__dict__["evaluate_actions"]
            active = False

    model.train = train_once
    try:
        yield clip_audit
        require(len(clip_audit.events) == clip_audit.optimizer_attempted ==
                clip_audit.optimizer_returned == clip_audit.max_steps,
                "arm did not complete exactly 256 clipped optimizer steps")
        require(clip_audit.clip_attempted == clip_audit.clip_returned ==
                clip_audit.max_steps, "clip interception closure differs")
        require(clip_audit.native_clip_attempted == clip_audit.native_clip_returned ==
                clip_audit.max_steps * (1 if clip_audit.arm == "global" else 2),
                "native clip call count differs")
    finally:
        torch.nn.utils.clip_grad_norm_ = original_clip
        if had_instance_train:
            model.train = instance_train
        else:
            del model.__dict__["train"]
