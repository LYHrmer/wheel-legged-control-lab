"""Warm-start grouped-clip PPO with actual KL-stop and optimizer audit.

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
from rl16_learning_11 import (LearningAudit, LEARNING_SCHEMA,
                              REQUIRED_LOG_FIELDS, HARD_FINITE_LOG_FIELDS,
                              hash_state)


PARENT_MODEL_SHA256 = "1fcfe833d8a7cfdbbc224c9007a35e44c035674ce948b51557434b77838476fb"
STAGE_SEEDS = (201001, 201011)
TARGET_KL = .03
HARD_KL = .30
STAGE20_LEARNING_SCHEMA = "d1-course20-grouped-clip-kl-audit-v1"
CLIP_SCHEMA = "d1-course20-grouped-clip-actual-batches-v1"


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


class ClipAudit20:
    """Bounded actual clip/Adam-step ledger, kept outside the saved PPO object."""

    def __init__(self, model, arm: str, max_steps: int):
        require(arm in ("A", "B"), "arm must be preregistered course A or B")
        require(type(max_steps) is int and max_steps == 512, "arm ceiling must be 512 steps")
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
        self.evaluated_batches = []
        self.train_records = []
        self.current_batch = None
        self.current_rollout = None
        self._train_first_step = None
        self._train_last_snapshot = None
        self._train_evaluations_start = 0
        self._train_optimizer_start = 0
        self._train_epoch_entries = 0

    def begin_train(self, rollout: int):
        require(type(rollout) is int and rollout == len(self.train_records),
                "train call does not match actual rollout order")
        require(self.current_rollout is None and self.current_batch is None,
                "nested or stale train state")
        self.current_rollout = rollout
        self._train_first_step = None
        self._train_last_snapshot = None
        self._train_evaluations_start = len(self.evaluated_batches)
        self._train_optimizer_start = self.optimizer_returned
        self._train_epoch_entries = 0

    def begin_epoch(self):
        require(self.current_rollout is not None and self.current_batch is None,
                "epoch began outside train or with an unconsumed batch")
        epoch = self._train_epoch_entries
        require(epoch < 4, "more than four PPO epochs entered")
        self._train_epoch_entries += 1
        return epoch

    def bind_batch(self, epoch, minibatch, old_log_prob):
        require(self.current_rollout is not None and self.current_batch is None
                and type(epoch) is int and 0 <= epoch < 4
                and type(minibatch) is int and 0 <= minibatch < 4,
                "invalid or overlapping PPO minibatch")
        self.current_batch = {"rollout": self.current_rollout, "epoch": epoch,
                              "minibatch": minibatch, "old_log_prob": old_log_prob,
                              "evaluated": False, "optimized": False}

    def evaluated(self, log_prob, torch):
        pending = self.current_batch
        require(pending is not None and not pending["evaluated"],
                "evaluate_actions is not bound to one PPO minibatch")
        with torch.no_grad():
            ratio = log_prob.detach() - pending["old_log_prob"].detach()
            kl = float(torch.mean((torch.exp(ratio) - 1) - ratio).cpu().numpy())
        record = {"rollout": pending["rollout"], "epoch": pending["epoch"],
                  "minibatch": pending["minibatch"], "approx_kl": kl,
                  "optimized": False, "optimizer_step": None,
                  "normal_kl_stop": math.isfinite(kl) and kl > 1.5*TARGET_KL,
                  "hard_stop": not math.isfinite(kl) or kl > HARD_KL}
        self.evaluated_batches.append(record)
        pending["evaluated"] = True
        pending["record"] = record
        if record["hard_stop"]:
            raise RuntimeError("nonfinite or >0.30 actual minibatch KL: " + repr(kl))
        return record

    def finish_batch(self):
        pending = self.current_batch
        require(pending is not None and pending["evaluated"],
                "PPO minibatch advanced without evaluate_actions")
        record = pending["record"]
        require(pending["optimized"] is (not record["normal_kl_stop"]),
                "KL decision differs from actual optimizer step")
        self.current_batch = None

    def finish_train(self, model):
        require(self.current_rollout is not None and self.current_batch is None,
                "train ended with an unclosed minibatch")
        actual = self.optimizer_returned - self._train_optimizer_start
        require(actual >= 1, "train call made zero optimizer updates")
        if self._train_last_snapshot is not None:
            step, pre, post, ids = self._train_last_snapshot
            if step != self._train_first_step:
                require(self.gradient_writer is not None, "last gradient writer missing")
                receipt = self.gradient_writer(step, pre, post, ids)
                require(isinstance(receipt, dict) and receipt,
                        "last gradient writer returned no receipt")
                self.gradient_sample_records.append(receipt)
                self.events[step]["full_gradient_saved"] = True
        evaluated = self.evaluated_batches[self._train_evaluations_start:]
        require(len(evaluated) >= actual and len(evaluated) <= 16,
                "evaluated/optimized minibatch counts differ")
        stopped = bool(evaluated[-1]["normal_kl_stop"])
        require(sum(not event["optimized"] for event in evaluated) == int(stopped),
                "unoptimized minibatch was not the terminal KL rejection")
        epochs = self._train_epoch_entries
        require(1 <= epochs <= 4, "entered epoch count differs")
        self.train_records.append({"rollout": self.current_rollout,
                                   "entered_epochs": epochs,
                                   "evaluated_minibatches": len(evaluated),
                                   "optimizer_steps": actual,
                                   "normal_kl_stop": stopped,
                                   "ending_n_updates": int(model._n_updates)})
        self.current_rollout = None
        self._train_last_snapshot = None

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
        pending = self.current_batch
        require(pending is not None and pending["evaluated"]
                and not pending["record"]["normal_kl_stop"],
                "clip was reached without an accepted evaluated minibatch")
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
        ids = {"global_step": step, "rollout": pending["rollout"],
               "epoch": pending["epoch"], "minibatch": pending["minibatch"]}
        keep_full = self._train_first_step is None
        pre_hashes = {"actor": gradient_hash(self.actor),
                      "critic": gradient_hash(self.critic)}
        pre_full = ({"actor::" + k: v for k, v in gradient_snapshot(self.actor).items()}
                    | {"critic::" + k: v for k, v in gradient_snapshot(self.critic).items()})
        kwargs = {"norm_type": norm_type, "error_if_nonfinite": error_if_nonfinite}
        if foreach is not None:
            kwargs["foreach"] = foreach
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
        post_full = ({"actor::" + k: v for k, v in gradient_snapshot(self.actor).items()}
                     | {"critic::" + k: v for k, v in gradient_snapshot(self.critic).items()})
        self._train_last_snapshot = (step, pre_full, post_full, ids)
        if keep_full:
            require(self.gradient_writer is not None,
                    "first actual minibatch gradient writer is missing")
            receipt = self.gradient_writer(step, pre_full, post_full, ids)
            require(isinstance(receipt, dict) and receipt,
                    "gradient writer did not return a persistence receipt")
            self.gradient_sample_records.append(receipt)
            self._train_first_step = step
        alpha_joint = min(1., .5 / (pre_joint + 1e-6))
        alpha_actor = min(1., .5 / (pre_actor + 1e-6))
        alpha_critic = min(1., .5 / (pre_critic + 1e-6))
        require(post_actor <= .50001 and post_critic <= .50001 and
                math.isclose(post_actor, pre_actor * alpha_actor, rel_tol=2e-5, abs_tol=1e-5)
                and math.isclose(post_critic, pre_critic * alpha_critic,
                                 rel_tol=2e-5, abs_tol=1e-5),
                "grouped clip did not cap each gradient group")
        self.events.append({
            "step": step, "arm": self.arm, **ids,
            "evaluated_approx_kl": pending["record"]["approx_kl"],
            "pre_actor_l2": pre_actor, "pre_critic_l2": pre_critic,
            "pre_joint_l2": pre_joint, "post_actor_l2": post_actor,
            "post_critic_l2": post_critic, "post_joint_l2": post_joint,
            "expected_global_alpha": alpha_joint,
            "hypothetical_global_alpha_on_same_pregradient": alpha_joint,
            "expected_actor_alpha": alpha_actor,
            "expected_critic_alpha": alpha_critic,
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
        pending = self.current_batch
        require(pending is not None and pending["evaluated"]
                and not pending["optimized"], "optimizer step lost PPO batch identity")
        pending["optimized"] = True
        pending["record"]["optimized"] = True
        pending["record"]["optimizer_step"] = self.optimizer_returned - 1
        if self.on_step is not None:
            self.on_step(dict(event))

    def as_dict(self):
        return {"schema": CLIP_SCHEMA, "arm": self.arm,
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
                "evaluated_batches": self.evaluated_batches,
                "train_records": self.train_records,
                "events": self.events}


class LearningAudit20(LearningAudit):
    """The old audited PPO seam with actual KL-shortened epoch accounting."""

    def __init__(self, budget, ppo_seed, arm, target_kl):
        super().__init__(budget, ppo_seed)
        require(arm in ("A", "B"), "arm is a preregistered course identity")
        require(math.isclose(float(target_kl), TARGET_KL, rel_tol=0, abs_tol=1e-12),
                "target_kl differs from frozen C20 contract")
        self.schema = STAGE20_LEARNING_SCHEMA
        self.arm = arm
        self.target_kl = float(target_kl)
        self.clip_audit = None

    def finish_update(self, context, model, logged):
        require(self.clip_audit is not None, "C20 clip audit is missing")
        # AuditedPPO calls this before scoped train_once regains control.
        # SB3 may leave the rejected minibatch's generator suspended on break.
        if self.clip_audit.current_batch is not None:
            self.clip_audit.finish_batch()
        self.clip_audit.finish_train(model)
        require(context.index < len(self.clip_audit.train_records),
                "PPO update is missing actual train/minibatch record")
        train = self.clip_audit.train_records[context.index]
        epochs = int(model._n_updates) - context.start_n_updates
        require(epochs == train["entered_epochs"] and 1 <= epochs <= 4,
                "_n_updates does not count actual entered epochs")
        require(context.optimizer_steps == train["optimizer_steps"]
                and 1 <= context.optimizer_steps <= 16
                and len(context.grad_norms) == context.optimizer_steps,
                "actual optimizer/gradient count differs")
        for name, parameter in model.policy.named_parameters():
            require(np.isfinite(parameter.detach().cpu().numpy()).all(),
                    "nonfinite policy parameter after update: " + name)
        record = {
            "update_index": context.index,
            "num_timesteps": int(model.num_timesteps),
            "timesteps_delta": int(model.num_timesteps)-context.start_timesteps,
            "n_updates_delta": epochs,
            "n_updates_total": int(model._n_updates),
            "entered_epochs": epochs,
            "evaluated_minibatches": train["evaluated_minibatches"],
            "optimizer_steps": context.optimizer_steps,
            "normal_kl_stop": train["normal_kl_stop"],
            "optimizer_instrumentation": context.instrumentation,
            "missing_grad_params": context.missing_grad_params,
            "nonfinite_grad_steps": context.nonfinite_grad_steps,
            "grad_norm_at_step_min": min(context.grad_norms),
            "grad_norm_at_step_max": max(context.grad_norms),
            "grad_norm_at_step_mean": sum(context.grad_norms)/len(context.grad_norms),
            "grad_norm_at_step_last": context.grad_norms[-1],
            "grad_norm_scope": "l2 norm over policy.parameters() after clip_grad_norm_",
        }
        for field in REQUIRED_LOG_FIELDS:
            if field not in logged:
                if field == "train/explained_variance":
                    self.diagnostics.append({"update_index": context.index,
                                             "field": field, "status": "missing_diagnostic"})
                    record[field] = None
                else:
                    self.note("missing_logger_field_" + field)
                    record[field] = None
                continue
            value = float(logged[field])
            if not math.isfinite(value):
                if field in HARD_FINITE_LOG_FIELDS:
                    raise RuntimeError(f"{field} is nonfinite at update {context.index}")
                if field == "train/explained_variance":
                    self.diagnostics.append({"update_index": context.index,
                                             "field": field, "status": "undefined_nonfinite"})
                else:
                    self.note("nonfinite_logger_field_" + field)
                record[field] = None
            else:
                record[field] = value
        require(record["train/n_updates"] is None
                or int(record["train/n_updates"]) == int(model._n_updates),
                "logger train/n_updates is stale")
        self.optimizer_steps += context.optimizer_steps
        self.train_calls += 1
        self.updates.append(record)


def training_receipt20(model, audit: LearningAudit20, clip: ClipAudit20):
    """Actual 32-rollout receipt; epoch/step ceiling is never reported as achieved."""
    require(isinstance(audit, LearningAudit20) and isinstance(clip, ClipAudit20)
            and model.learning_audit is audit and audit.clip_audit is clip,
            "C20 model/audit/clip binding differs")
    audit.snapshot_final(model)
    actual = {"num_timesteps": int(model.num_timesteps),
              "train_calls": audit.train_calls,
              "epochs": int(model._n_updates),
              "optimizer_steps": audit.optimizer_steps,
              "evaluated_minibatches": len(clip.evaluated_batches),
              "kl_early_stop_train_calls": sum(r["normal_kl_stop"] for r in clip.train_records),
              "rollouts": audit.rollout_count,
              "transitions": audit.transitions}
    violations = list(audit.violations)
    if (actual["num_timesteps"] != 32768 or actual["train_calls"] != 32
            or actual["rollouts"] != 32 or actual["transitions"] != 32768
            or len(audit.updates) != 32 or len(clip.train_records) != 32):
        violations.append("32 rollout/train-call and 32768 control closure differs")
    if (not 32 <= actual["epochs"] <= 128
            or not 32 <= actual["optimizer_steps"] <= 512
            or not actual["optimizer_steps"] <= actual["evaluated_minibatches"] <= 512
            or actual["evaluated_minibatches"] != actual["optimizer_steps"]
            + actual["kl_early_stop_train_calls"]):
        violations.append("actual KL/epoch/optimizer upper bound differs")
    if (clip.clip_attempted != clip.clip_returned
            or clip.clip_returned != clip.optimizer_attempted
            or clip.optimizer_attempted != clip.optimizer_returned
            or clip.optimizer_returned != actual["optimizer_steps"]
            or clip.native_clip_attempted != clip.native_clip_returned
            or clip.native_clip_returned != 2*actual["optimizer_steps"]):
        violations.append("actual grouped clip/Adam ledger differs")
    if audit.failures or any(event["hard_stop"] for event in clip.evaluated_batches):
        violations.append("training raised or KL hard stop occurred")
    initial, final = audit.initial_hashes, audit.final_hashes
    policy_changed = bool(initial and final and initial["policy_state"] != final["policy_state"])
    optimizer_changed = bool(initial and final and initial["optimizer_state"] != final["optimizer_state"])
    actor_changed = bool(initial and final and initial["actor_mean_state"] != final["actor_mean_state"])
    logstd_changed = bool(initial and final and initial["log_std_state"] != final["log_std_state"])
    if not policy_changed or not optimizer_changed:
        violations.append("policy or Adam state hash did not change")
    log_std = np.asarray(model.policy.log_std.detach().cpu().numpy(), dtype=np.float64)
    receipt = {"learning_schema": LEARNING_SCHEMA,
               "stage20_learning_schema": STAGE20_LEARNING_SCHEMA,
               "budget_spec": audit.budget.as_dict(),
               "budget_spec_sha256": audit.budget.canonical_sha256(),
               "ppo_seed": audit.ppo_seed, "course_arm": audit.arm,
               "target_kl": audit.target_kl,
               "normal_kl_threshold": 1.5*audit.target_kl,
               "hard_kl_threshold": HARD_KL,
               "status": "complete" if not violations else "incomplete",
               "qualified_for_final_checkpoint": not violations,
               "claims_learning_quality": False,
               "expected": dict(audit.expected), "actual": actual,
               "violations": violations,
               "diagnostics": list(audit.diagnostics),
               "construction": dict(audit.construction),
               "initial_hashes": dict(initial), "final_hashes": dict(final),
               "policy_state_changed": policy_changed,
               "actor_mean_state_changed": actor_changed,
               "log_std_state_changed": logstd_changed,
               "optimizer_state_changed": optimizer_changed,
               "final_log_std": log_std.tolist(),
               "failures": list(audit.failures),
               "updates": list(audit.updates), "rollouts": list(audit.rollouts),
               "note": "actual KL-shortened PPO update counts; heldout pairs determine RL contribution"}
    json.dumps(receipt, sort_keys=True, allow_nan=False)
    return receipt


def load_warm_start20(env, budget, seed, arm, parentpaths, *, target_kl, stage=1):
    """Load exactly one prior final, retaining policy and Adam moments."""
    from budget_spec_11 import BudgetSpec
    from rl16_learning_11 import _audited_ppo_class, LEARNING_RATE

    require(isinstance(budget, BudgetSpec) and budget.as_dict() ==
            BudgetSpec(32768, 1024, 256, 4).as_dict(), "arm budget differs")
    require(type(stage) is int and stage in (1, 2), "stage must be one or two")
    require(type(seed) is int and seed == STAGE_SEEDS[stage-1], "stage seed differs")
    require(arm in ("A", "B"), "unknown course arm")
    require(math.isclose(float(target_kl), TARGET_KL, rel_tol=0, abs_tol=1e-12),
            "target_kl differs")
    model_path = Path(parentpaths["model"]).resolve()
    metadata_path = Path(parentpaths["metadata"]).resolve()
    expected_zip = parentpaths["model_sha256"]
    require(file_identity(model_path)["sha256"] == expected_zip,
            "parent final ZIP identity differs")
    if stage == 1:
        require(expected_zip == PARENT_MODEL_SHA256,
                "stage1 parent is not the frozen C15 grouped final")
    metadata = json.loads(metadata_path.read_text())
    expected = metadata["training_receipt"]["final_hashes"]
    require(metadata["files"]["final_model.zip"]["sha256"] == expected_zip
            and metadata["training_receipt"]["status"] == "complete",
            "parent metadata/ZIP closure differs")
    if stage == 1:
        require(metadata["training_receipt"]["actual"] == {
            "num_timesteps": 16384, "train_calls": 16, "epochs": 64,
            "optimizer_steps": 256, "rollouts": 16, "transitions": 16384},
            "C15 grouped parent training receipt differs")
    else:
        require(metadata["training_receipt"]["stage20_learning_schema"]
                == STAGE20_LEARNING_SCHEMA
                and metadata["training_receipt"]["course_arm"] == arm
                and metadata["training_receipt"]["actual"]["num_timesteps"] == 32768,
                "stage2 parent is not the same arm's complete stage1 final")
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
    parent_actual = metadata["training_receipt"]["actual"]
    require(int(model.num_timesteps) == parent_actual["num_timesteps"]
            and int(model._n_updates) == parent_actual["epochs"],
            "parent PPO counters differ")
    steps = {int(float(state["step"])) for state in policy.optimizer.state.values()}
    if stage == 1:
        require(steps == {1280}, "C15 grouped Adam step counters differ")
    else:
        require(steps == {1280 + parent_actual["optimizer_steps"]},
                "stage2 Adam moments differ from stage1 actual optimizer steps")
    require(model.n_envs == 1 and model.device.type == "cpu" and
            model.get_vec_normalize_env() is None and model.gamma == .99 and
            model.gae_lambda == .95 and model.vf_coef == .5 and
            model.max_grad_norm == .5 and
            (model.target_kl is None if stage == 1 else model.target_kl == TARGET_KL) and
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
                 "parent_hashes": parent_hashes,
                 "parent_num_timesteps": parent_actual["num_timesteps"],
                 "parent_n_updates": parent_actual["epochs"],
                 "parent_optimizer_steps": next(iter(steps)),
                 "arm": arm, "stage": stage, "stage_seed": seed,
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
    model.target_kl = TARGET_KL
    audit = LearningAudit20(budget, seed, arm, target_kl)
    audit.construction = {
        "warm_start": True, "parent_final_zip_sha256": expected_zip,
        "parent_hashes": parent_hashes, "actor_reinitialized": False,
        "optimizer_reinitialized": False, "stage_counters_reset_only": True,
        "arm": arm, "stage": stage, "stage_seed": seed,
        "target_kl": TARGET_KL, "budget_spec": budget.as_dict(),
        "optimizer_lr": float(policy.optimizer.param_groups[0]["lr"]),
    }
    model.learning_audit = audit
    audit.snapshot_initial(model)
    require(audit.initial_hashes == parent_hashes,
            "stage initial hashes are not bitwise parent hashes")
    clip_audit = ClipAudit20(model, arm, budget.optimizer_steps)
    audit.clip_audit = clip_audit
    return model, audit, clip_audit, parentage


@contextmanager
def scoped_clip_audit20(model, clip_audit: ClipAudit20, *, soft_stop=None):
    """Bind actual PPO batches/KL and grouped clip only during original train()."""
    import torch

    require(model.policy is not None and isinstance(clip_audit, ClipAudit20),
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
        buffer = model.rollout_buffer
        original_get = buffer.get
        had_instance_get = "get" in buffer.__dict__
        instance_get = buffer.__dict__.get("get")
        original_evaluate = policy.evaluate_actions
        had_instance_evaluate = "evaluate_actions" in policy.__dict__
        instance_evaluate = policy.__dict__.get("evaluate_actions")
        handles = []
        clip_audit.begin_train(model.learning_audit.train_calls)

        def ensure_hooks_after_old_audit():
            if not handles:
                # AuditedPPO.train installs its hooks before calling super().train.
                # Register here so its post-hook books the completed Adam step first.
                handles.extend((optimizer.register_step_pre_hook(pre_hook),
                                optimizer.register_step_post_hook(post_hook)))

        def actual_batches(*get_args, **get_kwargs):
            epoch = clip_audit.begin_epoch()
            for minibatch, row in enumerate(original_get(*get_args, **get_kwargs)):
                if clip_audit.current_batch is not None:
                    clip_audit.finish_batch()
                clip_audit.bind_batch(epoch, minibatch, row.old_log_prob)
                yield row
            if clip_audit.current_batch is not None:
                clip_audit.finish_batch()

        def evaluate_at_minibatch_boundary(*call_args, **call_kwargs):
            if soft_stop is not None and soft_stop():
                raise TimeoutError("soft stop latched before next PPO minibatch")
            result = original_evaluate(*call_args, **call_kwargs)
            require(isinstance(result, tuple) and len(result) == 3,
                    "PPO evaluate_actions return contract differs")
            clip_audit.evaluated(result[1], torch)
            return result

        def intercept(parameters, max_norm, norm_type=2.0,
                      error_if_nonfinite=False, foreach=None):
            ensure_hooks_after_old_audit()
            return clip_audit.intercept(original_clip, parameters, max_norm,
                                        norm_type, error_if_nonfinite, foreach)

        require(torch.nn.utils.clip_grad_norm_ is original_clip,
                "clip function was already replaced")
        torch.nn.utils.clip_grad_norm_ = intercept
        buffer.get = actual_batches
        policy.evaluate_actions = evaluate_at_minibatch_boundary
        try:
            return original_train(*args, **kwargs)
        finally:
            torch.nn.utils.clip_grad_norm_ = original_clip
            for handle in handles:
                handle.remove()
            if had_instance_get:
                buffer.get = instance_get
            else:
                del buffer.__dict__["get"]
            if had_instance_evaluate:
                policy.evaluate_actions = instance_evaluate
            else:
                del policy.__dict__["evaluate_actions"]
            active = False

    model.train = train_once
    try:
        yield clip_audit
        require(len(clip_audit.train_records) == 32
                and 32 <= len(clip_audit.events) <= clip_audit.max_steps
                and len(clip_audit.events) == clip_audit.optimizer_attempted
                == clip_audit.optimizer_returned
                == clip_audit.clip_attempted == clip_audit.clip_returned,
                "actual KL-shortened clip/optimizer closure differs")
        require(clip_audit.native_clip_attempted == clip_audit.native_clip_returned ==
                2*clip_audit.optimizer_returned,
                "native clip call count differs")
    finally:
        torch.nn.utils.clip_grad_norm_ = original_clip
        if had_instance_train:
            model.train = instance_train
        else:
            del model.__dict__["train"]
