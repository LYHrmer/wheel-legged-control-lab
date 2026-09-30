"""Audited PPO construction, learning receipt and strict final-checkpoint tools.

This module owns *only* the learning side of the 08-R pilot: it builds one
actual ``stable_baselines3.PPO`` with the contract hyper-parameters, instruments
the real ``train``/rollout boundaries read-only, and saves/reloads exactly one
final checkpoint. It never constructs a plant or model, never resets or steps an
environment, never evaluates a policy against physics and never retries
anything. Root owns the single environment, the C/Python ledger, every physical
step, the curriculum and the one ``model.learn`` call.

Import-time dependencies are stdlib + numpy + the pure ``residual16_math_08``
constants; ``torch``, ``stable_baselines3`` and ``gymnasium`` are imported only
inside the explicit builders/loaders, after root's preflight has finished.

Nothing in this file has been executed. Every count below is a claim to be
verified by the actual run, not evidence of learning.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from collections.abc import Callable
from typing import Any

import numpy as np
from residual16_math_08 import (
    ACTION_SCHEMA,
    CONTROL_DT_S,
    LEG_ACTION_SCALE_RAD,
    WHEEL_ACTION_SCALE_RAD_S,
)

LEARNING_SCHEMA = "d1-course-rl16-ppo-learning-audit-v1"
FINAL_CHECKPOINT_SCHEMA = "d1-course-rl16-final-checkpoint-v1"

OBSERVATION_SIZE = 99
ACTION_SIZE = 16
OBSERVATION_ABS_BOUND = 5.0
ACTION_ABS_BOUND = 1.0

PPO_SEED = 88401
N_STEPS = 1024
BATCH_SIZE = 256
N_EPOCHS = 4
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.001
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
LOG_STD_INIT = -2.0
NET_ARCH = (128, 128)

TOTAL_TIMESTEPS = 262144
EXPECTED_TRAIN_CALLS = 256
EXPECTED_EPOCHS = 1024
EXPECTED_MINIBATCHES_PER_EPOCH = 4
EXPECTED_OPTIMIZER_STEPS = 4096
PHYSICS_STEPS_PER_CONTROL = 5
MAX_PROBE_OBSERVATIONS = 32

# Fields recorded by the actual local PPO.train / BaseAlgorithm._update_learning_rate.
REQUIRED_LOG_FIELDS = (
    "train/loss",
    "train/value_loss",
    "train/policy_gradient_loss",
    "train/entropy_loss",
    "train/approx_kl",
    "train/clip_fraction",
    "train/explained_variance",
    "train/n_updates",
    "train/clip_range",
    "train/learning_rate",
    "train/std",
)

# Losses/KL must be finite or the run is dead; explained_variance is only a
# diagnostic (it is nan when the return batch has zero variance), so its
# undefined value is recorded separately and never counted as a violation.
HARD_FINITE_LOG_FIELDS = (
    "train/loss",
    "train/value_loss",
    "train/policy_gradient_loss",
    "train/entropy_loss",
    "train/approx_kl",
    "train/clip_fraction",
    "train/std",
)

REQUIRED_METADATA_KEYS = (
    "run_id",
    "wall_clock_utc",
    "contract_documents",
    "source_files",
    "engine_elf",
    "dependency_hashes",
    "task_schema",
    "reward_schema",
    "observation_schema",
    "controller_schema",
    "control_loop_schema",
    "control_dt_s",
    "physics_dt_s",
    "physics_steps_per_control",
    "curriculum",
    "command_seed",
    "measurement_seed_stream",
    "spawn_position_m",
    "qualified_caps",
    "budget_ledger",
    "residual_permission_semantics",
    "baseline_credit_note",
)

# Keys this module writes itself; a caller must not pre-empt them, so a hand
# edited sidecar cannot claim a schema/receipt the run did not produce.
MODULE_OWNED_METADATA_KEYS = (
    "loader_schema",
    "learning_schema",
    "observation_size",
    "action_size",
    "observation_abs_bound",
    "action_abs_bound",
    "action_schema",
    "leg_action_scale_rad",
    "wheel_action_scale_rad_s",
    "net_arch",
    "log_std_init",
    "ppo_seed",
    "sb3_version",
    "torch_version",
    "numpy_version",
    "probe_observation_shape",
    "probe_action_shape",
    "files",
    "training_receipt",
)

MODEL_FILE = "final_model.zip"
METADATA_FILE = "final_metadata.json"
PROBE_OBS_FILE = "final_probe_observations_f32.bin"
PROBE_ACTION_FILE = "final_probe_actions_f32.bin"
FAILURE_MODEL_FILE = "failure_checkpoint.zip"
FAILURE_METADATA_FILE = "failure_receipt.json"


# --------------------------------------------------------------------------- #
# deterministic hashing / small validators
# --------------------------------------------------------------------------- #


def _feed(digest: hashlib._Hash, value: Any) -> None:
    """Feed one canonical, dtype/shape/raw-byte encoding of ``value``.

    Torch tensors are handled by duck typing so this module never imports
    torch. The encoding is independent of ``torch.save`` archive framing.
    """
    if value is None:
        digest.update(b"N")
    elif isinstance(value, (bool, np.bool_)):
        digest.update(b"B1" if bool(value) else b"B0")
    elif isinstance(value, (int, np.integer)):
        digest.update(b"I" + repr(int(value)).encode("ascii"))
    elif isinstance(value, (float, np.floating)):
        digest.update(b"F" + struct.pack("<d", float(value)))
    elif isinstance(value, str):
        digest.update(b"S" + value.encode("utf-8"))
    elif isinstance(value, bytes):
        digest.update(b"Y" + value)
    elif isinstance(value, dict):
        digest.update(b"D")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            digest.update(b"K" + repr(key).encode("utf-8"))
            _feed(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"L")
        for item in value:
            _feed(digest, item)
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"A" + str(array.dtype).encode("ascii")
                      + repr(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    elif hasattr(value, "detach") and hasattr(value, "cpu"):
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(b"T" + str(array.dtype).encode("ascii")
                      + repr(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    else:
        raise TypeError(f"state hashing does not support {type(value).__name__}")


def hash_state(label: str, state: Any) -> str:
    """SHA256 of one canonical state-dict encoding (sorted keys, raw bytes)."""
    digest = hashlib.sha256()
    digest.update(b"L" + label.encode("utf-8"))
    _feed(digest, state)
    return digest.hexdigest()


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_file(path: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _float32_batch(value: Any, width: int, label: str, limit: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float32 or array.ndim != 2 or array.shape[1] != width:
        raise TypeError(f"{label} must be a float32 array of shape (N, {width})")
    if not 1 <= array.shape[0] <= limit:
        raise ValueError(f"{label} must hold between 1 and {limit} rows")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite")
    return np.ascontiguousarray(array, dtype=np.float32)


def _describe_box(space: Any, size: int, bound: float, label: str) -> dict[str, Any]:
    """Validate one actual float32 Box of ``size`` with symmetric ``bound``."""
    if type(space).__name__ != "Box":
        raise TypeError(f"{label} must be a gymnasium Box, got {type(space).__name__}")
    low, high = np.asarray(space.low), np.asarray(space.high)
    if (space.shape != (size,) or np.dtype(space.dtype) != np.float32
            or low.shape != (size,) or high.shape != (size,)
            or not np.all(low == np.float32(-bound))
            or not np.all(high == np.float32(bound))):
        raise ValueError(
            f"{label} must be Box(-{bound}, {bound}, ({size},), float32)"
        )
    return {"shape": [size], "dtype": "float32", "low": -bound, "high": bound}


# --------------------------------------------------------------------------- #
# audit state
# --------------------------------------------------------------------------- #


class _UpdateContext:
    """Per-``train``-call scratch state for the scoped optimizer probe."""

    __slots__ = (
        "grad_norms", "index", "instrumentation", "missing_grad_params",
        "nonfinite_grad_steps", "optimizer_steps", "start_n_updates",
        "start_timesteps",
    )

    def __init__(self, index: int, start_n_updates: int, start_timesteps: int) -> None:
        self.index = index
        self.start_n_updates = start_n_updates
        self.start_timesteps = start_timesteps
        self.optimizer_steps = 0
        self.grad_norms: list[float] = []
        self.nonfinite_grad_steps = 0
        self.missing_grad_params = 0
        self.instrumentation = "none"

    def record_gradients(self, optimizer: Any) -> None:
        """Read-only gradient audit at the actual step boundary (post-clip)."""
        total = 0.0
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                gradient = getattr(parameter, "grad", None)
                if gradient is None:
                    self.missing_grad_params += 1
                    continue
                array = np.asarray(gradient.detach().cpu().numpy(), dtype=np.float64)
                if not np.isfinite(array).all():
                    self.nonfinite_grad_steps += 1
                    raise RuntimeError(
                        "nonfinite gradient entered optimizer.step at update "
                        f"{self.index}, step {self.optimizer_steps}"
                    )
                total += float(np.sum(array * array))
        norm = math.sqrt(total)
        if not math.isfinite(norm):
            self.nonfinite_grad_steps += 1
            raise RuntimeError(f"nonfinite gradient norm at update {self.index}")
        self.grad_norms.append(norm)


class LearningAudit:
    """Bounded, append-only record of one actual PPO run."""

    def __init__(self) -> None:
        self.schema = LEARNING_SCHEMA
        self.expected = {
            "total_timesteps": TOTAL_TIMESTEPS,
            "train_calls": EXPECTED_TRAIN_CALLS,
            "epochs": EXPECTED_EPOCHS,
            "optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
            "minibatches_per_epoch": EXPECTED_MINIBATCHES_PER_EPOCH,
            "rollouts": EXPECTED_TRAIN_CALLS,
            "transitions": TOTAL_TIMESTEPS,
        }
        self.construction: dict[str, Any] = {}
        self.updates: list[dict[str, Any]] = []
        self.rollouts: list[dict[str, Any]] = []
        self.violations: list[str] = []
        self.diagnostics: list[dict[str, Any]] = []
        self.train_calls = 0
        self.optimizer_steps = 0
        self.transitions = 0
        self.rollout_count = 0
        self.initial_hashes: dict[str, str] = {}
        self.final_hashes: dict[str, str] = {}
        self.training_started = False
        self.failures: list[dict[str, Any]] = []

    # -- hashes ----------------------------------------------------------- #

    def _hashes(self, model: Any) -> dict[str, str]:
        for name, parameter in model.policy.named_parameters():
            if not np.isfinite(parameter.detach().cpu().numpy()).all():
                raise RuntimeError(f"nonfinite policy parameter {name}")
        policy = model.policy
        return {
            "policy_state": hash_state("policy", policy.state_dict()),
            "actor_mean_state": hash_state("actor_mean", {
                "policy_net": policy.mlp_extractor.policy_net.state_dict(),
                "action_net": policy.action_net.state_dict(),
            }),
            "log_std_state": hash_state("log_std", policy.log_std.detach()),
            "optimizer_state": hash_state("optimizer",
                                          policy.optimizer.state_dict()),
        }

    def snapshot_initial(self, model: Any) -> None:
        if self.initial_hashes:
            raise RuntimeError("initial parameter hashes are already captured")
        self.initial_hashes = self._hashes(model)

    def snapshot_final(self, model: Any) -> None:
        self.final_hashes = self._hashes(model)

    def note(self, violation: str) -> None:
        if violation not in self.violations:
            self.violations.append(violation)

    # -- train-call bookkeeping ------------------------------------------- #

    def begin_update(self, n_updates: int, timesteps: int) -> _UpdateContext:
        if self.train_calls >= EXPECTED_TRAIN_CALLS:
            raise RuntimeError(
                f"train() was called more than the contracted {EXPECTED_TRAIN_CALLS} times"
            )
        return _UpdateContext(self.train_calls, int(n_updates), int(timesteps))

    def fail_update(self, context: _UpdateContext, error: BaseException) -> None:
        """Preserve the partial counts of a raising train() call."""
        self.optimizer_steps += context.optimizer_steps
        self.failures.append({
            "update_index": context.index,
            "exception_type": type(error).__name__,
            "exception_text": str(error)[:512],
            "partial_optimizer_steps": context.optimizer_steps,
            "partial_grad_norm_samples": len(context.grad_norms),
            "nonfinite_grad_steps": context.nonfinite_grad_steps,
        })
        self.note(f"train_call_{context.index}_raised_{type(error).__name__}")

    def finish_update(self, context: _UpdateContext, model: Any,
                      logged: dict[str, float]) -> None:
        for name, parameter in model.policy.named_parameters():
            if not np.isfinite(parameter.detach().cpu().numpy()).all():
                raise RuntimeError(f"nonfinite policy parameter after update: {name}")
        epochs = int(model._n_updates) - context.start_n_updates
        self.optimizer_steps += context.optimizer_steps
        self.train_calls += 1
        norms = context.grad_norms
        record = {
            "update_index": context.index,
            "num_timesteps": int(model.num_timesteps),
            "timesteps_delta": int(model.num_timesteps) - context.start_timesteps,
            "n_updates_delta": epochs,
            "n_updates_total": int(model._n_updates),
            "optimizer_steps": context.optimizer_steps,
            "optimizer_instrumentation": context.instrumentation,
            "missing_grad_params": context.missing_grad_params,
            "nonfinite_grad_steps": context.nonfinite_grad_steps,
            "grad_norm_at_step_min": min(norms) if norms else None,
            "grad_norm_at_step_max": max(norms) if norms else None,
            "grad_norm_at_step_mean": (sum(norms) / len(norms)) if norms else None,
            "grad_norm_at_step_last": norms[-1] if norms else None,
            "grad_norm_scope": "l2 norm over policy.parameters() after clip_grad_norm_",
        }
        for field in REQUIRED_LOG_FIELDS:
            if field not in logged:
                if field == "train/explained_variance":
                    self.diagnostics.append({
                        "update_index": context.index,
                        "field": field,
                        "status": "missing_diagnostic",
                    })
                    record[field] = None
                    continue
                self.note(f"missing_logger_field_{field}")
                record[field] = None
                continue
            value = float(logged[field])
            if not math.isfinite(value):
                if field in HARD_FINITE_LOG_FIELDS:
                    raise RuntimeError(f"{field} is nonfinite at update {context.index}")
                if field == "train/explained_variance":
                    self.diagnostics.append({
                        "update_index": context.index,
                        "field": field,
                        "status": "undefined_nonfinite",
                    })
                    record[field] = None
                    continue
                self.note(f"nonfinite_logger_field_{field}")
                record[field] = None
                continue
            record[field] = value
        if record["train/n_updates"] is not None and int(record["train/n_updates"]) != int(model._n_updates):
            raise RuntimeError(
                "logger snapshot is stale: train/n_updates does not match _n_updates"
            )
        if epochs != N_EPOCHS:
            raise RuntimeError(
                f"train() advanced _n_updates by {epochs}, contract requires {N_EPOCHS}"
            )
        if context.optimizer_steps != EXPECTED_MINIBATCHES_PER_EPOCH * N_EPOCHS:
            raise RuntimeError(
                f"train() ran {context.optimizer_steps} optimizer steps, contract "
                f"requires {EXPECTED_MINIBATCHES_PER_EPOCH * N_EPOCHS}"
            )
        if len(norms) != context.optimizer_steps:
            raise RuntimeError("gradient probe and optimizer step count disagree")
        self.updates.append(record)

    # -- rollout bookkeeping ---------------------------------------------- #

    def add_rollout(self, summary: dict[str, Any]) -> None:
        if self.rollout_count >= EXPECTED_TRAIN_CALLS:
            raise RuntimeError(
                f"more than the contracted {EXPECTED_TRAIN_CALLS} rollouts were collected"
            )
        self.rollout_count += 1
        self.rollouts.append(summary)


# --------------------------------------------------------------------------- #
# audited PPO construction
# --------------------------------------------------------------------------- #


def _install_optimizer_probes(optimizer: Any, context: _UpdateContext) -> Callable[[], None]:
    """Attach scoped read-only step probes; returns the restore callable."""

    def pre_hook(opt: Any, args: Any, kwargs: Any) -> None:
        context.record_gradients(opt)

    def post_hook(opt: Any, args: Any, kwargs: Any) -> None:
        context.optimizer_steps += 1

    if (hasattr(optimizer, "register_step_pre_hook")
            and hasattr(optimizer, "register_step_post_hook")):
        handles = (optimizer.register_step_pre_hook(pre_hook),
                   optimizer.register_step_post_hook(post_hook))
        context.instrumentation = "torch_step_pre_post_hooks"

        def restore_hooks() -> None:
            for handle in handles:
                handle.remove()

        return restore_hooks

    original_step = optimizer.step

    def counted_step(*args: Any, **kwargs: Any) -> Any:
        pre_hook(optimizer, args, kwargs)
        result = original_step(*args, **kwargs)
        post_hook(optimizer, args, kwargs)
        return result

    optimizer.step = counted_step
    context.instrumentation = "bound_method_wrapper"

    def restore_step() -> None:
        if optimizer.__dict__.get("step") is counted_step:
            del optimizer.__dict__["step"]

    return restore_step


_AUDITED_PPO_CLASS: Any = None


def _audited_ppo_class() -> Any:
    """Build (once) the PPO subclass that overrides only ``train``."""
    global _AUDITED_PPO_CLASS
    if _AUDITED_PPO_CLASS is not None:
        return _AUDITED_PPO_CLASS

    from stable_baselines3 import PPO

    class AuditedPPO(PPO):
        """Actual SB3 PPO; ``train`` is surrounded, never reimplemented."""

        learning_audit: LearningAudit | None = None

        def train(self) -> None:
            audit = self.learning_audit
            if not isinstance(audit, LearningAudit):
                raise TypeError("AuditedPPO.train requires an attached LearningAudit")
            context = audit.begin_update(self._n_updates, self.num_timesteps)
            restore = _install_optimizer_probes(self.policy.optimizer, context)
            try:
                try:
                    super().train()
                    logged = {key: value for key, value
                              in self.logger.name_to_value.items()
                              if key.startswith("train/")}
                except BaseException as error:  # partial counts, then rethrow
                    audit.fail_update(context, error)
                    raise
            finally:
                restore()
            audit.finish_update(context, self, logged)

    _AUDITED_PPO_CLASS = AuditedPPO
    return AuditedPPO


def build_audited_ppo(env: Any) -> tuple[Any, LearningAudit]:
    """Construct the contract PPO on one already-prepared env.

    ``env`` may be a single gym Env or a single-env VecEnv; it is never reset,
    stepped or wrapped here. Returns ``(model, audit)``; the caller invokes
    ``model.learn`` exactly once and this module never retries.
    """
    import stable_baselines3 as sb3
    import torch as th

    observation_box = _describe_box(env.observation_space, OBSERVATION_SIZE,
                                    OBSERVATION_ABS_BOUND, "observation_space")
    action_box = _describe_box(env.action_space, ACTION_SIZE,
                               ACTION_ABS_BOUND, "action_space")
    num_envs = getattr(env, "num_envs", None)
    if num_envs is not None and int(num_envs) != 1:
        raise ValueError("the 08-R pilot trains on exactly one environment")

    audited = _audited_ppo_class()
    model = audited(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        use_sde=False,
        target_kl=None,
        seed=PPO_SEED,
        device="cpu",
        verbose=0,
        policy_kwargs={
            "net_arch": {"pi": list(NET_ARCH), "vf": list(NET_ARCH)},
            "activation_fn": th.nn.Tanh,
            "log_std_init": LOG_STD_INIT,
        },
    )

    policy = model.policy
    with th.no_grad():
        policy.action_net.weight.zero_()
        policy.action_net.bias.zero_()
        policy.log_std.fill_(LOG_STD_INIT)

    if model.n_envs != 1 or model.get_vec_normalize_env() is not None:
        raise RuntimeError("the audited PPO requires one env and no VecNormalize")
    if model.device.type != "cpu" or model.target_kl is not None:
        raise RuntimeError("the audited PPO must be CPU with no target_kl early stop")
    if policy.squash_output or policy.use_sde:
        raise RuntimeError("the audited PPO must use an unsquashed diagonal Gaussian")
    zero_actor = (not np.any(policy.action_net.weight.detach().cpu().numpy())
                  and not np.any(policy.action_net.bias.detach().cpu().numpy()))
    log_std = policy.log_std.detach().cpu().numpy()
    if not zero_actor or log_std.shape != (ACTION_SIZE,) or not np.all(log_std == np.float32(LOG_STD_INIT)):
        raise RuntimeError("actor output must start at zero with log_std = -2")
    learning_rate = float(model.policy.optimizer.param_groups[0]["lr"])
    if not math.isclose(learning_rate, LEARNING_RATE, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("optimizer learning rate does not match the contract")
    buffer_size = N_STEPS * model.n_envs
    minibatches = -(-buffer_size // BATCH_SIZE)
    if (minibatches != EXPECTED_MINIBATCHES_PER_EPOCH
            or minibatches * N_EPOCHS * EXPECTED_TRAIN_CALLS != EXPECTED_OPTIMIZER_STEPS
            or EXPECTED_TRAIN_CALLS * buffer_size != TOTAL_TIMESTEPS
            or EXPECTED_TRAIN_CALLS * N_EPOCHS != EXPECTED_EPOCHS):
        raise RuntimeError("the configured PPO cannot reach the contracted counts")

    audit = LearningAudit()
    model.learning_audit = audit
    audit.construction = {
        "sb3_version": sb3.__version__,
        "torch_version": th.__version__,
        "numpy_version": np.__version__,
        "policy_class": type(policy).__name__,
        "algorithm_class": type(model).__name__,
        "env_class": type(model.env).__name__,
        "observation_space": observation_box,
        "action_space": action_box,
        "net_arch": {"pi": list(NET_ARCH), "vf": list(NET_ARCH)},
        "activation": "Tanh",
        "optimizer_class": type(model.policy.optimizer).__name__,
        "optimizer_lr": learning_rate,
        "optimizer_eps": model.policy.optimizer.param_groups[0].get("eps"),
        "share_features_extractor": bool(policy.share_features_extractor),
        "features_extractor": type(policy.features_extractor).__name__,
        "log_std_init": LOG_STD_INIT,
        "ppo_seed": PPO_SEED,
        "minibatches_per_epoch": minibatches,
        "n_updates_semantics": (
            "PPO.train increments _n_updates once per epoch, so 256 train calls "
            "give 1024 _n_updates and 4096 optimizer.step calls"
        ),
        "seed_side_effect": (
            "PPO(seed=88401) calls set_random_seed and env.seed(88401); that only "
            "reseeds the env measurement stream, not root's physical command seeds"
        ),
    }
    audit.snapshot_initial(model)
    return model, audit


# --------------------------------------------------------------------------- #
# rollout callback bridge
# --------------------------------------------------------------------------- #


def make_rollout_callback(audit: LearningAudit,
                         on_transition: Callable[[dict[str, Any]], None] | None = None,
                         on_rollout: Callable[[dict[str, Any]], None] | None = None,
                         *, include_controller_record: bool = False) -> Any:
    """Return one SB3 callback reading the actual rollout locals.

    ``on_transition`` receives one small record per real environment
    transition (raw Gaussian, SB3-clipped and controller-effective action plus
    reward/terminal flags); this module keeps only bounded per-rollout scalars,
    so root's chunk buffers stay the single transition store.
    """
    if not isinstance(audit, LearningAudit):
        raise TypeError("the rollout callback needs the LearningAudit of this run")
    for label, hook in (("on_transition", on_transition), ("on_rollout", on_rollout)):
        if hook is not None and not callable(hook):
            raise TypeError(f"{label} must be callable when supplied")

    from stable_baselines3.common.callbacks import BaseCallback

    class _AuditRolloutCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self._rollout_index = -1
            self._reset_rollout()

        def _reset_rollout(self) -> None:
            self._steps = 0
            self._dones = 0
            self._clip_steps = 0
            self._gate_zero_steps = 0
            self._saturated_components = 0
            self._reward_sum = 0.0
            self._raw_abs_max = 0.0

        def _on_training_start(self) -> None:
            total = self.locals.get("total_timesteps")
            if type(total) is not int:
                raise RuntimeError("learn() did not publish an integer total_timesteps")
            if total != TOTAL_TIMESTEPS:
                raise RuntimeError(
                    f"learn(total_timesteps={total}) differs from the contracted "
                    f"{TOTAL_TIMESTEPS}"
                )
            if int(self.model.num_timesteps) != 0 or self.model.n_envs != 1:
                raise RuntimeError("the pilot expects one fresh single-env run")
            if audit.training_started:
                raise RuntimeError("this audit has already seen a learn() call")
            audit.training_started = True

        def _on_rollout_start(self) -> None:
            self._rollout_index += 1
            self._reset_rollout()

        def _on_step(self) -> bool:
            actions = self.locals["actions"]
            clipped = self.locals["clipped_actions"]
            rewards = self.locals["rewards"]
            dones = self.locals["dones"]
            infos = self.locals["infos"]
            if (not isinstance(actions, np.ndarray) or actions.shape != (1, ACTION_SIZE)
                    or not isinstance(clipped, np.ndarray)
                    or clipped.shape != (1, ACTION_SIZE)):
                raise RuntimeError("rollout actions must be actual (1, 16) arrays")
            if not np.isfinite(actions).all() or not np.isfinite(clipped).all():
                raise RuntimeError("rollout actions are nonfinite")
            if not np.array_equal(clipped, np.clip(actions, -ACTION_ABS_BOUND, ACTION_ABS_BOUND)):
                raise RuntimeError(
                    "clipped_actions is not the plain clip of the Gaussian sample"
                )
            if len(infos) != 1 or len(rewards) != 1 or len(dones) != 1:
                raise RuntimeError("the pilot expects one env per rollout step")
            info = infos[0]
            raw = np.asarray(actions[0], dtype=np.float64)
            clip_row = np.asarray(clipped[0], dtype=np.float64)
            handed = np.asarray(info["policy_input_action"], dtype=np.float64)
            env_clipped = np.asarray(info["policy_clipped_action"], dtype=np.float64)
            applied = np.asarray(info["applied_action"], dtype=np.float64)
            for label, array in (("policy_input_action", handed),
                                 ("policy_clipped_action", env_clipped),
                                 ("applied_action", applied)):
                if array.shape != (ACTION_SIZE,) or not np.isfinite(array).all():
                    raise RuntimeError(f"info {label} must be finite shape (16,)")
            if not np.array_equal(handed, clip_row):
                raise RuntimeError(
                    "env policy_input_action differs from the action SB3 handed over"
                )
            if not np.array_equal(env_clipped, clip_row):
                raise RuntimeError("env policy_clipped_action differs from the SB3 clip")
            if np.any(np.abs(applied) > ACTION_ABS_BOUND):
                raise RuntimeError("effective applied action escaped the +/-1 box")
            reward = float(rewards[0])
            if not math.isfinite(reward):
                raise RuntimeError("rollout reward is nonfinite")
            clip_active = bool(np.any(clip_row != raw))
            gate_zeroed = bool(not np.any(applied) and np.any(clip_row))
            self._steps += 1
            self._dones += int(bool(dones[0]))
            self._clip_steps += int(clip_active)
            self._gate_zero_steps += int(gate_zeroed)
            self._saturated_components += int(np.count_nonzero(np.abs(raw) > ACTION_ABS_BOUND))
            self._reward_sum += reward
            self._raw_abs_max = max(self._raw_abs_max, float(np.max(np.abs(raw))))
            audit.transitions += 1
            local_steps = self.locals.get("n_steps")
            if isinstance(local_steps, int) and local_steps + 1 != self._steps:
                raise RuntimeError("callback step count disagrees with collect_rollouts")
            if on_transition is not None:
                record = {
                    "num_timesteps": int(self.num_timesteps),
                    "rollout_index": self._rollout_index,
                    "step_in_rollout": self._steps - 1,
                    "raw_gaussian_action": np.array(actions[0], dtype=np.float32),
                    "sb3_clipped_action": np.array(clipped[0], dtype=np.float32),
                    "effective_applied_action": np.array(applied, dtype=np.float64),
                    "clip_active": clip_active,
                    "gate_zeroed_action": gate_zeroed,
                    "reward_before_bootstrap": reward,
                    "done": bool(dones[0]),
                    "truncated": bool(info.get("TimeLimit.truncated", False)),
                    "terminal_reason": info.get("terminal_reason"),
                    "completed_control_intervals": info.get("completed_control_intervals"),
                }
                if include_controller_record:
                    record["controller_record"] = info.get("controller_record")
                on_transition(record)
            return True

        def _on_rollout_end(self) -> None:
            expected = self.locals.get("n_rollout_steps")
            if type(expected) is not int or expected != N_STEPS or self._steps != N_STEPS:
                raise RuntimeError(
                    f"rollout {self._rollout_index} collected {self._steps} of "
                    f"{N_STEPS} contracted steps"
                )
            summary = {
                "rollout_index": self._rollout_index,
                "transitions": self._steps,
                "num_timesteps_at_end": int(self.model.num_timesteps),
                "terminal_steps": self._dones,
                "clipped_steps": self._clip_steps,
                "gate_zeroed_steps": self._gate_zero_steps,
                "saturated_action_components": self._saturated_components,
                "reward_sum_before_bootstrap": self._reward_sum,
                "raw_gaussian_abs_max": self._raw_abs_max,
            }
            audit.add_rollout(summary)
            if on_rollout is not None:
                on_rollout(dict(summary))

    return _AuditRolloutCallback()


# --------------------------------------------------------------------------- #
# receipt
# --------------------------------------------------------------------------- #


def training_receipt(model: Any, audit: LearningAudit) -> dict[str, Any]:
    """Return a json-ready receipt of the real counts and state change."""
    if not isinstance(audit, LearningAudit):
        raise TypeError("training_receipt needs the LearningAudit of this run")
    audit.snapshot_final(model)
    actual = {
        "num_timesteps": int(model.num_timesteps),
        "train_calls": audit.train_calls,
        "epochs": int(model._n_updates),
        "optimizer_steps": audit.optimizer_steps,
        "rollouts": audit.rollout_count,
        "transitions": audit.transitions,
    }
    violations = list(audit.violations)
    for key, expected in (("num_timesteps", TOTAL_TIMESTEPS),
                          ("train_calls", EXPECTED_TRAIN_CALLS),
                          ("epochs", EXPECTED_EPOCHS),
                          ("optimizer_steps", EXPECTED_OPTIMIZER_STEPS),
                          ("rollouts", EXPECTED_TRAIN_CALLS),
                          ("transitions", TOTAL_TIMESTEPS)):
        if actual[key] != expected:
            violations.append(f"{key}={actual[key]} expected {expected}")
    if audit.failures:
        violations.append(f"{len(audit.failures)} train call(s) raised")
    policy_changed = (bool(audit.initial_hashes) and bool(audit.final_hashes)
                      and audit.initial_hashes["policy_state"] != audit.final_hashes["policy_state"])
    optimizer_changed = (bool(audit.initial_hashes) and bool(audit.final_hashes)
                         and audit.initial_hashes["optimizer_state"] != audit.final_hashes["optimizer_state"])
    actor_mean_changed = (bool(audit.initial_hashes) and bool(audit.final_hashes)
                          and audit.initial_hashes["actor_mean_state"]
                          != audit.final_hashes["actor_mean_state"])
    log_std_changed = (bool(audit.initial_hashes) and bool(audit.final_hashes)
                       and audit.initial_hashes["log_std_state"]
                       != audit.final_hashes["log_std_state"])
    if not policy_changed:
        violations.append("policy state hash did not change")
    if not optimizer_changed:
        violations.append("optimizer state hash did not change")
    log_std = np.asarray(model.policy.log_std.detach().cpu().numpy(), dtype=np.float64)
    receipt = {
        "learning_schema": LEARNING_SCHEMA,
        "status": "complete" if not violations else "incomplete",
        "qualified_for_final_checkpoint": not violations,
        "claims_learning_quality": False,
        "expected": dict(audit.expected),
        "actual": actual,
        "violations": violations,
        "diagnostics": list(audit.diagnostics),
        "construction": dict(audit.construction),
        "initial_hashes": dict(audit.initial_hashes),
        "final_hashes": dict(audit.final_hashes),
        "policy_state_changed": policy_changed,
        "actor_mean_state_changed": actor_mean_changed,
        "log_std_state_changed": log_std_changed,
        "optimizer_state_changed": optimizer_changed,
        "final_log_std": log_std.tolist(),
        "failures": list(audit.failures),
        "updates": list(audit.updates),
        "rollouts": list(audit.rollouts),
        "note": (
            "counts and hashes only; reward level, 1.5 m/s capability and any RL "
            "contribution must come from root's heldout pairs"
        ),
    }
    json.dumps(receipt, sort_keys=True, allow_nan=False)
    return receipt


# --------------------------------------------------------------------------- #
# final checkpoint: exclusive save, strict reload, deterministic probe
# --------------------------------------------------------------------------- #


def _write_exclusive(path: str, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: str) -> None:
    descriptor = os.open(path, os.O_RDWR)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_dir(path: str) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _module_metadata_block(model: Any, probe_shape: tuple[int, int],
                           action_shape: tuple[int, int]) -> dict[str, Any]:
    import stable_baselines3 as sb3
    import torch as th

    return {
        "loader_schema": FINAL_CHECKPOINT_SCHEMA,
        "learning_schema": LEARNING_SCHEMA,
        "observation_size": OBSERVATION_SIZE,
        "action_size": ACTION_SIZE,
        "observation_abs_bound": OBSERVATION_ABS_BOUND,
        "action_abs_bound": ACTION_ABS_BOUND,
        "action_schema": ACTION_SCHEMA,
        "leg_action_scale_rad": np.asarray(LEG_ACTION_SCALE_RAD, dtype=np.float64).tolist(),
        "wheel_action_scale_rad_s": float(WHEEL_ACTION_SCALE_RAD_S),
        "net_arch": {"pi": list(NET_ARCH), "vf": list(NET_ARCH)},
        "log_std_init": LOG_STD_INIT,
        "ppo_seed": PPO_SEED,
        "sb3_version": sb3.__version__,
        "torch_version": th.__version__,
        "numpy_version": np.__version__,
        "probe_observation_shape": list(probe_shape),
        "probe_action_shape": list(action_shape),
    }


def _validate_caller_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise TypeError("checkpoint metadata must be a dict")
    missing = tuple(key for key in REQUIRED_METADATA_KEYS if key not in metadata)
    if missing:
        raise ValueError(f"checkpoint metadata is missing provenance {missing}")
    owned = tuple(key for key in MODULE_OWNED_METADATA_KEYS if key in metadata)
    if owned:
        raise ValueError(f"caller metadata must not pre-empt module keys {owned}")
    control_dt = float(metadata["control_dt_s"])
    physics_dt = float(metadata["physics_dt_s"])
    substeps = metadata["physics_steps_per_control"]
    if not math.isclose(control_dt, CONTROL_DT_S, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("metadata control_dt_s differs from the pure math constant")
    if type(substeps) is not int or substeps != PHYSICS_STEPS_PER_CONTROL:
        raise ValueError("metadata must record the five native substeps per control")
    if not math.isfinite(physics_dt) or physics_dt <= 0.0 or not math.isclose(
            physics_dt * substeps, control_dt, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("metadata physics_dt_s * substeps must equal control_dt_s")
    payload = dict(metadata)
    json.dumps(payload, sort_keys=True, allow_nan=False)
    return payload


def save_final_and_verify(model: Any, folder: str, metadata: dict[str, Any],
                          probe_observations: Any, *, audit: LearningAudit,
                          receipt: dict[str, Any] | None = None) -> dict[str, Any]:
    """Save the one preselected final checkpoint and verify it by reloading.

    ``folder`` must not exist. Nothing is written unless ``training_receipt``
    reports ``complete``. Returns the trusted manifest; root must record it in
    its own ledger, because the on-disk sidecar is never its own trust root.
    """
    if not isinstance(audit, LearningAudit):
        raise TypeError("save_final_and_verify needs the LearningAudit of this run")
    final_receipt = training_receipt(model, audit) if receipt is None else receipt
    if final_receipt.get("status") != "complete":
        raise RuntimeError(
            "refusing to save a final checkpoint: "
            f"{final_receipt.get('violations')}"
        )
    caller_metadata = _validate_caller_metadata(metadata)
    observations = _float32_batch(probe_observations, OBSERVATION_SIZE,
                                  "probe_observations", MAX_PROBE_OBSERVATIONS)
    if np.any(np.abs(observations) > np.float32(OBSERVATION_ABS_BOUND)):
        raise ValueError("probe observations must already lie in the +/-5 box")

    actions, state = model.predict(observations, deterministic=True)
    actions = np.ascontiguousarray(actions, dtype=np.float32)
    if (state is not None or actions.shape != (observations.shape[0], ACTION_SIZE)
            or not np.isfinite(actions).all()):
        raise RuntimeError("deterministic probe prediction has the wrong shape/state")

    os.mkdir(folder, 0o755)
    model_path = os.path.join(folder, MODEL_FILE)
    model.save(model_path, exclude=["learning_audit"])
    if not os.path.isfile(model_path):
        raise RuntimeError("SB3 did not write the expected final_model.zip")
    _fsync_file(model_path)
    _write_exclusive(os.path.join(folder, PROBE_OBS_FILE), observations.tobytes(order="C"))
    _write_exclusive(os.path.join(folder, PROBE_ACTION_FILE), actions.tobytes(order="C"))

    document = dict(caller_metadata)
    document.update(_module_metadata_block(model, observations.shape, actions.shape))
    document["training_receipt"] = final_receipt
    files: dict[str, Any] = {}
    for name in (MODEL_FILE, PROBE_OBS_FILE, PROBE_ACTION_FILE):
        digest, size = _hash_file(os.path.join(folder, name))
        files[name] = {"sha256": digest, "bytes": size}
    document["files"] = files
    payload = json.dumps(document, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
    _write_exclusive(os.path.join(folder, METADATA_FILE), payload)
    _fsync_dir(folder)

    manifest = {
        "loader_schema": FINAL_CHECKPOINT_SCHEMA,
        "folder": os.path.abspath(folder),
        "metadata_sha256": _hash_bytes(payload),
        "files": files,
        "trusted_fields": {
            "loader_schema": FINAL_CHECKPOINT_SCHEMA,
            "learning_schema": LEARNING_SCHEMA,
            "observation_size": OBSERVATION_SIZE,
            "action_size": ACTION_SIZE,
            "action_schema": ACTION_SCHEMA,
            "observation_schema": caller_metadata["observation_schema"],
            "task_schema": caller_metadata["task_schema"],
            "reward_schema": caller_metadata["reward_schema"],
            "controller_schema": caller_metadata["controller_schema"],
            "control_loop_schema": caller_metadata["control_loop_schema"],
            "run_id": caller_metadata["run_id"],
            "ppo_seed": PPO_SEED,
            "wheel_action_scale_rad_s": float(WHEEL_ACTION_SCALE_RAD_S),
            "probe_observation_shape": list(observations.shape),
            "probe_action_shape": list(actions.shape),
        },
    }
    manifest["reload_verification"] = load_and_verify_final(
        folder, manifest, reference_model=model,
    )
    json.dumps(manifest, sort_keys=True, allow_nan=False)
    return manifest


def save_failure_checkpoint(model: Any, folder: str, receipt: dict[str, Any]) -> dict[str, Any]:
    """Store an explicitly unqualified checkpoint for a failed run."""
    if receipt.get("status") == "complete":
        raise ValueError("a complete run must use save_final_and_verify")
    os.mkdir(folder, 0o755)
    path = os.path.join(folder, FAILURE_MODEL_FILE)
    model.save(path, exclude=["learning_audit"])
    _fsync_file(path)
    digest, size = _hash_file(path)
    document = {
        "loader_schema": FINAL_CHECKPOINT_SCHEMA + "-failure",
        "qualified_for_heldout": False,
        "is_final_model": False,
        "training_receipt": receipt,
        "files": {FAILURE_MODEL_FILE: {"sha256": digest, "bytes": size}},
    }
    payload = json.dumps(document, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
    _write_exclusive(os.path.join(folder, FAILURE_METADATA_FILE), payload)
    _fsync_dir(folder)
    return {"folder": os.path.abspath(folder),
            "metadata_sha256": _hash_bytes(payload),
            "files": document["files"],
            "qualified_for_heldout": False}


def _describe_linear_stack(module: Any) -> list[dict[str, Any]]:
    layers = []
    for child in module.children():
        entry: dict[str, Any] = {"type": type(child).__name__}
        if hasattr(child, "in_features"):
            entry["in_features"] = int(child.in_features)
            entry["out_features"] = int(child.out_features)
        layers.append(entry)
    return layers


def load_and_verify_final(folder: str, expected_manifest: dict[str, Any], *,
                         reference_model: Any = None,
                         return_model: bool = False) -> dict[str, Any] | tuple[Any, dict[str, Any]]:
    """Strict new loader for the 16D/99D final checkpoint.

    Trust comes from ``expected_manifest`` (root's recorded copy), not from the
    sidecar. Verifies file hashes, the loaded observation/action spaces, the
    actual network shapes, the pure action scales, and reproduces the stored
    deterministic probe byte-exactly. The default returns a JSON-ready report;
    ``return_model=True`` returns ``(loaded_model, report)`` for heldout use.
    No engine, no env, no reset, no step.
    """
    if not isinstance(expected_manifest, dict):
        raise TypeError("a trusted expected manifest is required")
    if type(return_model) is not bool:
        raise TypeError("return_model must be a bool")
    if expected_manifest.get("loader_schema") != FINAL_CHECKPOINT_SCHEMA:
        raise ValueError("expected manifest is not the new 08-R final schema")
    expected_files = expected_manifest["files"]
    trusted = expected_manifest["trusted_fields"]
    if set(expected_files) != {MODEL_FILE, PROBE_OBS_FILE, PROBE_ACTION_FILE}:
        raise ValueError("expected manifest does not name exactly the three payload files")

    metadata_path = os.path.join(folder, METADATA_FILE)
    with open(metadata_path, "rb") as handle:
        metadata_bytes = handle.read()
    metadata_sha = _hash_bytes(metadata_bytes)
    if metadata_sha != expected_manifest["metadata_sha256"]:
        raise ValueError("final metadata sha256 differs from the trusted manifest")
    document = json.loads(metadata_bytes.decode("utf-8"))
    if document.get("loader_schema") != FINAL_CHECKPOINT_SCHEMA:
        raise ValueError("stored checkpoint declares an unsupported loader schema")
    if (document.get("observation_size") != OBSERVATION_SIZE
            or document.get("action_size") != ACTION_SIZE
            or document.get("action_schema") != ACTION_SCHEMA):
        raise ValueError("stored checkpoint is not the 99D/16D residual schema")
    if not math.isclose(float(document["wheel_action_scale_rad_s"]),
                        float(WHEEL_ACTION_SCALE_RAD_S), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("stored wheel action scale differs from the pure math module")
    if not np.array_equal(np.asarray(document["leg_action_scale_rad"], dtype=np.float64),
                          np.asarray(LEG_ACTION_SCALE_RAD, dtype=np.float64)):
        raise ValueError("stored leg action scales differ from the pure math module")
    for key, value in trusted.items():
        if key not in document:
            raise ValueError(f"stored metadata is missing trusted field {key}")
        if document[key] != value:
            raise ValueError(f"stored metadata field {key} differs from the trusted manifest")
    for name, entry in expected_files.items():
        digest, size = _hash_file(os.path.join(folder, name))
        if digest != entry["sha256"] or size != entry["bytes"]:
            raise ValueError(f"{name} does not match the trusted manifest hash/size")
        stored = document["files"][name]
        if stored["sha256"] != entry["sha256"] or stored["bytes"] != entry["bytes"]:
            raise ValueError(f"{name} sidecar hash disagrees with the trusted manifest")

    rows = int(document["probe_observation_shape"][0])
    with open(os.path.join(folder, PROBE_OBS_FILE), "rb") as handle:
        observations = np.frombuffer(handle.read(), dtype=np.float32).reshape(rows, OBSERVATION_SIZE)
    with open(os.path.join(folder, PROBE_ACTION_FILE), "rb") as handle:
        stored_actions = np.frombuffer(handle.read(), dtype=np.float32).reshape(rows, ACTION_SIZE)
    if not np.isfinite(observations).all() or not np.isfinite(stored_actions).all():
        raise ValueError("stored probe batch is not finite")

    from stable_baselines3 import PPO

    model = PPO.load(os.path.join(folder, MODEL_FILE), env=None, device="cpu")
    observation_box = _describe_box(model.observation_space, OBSERVATION_SIZE,
                                    OBSERVATION_ABS_BOUND, "loaded observation_space")
    action_box = _describe_box(model.action_space, ACTION_SIZE,
                               ACTION_ABS_BOUND, "loaded action_space")
    policy = model.policy
    if type(policy).__name__ != "ActorCriticPolicy" or policy.squash_output or policy.use_sde:
        raise ValueError("loaded policy is not the unsquashed Gaussian ActorCriticPolicy")
    actor_layers = _describe_linear_stack(policy.mlp_extractor.policy_net)
    critic_layers = _describe_linear_stack(policy.mlp_extractor.value_net)
    expected_stack = [
        {"type": "Linear", "in_features": OBSERVATION_SIZE, "out_features": NET_ARCH[0]},
        {"type": "Tanh"},
        {"type": "Linear", "in_features": NET_ARCH[0], "out_features": NET_ARCH[1]},
        {"type": "Tanh"},
    ]
    if actor_layers != expected_stack or critic_layers != expected_stack:
        raise ValueError("loaded actor/critic are not independent [128, 128] Tanh MLPs")
    if (int(policy.action_net.in_features) != NET_ARCH[1]
            or int(policy.action_net.out_features) != ACTION_SIZE
            or int(policy.value_net.out_features) != 1
            or tuple(policy.log_std.shape) != (ACTION_SIZE,)):
        raise ValueError("loaded output heads do not match 16 actions / 1 value")
    loaded_policy_sha = hash_state("policy", policy.state_dict())
    final_hashes = document.get("training_receipt", {}).get("final_hashes", {})
    if (not isinstance(final_hashes, dict)
            or not isinstance(final_hashes.get("policy_state"), str)
            or loaded_policy_sha != final_hashes["policy_state"]):
        raise ValueError("loaded policy hash differs from the trusted training final hash")

    reloaded_actions, reloaded_state = model.predict(observations, deterministic=True)
    reloaded_actions = np.ascontiguousarray(reloaded_actions, dtype=np.float32)
    if reloaded_state is not None:
        raise ValueError("the final policy must be stateless")
    if reloaded_actions.tobytes(order="C") != stored_actions.tobytes(order="C"):
        raise ValueError("reloaded deterministic probe actions are not byte-identical")
    reference_match = None
    if reference_model is not None:
        original_actions, _ = reference_model.predict(observations, deterministic=True)
        original_actions = np.ascontiguousarray(original_actions, dtype=np.float32)
        reference_match = (original_actions.tobytes(order="C")
                           == stored_actions.tobytes(order="C"))
        if not reference_match:
            raise ValueError("in-memory policy and stored probe actions disagree")

    report = {
        "loader_schema": FINAL_CHECKPOINT_SCHEMA,
        "metadata_sha256": metadata_sha,
        "observation_space": observation_box,
        "action_space": action_box,
        "actor_layers": actor_layers,
        "critic_layers": critic_layers,
        "log_std_shape": [ACTION_SIZE],
        "probe_rows": rows,
        "probe_actions_byte_exact": True,
        "reference_policy_byte_exact": reference_match,
        "loaded_policy_state_sha256": loaded_policy_sha,
        "matches_training_final_policy_state": True,
        "verified_without_engine_or_reset": True,
    }
    return (model, report) if return_model else report
