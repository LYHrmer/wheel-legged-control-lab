"""85-dimensional heading-reference task on the frozen D1 locomotion control loop.

Original module authored by Claude Opus (Anthropic) as the core implementation of the
GPT-6-astra / ultra specification dated 2026-09-14; root-thread integration, review and
any local fixes are archived separately from this original response.

Scope of this module
--------------------
This file contains exactly two public entry points:

* :class:`D1HeadingTrackingEnv` -- a *thin* subclass of the frozen
  :class:`~wheel_legged_control.d1.locomotion_env.D1LocomotionEnv`.  It adds a
  simulation-time heading reference, a heading servo on the *yaw-rate channel only*, three
  appended observation dimensions and one bounded non-positive reward term.  Base
  ``reset``/``step``/dynamics/reward/execution chains are **not** copied: only ``_prepare``
  is overridden, and ``reset``/``step`` extend what ``super()`` already returned.
* :func:`load_heading_policy` -- a thin loader layer that compares the sidecar's recorded
  ``heading_task_config`` with the validated runtime configuration *before* deserializing
  a model, then delegates to the frozen ``load_locomotion_policy``.

No training loop, no curriculum, no PI outer loop, no action-space change, no cross-track
term and no terrain preview live here.  The physical scales, low-level controller,
actuators, action schema (``independent8``, ``float32`` Box ``[-1, 1]^8``) and the frozen
82-dimensional proprioceptive/servo encoding are untouched.

Timing contract (left-hold / zero-order hold)
---------------------------------------------
``_prepare`` consumes the *user* command exactly once per (loop, tick), reads the
already-published same-tick state cache, integrates the reference with the rate held over
the previous executed interval, computes the heading servo yaw rate, and only then calls
``loop.prepare``.  Repeating ``_prepare`` for an already successfully prepared (loop, tick)
returns the cached decision and does **not** consume the command callback again.

Reward
------
``heading_goal = actual_dt * w * (exp(-(wrap(psi_truth_end - psi_ref_end) / sigma)^2) - 1)``
with ``psi_ref_end = wrap(psi_ref_before + actual_dt * user_yaw_rate_before)`` and
``actual_dt = receipt.end_time_s - receipt.start_time_s``.  The term lies in
``[-actual_dt * w, 0]`` and is exactly zero at zero heading error.  Every base servo
reward term is preserved unchanged; ``info['servo_reward_terms']`` / ``info['servo_reward']``
keep the base decomposition, ``info['reward_terms']`` decomposes the returned reward.

Truth (``last_transition.truth``) is used for reward/metrics only.  The three appended
actor dimensions are computed from the *published state estimate* named by
``source_schema``, so ground truth cannot leak into the policy input.
"""

from __future__ import annotations

import json
import math
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
from numbers import Real
from pathlib import Path

import numpy as np
from gymnasium import spaces

from wheel_legged_control.d1.control_loop import D1MotionCommand
from wheel_legged_control.d1.locomotion_checkpoint import load_locomotion_policy
from wheel_legged_control.d1.locomotion_env import D1LocomotionEnv
from wheel_legged_control.d1.locomotion_observation import (
    LOCOMOTION_OBSERVATION_SIZE,
    encode_d1_locomotion_observation,
)

try:  # the heading reference lives next to this file in scripts/
    from d1_heading_reference import (
        HEADING_REFERENCE_SCHEMA,
        HeadingReference,
        heading_feedback,
        wrap_angle,
    )
except ModuleNotFoundError:  # pragma: no cover - import layout fallback only
    try:
        from scripts.d1_heading_reference import (
            HEADING_REFERENCE_SCHEMA,
            HeadingReference,
            heading_feedback,
            wrap_angle,
        )
    except ModuleNotFoundError:
        _SCRIPTS_DIR = str(Path(__file__).resolve().parent)
        if _SCRIPTS_DIR not in sys.path:
            sys.path.insert(0, _SCRIPTS_DIR)
        from d1_heading_reference import (
            HEADING_REFERENCE_SCHEMA,
            HeadingReference,
            heading_feedback,
            wrap_angle,
        )

__all__ = [
    "HEADING_OBSERVATION_APPENDED_FIELDS",
    "HEADING_OBSERVATION_CLIP",
    "HEADING_OBSERVATION_SCHEMA",
    "HEADING_OBSERVATION_SIZE",
    "HEADING_REWARD_FORMULA_VERSION",
    "HEADING_REWARD_SCHEMA",
    "HEADING_REWARD_TERM_NAME",
    "HEADING_TASK_CONFIG_SCHEMA",
    "HEADING_TASK_SCHEMA",
    "HEADING_TIMING_SCHEMA",
    "HEADING_USER_YAW_RATE_SCALE_RPS",
    "D1HeadingTrackingEnv",
    "load_heading_policy",
]

# --- frozen task identity -------------------------------------------------------------
HEADING_TASK_SCHEMA = "d1-heading-reference-task-v1"
HEADING_OBSERVATION_SCHEMA = "d1-proprio-servo82-heading85-v1"
HEADING_REWARD_SCHEMA = "d1-heading-goal-servo-rate-v1"
HEADING_TASK_CONFIG_SCHEMA = "d1-heading-task-config-v1"
HEADING_TIMING_SCHEMA = "d1-heading-left-hold-prepare-servo-yaw-rate-v1"
HEADING_REWARD_FORMULA_VERSION = (
    "heading_goal = actual_dt * w * (exp(-(wrap(psi_truth_end - psi_ref_end) / sigma)^2) - 1); "
    "psi_ref_end = wrap(psi_ref_before + actual_dt * user_yaw_rate_before); v1"
)
HEADING_REWARD_TERM_NAME = "heading_goal"

# --- frozen outer-loop gains (heading hold verified in the 14-episode probe) -----------
HEADING_OUTER_LOOP_KP = 2.0
HEADING_OUTER_LOOP_KD = 0.4
HEADING_OUTER_LOOP_LIMIT_RPS = 1.0

# --- appended observation contract ----------------------------------------------------
HEADING_OBSERVATION_APPENDED_FIELDS = (
    "sin_heading_error",
    "cos_heading_error",
    "user_yaw_rate_normalized",
)
HEADING_USER_YAW_RATE_SCALE_RPS = 0.5
HEADING_OBSERVATION_CLIP = (-5.0, 5.0)

if LOCOMOTION_OBSERVATION_SIZE != 82:  # pragma: no cover - guards a frozen-source change
    raise RuntimeError(
        "d1_heading_tracking_env assumes the frozen 82-dimensional locomotion observation, "
        f"found {LOCOMOTION_OBSERVATION_SIZE}"
    )
HEADING_OBSERVATION_SIZE = LOCOMOTION_OBSERVATION_SIZE + len(HEADING_OBSERVATION_APPENDED_FIELDS)

_CACHED_TERMINAL_SOURCE = "last_executable_decision"
_NEXT_DECISION_SOURCE = "next_prepared_decision"
_YAW_RATE_FIELD = "yaw_rate_rps"


def _finite_scalar(name: str, value: object) -> float:
    """Validate ``value`` as a finite real scalar (bools rejected) and return a float."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar, not bool")
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def _state_heading(state) -> tuple[float, float]:
    """Read (yaw, body yaw rate) from a published state estimate, without touching truth."""
    rpy = np.asarray(state.base_rpy, dtype=np.float64)
    omega = np.asarray(state.base_angular_velocity_body, dtype=np.float64)
    if rpy.shape != (3,) or omega.shape != (3,):
        raise ValueError("state publication must expose base_rpy and base_angular_velocity_body")
    yaw, yaw_rate = float(rpy[2]), float(omega[2])
    if not (math.isfinite(yaw) and math.isfinite(yaw_rate)):
        raise ValueError("measured heading and yaw rate must be finite")
    return yaw, yaw_rate


def _servo_command(user: D1MotionCommand, servo_yaw_rate_rps: float) -> D1MotionCommand:
    """Replace only the yaw-rate channel; forward velocity and clearance stay the user's."""
    servo = replace(user, **{_YAW_RATE_FIELD: float(servo_yaw_rate_rps)})
    for item in fields(user):
        if item.name == _YAW_RATE_FIELD:
            continue
        if not np.array_equal(getattr(servo, item.name), getattr(user, item.name)):
            raise RuntimeError(
                "heading servo substitution must change the yaw rate only, "
                f"but {item.name} differs"
            )
    return servo


def _canonical(payload: object) -> str:
    """Canonical JSON text; distinguishes True from 1 and 1 from 1.0."""
    return json.dumps(payload, sort_keys=True, allow_nan=False)


@dataclass(frozen=True, slots=True)
class _HeadingDecision:
    """Immutable per-decision heading context registered by a successful ``_prepare``."""

    loop: object
    tick: int
    control_time_s: float
    user_command: D1MotionCommand
    servo_command: D1MotionCommand
    reference_time_s: float
    reference_heading_rad: float
    reference_held_yaw_rate_rps: float
    measured_heading_rad: float
    measured_yaw_rate_rps: float
    heading_error_rad: float
    unclipped_yaw_rate_rps: float
    saturated: bool
    state_time_s: float
    state_age_s: float
    decision: object

    @property
    def user_yaw_rate_rps(self) -> float:
        return float(getattr(self.user_command, _YAW_RATE_FIELD))


class D1HeadingTrackingEnv(D1LocomotionEnv):
    """The frozen locomotion task plus an observable, rewarded heading reference.

    The user command (schedule or external callback) is the *operator* intent.  It is
    integrated into a heading target by an explicit simulation-time reference, and a
    P/D heading feedback term converts that target into the yaw rate actually handed to
    the low-level controller.  The actor sees the frozen 82-dimensional servo encoding
    followed by ``[sin(psi_ref - psi_est), cos(psi_ref - psi_est), user_yaw_rate / 0.5]``.

    The new task/observation/reward schemas make an old 82-dimensional checkpoint fail to
    load: nothing is zero-padded or truncated to work around that check.
    """

    observation_schema = HEADING_OBSERVATION_SCHEMA
    reward_schema = HEADING_REWARD_SCHEMA
    task_schema = HEADING_TASK_SCHEMA

    def __init__(
        self,
        *,
        episode_seconds: float = 60.0,
        terrain=None,
        command_mode: str = "random",
        provider_config=None,
        actuator_config=None,
        randomization=None,
        reward_config=None,
        command_source=None,
        wheel_leg_control=None,
        heading_reward_weight: float = 0.5,
        heading_sigma_rad: float = math.radians(5.0),
        baseline: str = "wheel_leg",
        action_mode: str = "independent8",
    ):
        if baseline != "wheel_leg":
            raise ValueError("the heading task is fixed to the wheel_leg baseline")
        if action_mode != "independent8":
            raise ValueError("the heading task is fixed to the independent8 action mode")
        weight = _finite_scalar("heading_reward_weight", heading_reward_weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"heading_reward_weight must lie within [0, 1], got {weight!r}")
        sigma = _finite_scalar("heading_sigma_rad", heading_sigma_rad)
        if not 1e-3 <= sigma <= math.pi:
            raise ValueError(f"heading_sigma_rad must lie within [1e-3, pi], got {sigma!r}")
        self.heading_reward_weight = weight
        self.heading_sigma_rad = sigma
        self.heading_kp = HEADING_OUTER_LOOP_KP
        self.heading_kd = HEADING_OUTER_LOOP_KD
        self.heading_limit_rps = HEADING_OUTER_LOOP_LIMIT_RPS
        # Reference/decision bookkeeping must exist before the base constructor can call
        # anything on this instance.
        self._reference = HeadingReference()
        self._reference_loop = None
        self._reference_initial: dict | None = None
        self._heading_context: _HeadingDecision | None = None
        super().__init__(
            baseline="wheel_leg",
            episode_seconds=episode_seconds,
            terrain=terrain,
            command_mode=command_mode,
            provider_config=provider_config,
            actuator_config=actuator_config,
            randomization=randomization,
            reward_config=reward_config,
            command_source=command_source,
            wheel_leg_control=wheel_leg_control,
            action_mode="independent8",
        )
        if self.baseline_name != "wheel_leg" or self.action_mode != "independent8":
            raise RuntimeError("base environment resolved an unexpected baseline/action mode")
        if self.policy_action_size != 8 or self.physical_action_size != 8:
            raise RuntimeError("the heading task requires the unchanged eight-dimensional action")
        low, high = HEADING_OBSERVATION_CLIP
        self.observation_space = spaces.Box(
            low, high, (HEADING_OBSERVATION_SIZE,), dtype=np.float32
        )
        self._heading_task_config = self._build_heading_task_config()

    # -- configuration ----------------------------------------------------------------
    def _build_heading_task_config(self) -> dict:
        """The comparable task definition: gains, reward, schemas, observation, timing.

        Deliberately excludes the episode's actual reference initial state, terrain and
        seeds, so a matching policy can still be evaluated on a new scenario.  Those live
        in separate episode-metadata fields.
        """
        low, high = HEADING_OBSERVATION_CLIP
        config = {
            "schema": HEADING_TASK_CONFIG_SCHEMA,
            "task_schema": HEADING_TASK_SCHEMA,
            "observation_schema": HEADING_OBSERVATION_SCHEMA,
            "reward_schema": HEADING_REWARD_SCHEMA,
            "reference_schema": HEADING_REFERENCE_SCHEMA,
            "baseline": "wheel_leg",
            "action_mode": "independent8",
            "outer_loop": {
                "kp": float(self.heading_kp),
                "kd": float(self.heading_kd),
                "limit_rps": float(self.heading_limit_rps),
                "feed_forward": "user_yaw_rate_rps",
                "servo_substitution": "yaw_rate_only",
            },
            "reward": {
                "heading_reward_weight": float(self.heading_reward_weight),
                "heading_sigma_rad": float(self.heading_sigma_rad),
                "term_name": HEADING_REWARD_TERM_NAME,
                "bounds": "[-actual_dt * heading_reward_weight, 0]",
                "formula_version": HEADING_REWARD_FORMULA_VERSION,
                "target": "simulation_time_heading_reference_endpoint",
                "truth_source": "last_transition.truth",
            },
            "observation": {
                "base_size": int(LOCOMOTION_OBSERVATION_SIZE),
                "total_size": int(HEADING_OBSERVATION_SIZE),
                "appended_fields": list(HEADING_OBSERVATION_APPENDED_FIELDS),
                "user_yaw_rate_scale_rps": float(HEADING_USER_YAW_RATE_SCALE_RPS),
                "clip_low": float(low),
                "clip_high": float(high),
                "dtype": "float32",
                "heading_error_source": "published_state_estimate",
            },
            "timing_schema": HEADING_TIMING_SCHEMA,
            "terminal_reference_handling": (
                "cached-terminal-observation-uses-pre-step-heading-context-v1"
            ),
        }
        _canonical(config)  # fail loudly here if a non-JSON value ever sneaks in
        return config

    @property
    def heading_task_config(self) -> dict:
        """A copy of the validated, comparable heading task configuration."""
        return deepcopy(self._heading_task_config)

    @property
    def heading_decision(self) -> _HeadingDecision | None:
        """The immutable context of the currently prepared decision (or None)."""
        return self._heading_context

    # -- minimal inheritance seam ------------------------------------------------------
    def _prepare(self):
        """Prepare one tick: user command -> reference -> heading servo -> loop.prepare.

        Returns the *frozen 82-dimensional* encoding; the appended three dimensions are
        added by ``reset``/``step`` from the registered context.
        """
        loop = self.loop
        if loop is None:
            raise RuntimeError("the control loop must exist before a decision is prepared")
        tick = int(self._steps)
        cached = self._heading_context
        if cached is not None and cached.loop is loop and cached.tick == tick:
            # Repeated prepare for an already successful decision: reuse it and do not
            # consume the command callback or republish the reference again.
            self.decision = cached.decision
            return encode_d1_locomotion_observation(cached.decision)

        time_s = tick * self.plant.control_dt
        if loop is not self._reference_loop:
            # A freshly reset loop: seed the reference from the *measured* initial yaw at
            # t = 0, after the base has already validated its reset options.
            initial_state = loop.provider.read()
            initial_yaw, initial_rate = _state_heading(initial_state)
            initial = self._reference.reset(initial_yaw, 0.0)
            self._reference_loop = loop
            self._heading_context = None
            self._reference_initial = {
                "reference_schema": HEADING_REFERENCE_SCHEMA,
                "source_schema": self.source_schema,
                "simulation_time_s": float(initial.simulation_time_s),
                "heading_rad": float(initial.heading_rad),
                "held_user_yaw_rate_rps": float(initial.user_yaw_rate_rps),
                "measured_heading_rad": float(initial_yaw),
                "measured_yaw_rate_rps": float(initial_rate),
                "measurement_age_s": float(initial_state.age_s),
            }

        user = (
            self.command_source(time_s)
            if self.command_source is not None
            else self.schedule.cmd_at(time_s)
        )
        if not isinstance(user, D1MotionCommand):
            raise TypeError("the user command must be a D1MotionCommand")
        user_rate = _finite_scalar("user yaw rate", getattr(user, _YAW_RATE_FIELD))

        state = loop.provider.read()
        measured_yaw, measured_rate = _state_heading(state)
        # Left hold: integrate the rate held over the interval just executed, then latch
        # this tick's user rate for the next interval.
        reference = self._reference.advance(user_rate, time_s)
        feedback = heading_feedback(
            reference.heading_rad,
            measured_yaw,
            measured_rate,
            user_rate,
            kp=self.heading_kp,
            kd=self.heading_kd,
            limit_rps=self.heading_limit_rps,
        )
        servo = _servo_command(user, feedback.servo_yaw_rate_rps)

        decision = loop.prepare(servo)  # may raise: no context is registered then
        self.decision = decision
        self._heading_context = _HeadingDecision(
            loop=loop,
            tick=tick,
            control_time_s=float(time_s),
            user_command=user,
            servo_command=servo,
            reference_time_s=float(reference.simulation_time_s),
            reference_heading_rad=float(reference.heading_rad),
            reference_held_yaw_rate_rps=float(reference.user_yaw_rate_rps),
            measured_heading_rad=float(measured_yaw),
            measured_yaw_rate_rps=float(measured_rate),
            heading_error_rad=float(feedback.heading_error_rad),
            unclipped_yaw_rate_rps=float(feedback.unclipped_yaw_rate_rps),
            saturated=bool(feedback.saturated),
            state_time_s=float(state.control_time_s),
            state_age_s=float(state.age_s),
            decision=decision,
        )
        return encode_d1_locomotion_observation(decision)

    def _extend(self, observation, context: _HeadingDecision) -> np.ndarray:
        """Append the three heading dimensions to a frozen 82-dimensional encoding."""
        base = np.asarray(observation)
        if base.shape != (LOCOMOTION_OBSERVATION_SIZE,) or base.dtype != np.float32:
            raise RuntimeError(
                "the base task must return the frozen 82-dimensional float32 encoding"
            )
        error = context.heading_error_rad
        appended = np.array(
            (
                math.sin(error),
                math.cos(error),
                context.user_yaw_rate_rps / HEADING_USER_YAW_RATE_SCALE_RPS,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(appended).all():
            raise ValueError("the appended heading observation must be finite")
        low, high = HEADING_OBSERVATION_CLIP
        appended = np.clip(appended, low, high)
        result = np.concatenate(
            (base.astype(np.float32, copy=True), appended.astype(np.float32))
        ).astype(np.float32, copy=False)
        if result.shape != (HEADING_OBSERVATION_SIZE,) or not np.isfinite(result).all():
            raise RuntimeError("the heading observation must be a finite 85-dimensional vector")
        return result

    def _require_context(self, where: str) -> _HeadingDecision:
        context = self._heading_context
        if context is None or self.loop is None or context.loop is not self.loop:
            raise RuntimeError(f"{where} requires a successfully prepared heading decision")
        return context

    # -- Gym API ----------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        """Base reset (validation, RNG, spawn, metadata) first; then extend to 85 dims."""
        observation, info = super().reset(seed=seed, options=options)
        context = self._require_context("reset")
        if context.tick != 0:
            raise RuntimeError("reset must register the initial prepared decision at tick 0")
        if self._reference_initial is None:
            raise RuntimeError("reset did not record the heading reference initial state")
        metadata = self._episode_metadata
        for key in ("heading_task_config", "heading_reference_initial"):
            if key in metadata:
                raise RuntimeError(f"episode metadata already carries {key}")
        metadata["heading_task_config"] = self.heading_task_config
        metadata["heading_reference_initial"] = deepcopy(self._reference_initial)
        result = dict(info)
        result["episode_metadata"] = self.episode_metadata
        result["heading_reference_initial"] = deepcopy(self._reference_initial)
        return self._extend(observation, context), result

    def step(self, action):
        """Base step once; then extend the observation and add the heading reward term."""
        before = self._require_context("step")
        observation, servo_reward, terminated, truncated, info = super().step(action)

        terminal_source = info.get("terminal_observation_source")
        cached_terminal = terminal_source == _CACHED_TERMINAL_SOURCE
        if cached_terminal:
            # The base rejected the next reference domain and returned the *last
            # executable* 82-dimensional decision: the appended dimensions must come from
            # the pre-step context, never from the already advanced reference.
            context = before
            if not terminated or truncated:
                raise RuntimeError("a cached terminal observation must terminate the episode")
        else:
            context = self._heading_context
            if context is None or context.loop is not self.loop or context.tick != before.tick + 1:
                raise RuntimeError("step did not register the next prepared heading decision")

        observation = self._extend(observation, context)

        transition = self.last_transition
        if transition is None:
            raise RuntimeError("step must publish the completed transition")
        receipt = transition.receipt
        actual_dt = _finite_scalar(
            "actual_dt_s", float(receipt.end_time_s) - float(receipt.start_time_s)
        )
        if actual_dt <= 0.0:
            raise RuntimeError("the completed control interval must be positive")

        user_rate_before = before.user_yaw_rate_rps
        psi_ref_before = before.reference_heading_rad
        psi_ref_end = wrap_angle(psi_ref_before + actual_dt * user_rate_before)
        psi_truth_end = float(np.asarray(transition.truth.base_rpy, dtype=np.float64)[2])
        heading_error_after = wrap_angle(psi_truth_end - psi_ref_end)
        heading_goal = _finite_scalar(
            "heading_goal",
            actual_dt
            * self.heading_reward_weight
            * (math.exp(-((heading_error_after / self.heading_sigma_rad) ** 2)) - 1.0),
        )
        if not -actual_dt * self.heading_reward_weight - 1e-12 <= heading_goal <= 1e-12:
            raise RuntimeError("the heading reward term left its bounded non-positive range")

        servo_terms = dict(info["reward_terms"])
        if HEADING_REWARD_TERM_NAME in servo_terms:
            raise RuntimeError(
                f"the base reward already publishes a {HEADING_REWARD_TERM_NAME!r} term"
            )
        servo_total = float(sum(servo_terms.values()))
        if not math.isclose(servo_total, float(servo_reward), rel_tol=1e-9, abs_tol=1e-9):
            raise RuntimeError("the base reward differs from the sum of its terms")
        reward_terms = {**servo_terms, HEADING_REWARD_TERM_NAME: heading_goal}
        reward = float(sum(reward_terms.values()))

        truth_yaw_rate_end = float(
            np.asarray(transition.truth.base_angular_velocity_body, dtype=np.float64)[2]
        )
        heading_task = {
            "timing_schema": HEADING_TIMING_SCHEMA,
            "reward_formula_version": HEADING_REWARD_FORMULA_VERSION,
            "reference_schema": HEADING_REFERENCE_SCHEMA,
            "source_schema": self.source_schema,
            "decision_tick_before": before.tick,
            "decision_time_before_s": before.control_time_s,
            "user_command_before": asdict(before.user_command),
            "servo_command_before": asdict(before.servo_command),
            "reference_heading_before": psi_ref_before,
            "reference_heading_after": psi_ref_end,
            "reference_time_before_s": before.reference_time_s,
            "heading_error_before": before.heading_error_rad,
            "heading_error_after": heading_error_after,
            "user_yaw_rate_before_rps": user_rate_before,
            "user_yaw_rate_error_after": truth_yaw_rate_end - user_rate_before,
            "servo_yaw_rate_saturated_before": before.saturated,
            "servo_unclipped_yaw_rate_before_rps": before.unclipped_yaw_rate_rps,
            "truth_heading_after": psi_truth_end,
            "truth_yaw_rate_after_rps": truth_yaw_rate_end,
            "actual_dt_s": actual_dt,
            "heading_goal_reward": heading_goal,
            "heading_target_truth_source": "last_transition.truth",
            "appended_observation_source": (
                _CACHED_TERMINAL_SOURCE if cached_terminal else _NEXT_DECISION_SOURCE
            ),
            "appended_observation_decision_tick": context.tick,
            "appended_observation_decision_time_s": context.control_time_s,
            "appended_observation_state_time_s": context.state_time_s,
            "appended_observation_state_age_s": context.state_age_s,
            "appended_observation_heading_error_rad": context.heading_error_rad,
            "appended_observation_user_yaw_rate_rps": context.user_yaw_rate_rps,
            "appended_observation_fields": list(HEADING_OBSERVATION_APPENDED_FIELDS),
            "terminal_observation_source": terminal_source,
            "bootstrap_next_observation": bool(not terminated and not cached_terminal),
        }

        result = dict(info)
        result["servo_reward_terms"] = servo_terms
        result["servo_reward"] = servo_total
        result["reward_terms"] = reward_terms
        result["heading_task"] = heading_task
        return observation, reward, terminated, truncated, result


def load_heading_policy(model_path, metadata_path, env):
    """Reject a mismatched heading task *before* deserializing a trusted local model.

    ``load_locomotion_policy`` compares baseline/schemas/spaces/hashes but knows nothing
    about the heading outer-loop gains, the reward weight/sigma or the appended
    observation contract, so this layer compares the sidecar's recorded
    ``heading_task_config`` with the validated runtime configuration first.  Comparison is
    canonical-JSON equality, which distinguishes ``true`` from ``1`` and ``1`` from ``1.0``.
    The episode's actual reference initial state, terrain and seeds are intentionally not
    part of the comparison: evaluating a matching policy on a new scenario stays legal.
    """
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    if not isinstance(metadata, dict):
        raise TypeError("checkpoint metadata must be an object")
    episode = metadata.get("recorded_episode")
    if not isinstance(episode, dict):
        raise ValueError("checkpoint metadata must record its episode settings")
    recorded = episode.get("heading_task_config")
    if recorded is None:
        raise ValueError(
            "checkpoint recorded_episode.heading_task_config is missing: the sidecar "
            "predates the heading task and cannot be loaded into it"
        )
    if not isinstance(recorded, dict):
        raise ValueError("checkpoint heading_task_config must be an object")

    try:
        expected = env.get_wrapper_attr("heading_task_config")
    except AttributeError as exc:
        raise TypeError("env does not expose heading_task_config") from exc
    if not isinstance(expected, dict):
        raise TypeError("heading_task_config must be a dict")
    if _canonical(recorded) != _canonical(expected):
        raise ValueError("checkpoint heading_task_config is incompatible with the environment")

    return load_locomotion_policy(model_path, metadata_path, env)
