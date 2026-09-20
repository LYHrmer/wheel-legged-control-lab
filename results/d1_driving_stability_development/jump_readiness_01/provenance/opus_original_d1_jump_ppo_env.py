"""Thin residual-capable environment for the scaffolded hop PPO task.

Only the new task logic (raw command callback, 95-entry observation, fixed reward,
progress accounting) and the plant *identity* change here.  The frozen parent
reset/step/integrator implementations are reused, never copied, and the parent is
stepped exactly once per decision.  All geometry comes from the approved
read-only helper :mod:`scripts.d1_jump_readiness`; this module adds no competing
geometry interface and performs no integration of its own.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces

from wheel_legged_control.d1.locomotion_checkpoint import load_locomotion_policy
from wheel_legged_control.d1.model import LEG_PREFIXES

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:  # pragma: no cover - import layout only
    sys.path.insert(0, _SCRIPTS_DIR)

from d1_flat_plane_env import D1FlatPlanePlant  # noqa: E402
from d1_heading_tracking_env import (  # noqa: E402
    HEADING_OBSERVATION_SIZE,
    D1HeadingTrackingEnv,
)
from d1_jump_ppo_task import (  # noqa: E402
    JUMP_ACTION_SIZE,
    JUMP_CONTROL_DT_S,
    JUMP_HORIZON_TICKS,
    JUMP_LEG_ORDER,
    JUMP_OBSERVATION_CLIP,
    JUMP_OBSERVATION_SCHEMA,
    JUMP_OBSERVATION_SIZE,
    JUMP_PARENT_OBSERVATION_SIZE,
    JUMP_REWARD_SCHEMA,
    JUMP_TASK_CONFIG_SCHEMA,
    JUMP_TASK_SCHEMA,
    JumpEndpointMetrics,
    JumpEpisodeSpec,
    JumpProgress,
    append_jump_observation,
    compose_jump_reward,
    jump_task_definition,
    phase_at_tick,
    raw_command_at_tick,
)
from d1_jump_readiness import (  # noqa: E402
    bind_wheel_plane_geometry,
    sample_wheel_clearance,
)

__all__ = ["D1JumpPPOEnv", "load_jump_policy", "ordered_wheel_body_ids"]

_MASS_TOLERANCE_KG = 1e-9


def ordered_wheel_body_ids(plant: Any) -> tuple[int, int, int, int]:
    """Adapt the plant's leg-keyed wheel bodies into canonical ``LEG_ORDER`` ints.

    ``bind_wheel_plane_geometry`` requires four ordered plain integers and
    explicitly rejects a mapping, so the actual mapping/sequence exposed by the
    plant is flattened here in the canonical order without renaming anything.
    """
    if tuple(LEG_PREFIXES) != tuple(JUMP_LEG_ORDER):  # pragma: no cover - guard
        raise RuntimeError("the canonical leg order must match the original model order")
    source = plant.wheel_body_ids_by_leg
    if hasattr(source, "keys"):
        missing = [leg for leg in JUMP_LEG_ORDER if leg not in source]
        if missing:
            raise ValueError(f"the plant does not publish wheel bodies for legs {missing}")
        values = [source[leg] for leg in JUMP_LEG_ORDER]
    else:
        values = list(source)
    if len(values) != len(JUMP_LEG_ORDER):
        raise ValueError(
            f"expected {len(JUMP_LEG_ORDER)} wheel body IDs in order {JUMP_LEG_ORDER}"
        )
    ordered = []
    for leg, value in zip(JUMP_LEG_ORDER, values, strict=True):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"wheel body ID of leg {leg} must be an integer, got {value!r}")
        ordered.append(int(value))
    if len(set(ordered)) != len(ordered):
        raise ValueError(f"wheel body IDs must be distinct, got {ordered}")
    return tuple(ordered)  # type: ignore[return-value]


class D1JumpPPOEnv(D1HeadingTrackingEnv):
    """Residual hop environment: heading parent, flat native plane, 95 observations.

    The constructor owns the fixed raw command callback (forward/yaw exactly zero
    plus the scheduled height), so no user callback, residual policy, release
    governor or controller injection is accepted.  The parent builds the original
    residual-capable controller; immediately afterwards its unstepped heightfield
    plant is replaced by :class:`D1FlatPlanePlant` carrying the same actuator
    channel, control period and ground friction, before any reset or loop exists.
    """

    def __init__(
        self,
        *,
        episode_spec: JumpEpisodeSpec | None = None,
        ground_friction: float | None = None,
        **heading_kwargs: Any,
    ) -> None:
        """Construct the jump environment around the original heading parent."""
        for reserved in ("command_source", "schedule"):
            if reserved in heading_kwargs:
                raise TypeError(
                    f"{reserved!r} is owned by the jump task; an arbitrary command "
                    "callback or schedule is not accepted"
                )
        baseline = heading_kwargs.pop("baseline", "wheel_leg")
        if baseline != "wheel_leg":
            raise ValueError("the jump task requires the original wheel_leg baseline")
        action_mode = heading_kwargs.pop("action_mode", "independent8")
        if action_mode != "independent8":
            raise ValueError("the jump task requires the independent8 action mode")
        spec = JumpEpisodeSpec(None, 0.0) if episode_spec is None else episode_spec
        if not isinstance(spec, JumpEpisodeSpec):
            raise TypeError("episode_spec must be a JumpEpisodeSpec")

        self._jump_spec = spec
        self._pending_spec: JumpEpisodeSpec | None = None
        self._progress = JumpProgress()
        self._initial_origin = np.zeros(2, dtype=np.float64)
        self._origin_recorded = False
        self._last_geometry: dict[str, Any] | None = None
        self._last_metrics: JumpEndpointMetrics | None = None
        self._last_reward_breakdown: Any = None
        self._last_progress_event: Any = None
        self._plant_step_original = None
        self._binding = None
        self._binding_receipt: dict[str, Any] = {}
        self._com_jacobian: np.ndarray | None = None

        super().__init__(
            baseline=baseline,
            action_mode=action_mode,
            command_source=self._raw_command_callback,
            **heading_kwargs,
        )

        if abs(float(self.control_dt) - JUMP_CONTROL_DT_S) > 1e-12:
            raise ValueError("the jump task requires the original 10 ms control period")
        if int(self.max_ticks) != JUMP_HORIZON_TICKS:
            raise ValueError("the jump task requires the fixed 6 s / 600 tick horizon")

        self._replace_plant_with_flat_plane(ground_friction)
        self._bind_geometry()
        self._install_contact_readout()

        self.observation_space = spaces.Box(
            low=JUMP_OBSERVATION_CLIP[0],
            high=JUMP_OBSERVATION_CLIP[1],
            shape=(JUMP_OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        if self.action_space.shape != (JUMP_ACTION_SIZE,):
            raise ValueError("the original independent8 action space must stay eight-dimensional")

    # -- construction helpers ---------------------------------------------------------
    def _replace_plant_with_flat_plane(self, ground_friction: float | None) -> None:
        """Swap the unstepped heightfield plant for the native flat plane."""
        old_plant = self.plant
        if float(old_plant.data.time) != 0.0:
            raise RuntimeError("the parent plant must be unstepped before replacement")
        if self.loop is not None or self.decision is not None or self._active:
            raise RuntimeError("no loop, decision or active episode may exist yet")
        friction = (
            float(old_plant.model.geom_friction[old_plant.floor_geom_id][0])
            if ground_friction is None
            else float(ground_friction)
        )
        plant = D1FlatPlanePlant(
            control_dt=float(old_plant.control_dt),
            ground_friction=friction,
            actuator_channel=old_plant.actuator_channel,
        )
        plant.validate_flat_plane()
        self.plant = plant
        self._flat_plane_metadata = dict(plant.collision_terrain_metadata())

    def _bind_geometry(self) -> None:
        """Bind the wheel/plane geometry once and keep its receipt."""
        plant = self.plant
        self._binding = bind_wheel_plane_geometry(
            plant.model,
            ordered_wheel_body_ids(plant),
            plane_geom_id=int(plant.floor_geom_id),
        )
        self._binding_receipt = dict(self._binding.as_receipt())
        masses = np.asarray(plant.model.body_mass, dtype=np.float64)
        subtree = np.asarray(plant.measurement_data.subtree_com, dtype=np.float64)
        if subtree.shape[0] < 1:  # pragma: no cover - guard
            raise RuntimeError("the model must publish subtree COM rows")
        total = float(masses.sum()) - float(masses[0])
        subtree_mass = float(np.asarray(plant.model.body_subtreemass, dtype=np.float64)[
            int(plant.base_body_id)
        ])
        if abs(subtree_mass - total) > 1e-6:
            raise RuntimeError(
                "the base subtree must contain the whole robot mass before COM velocity "
                f"may use mj_jacSubtreeCom: {subtree_mass} vs {total} kg"
            )
        self._robot_mass_kg = total
        self._com_jacobian = np.zeros((3, plant.model.nv), dtype=np.float64)

    def _install_contact_readout(self) -> None:
        """Install the instance-only plant.step wrapper forcing contact readout."""
        plant = self.plant
        original = plant.step
        self._plant_step_original = original

        def step_with_contact_readout(torque_nm, **kwargs):
            """Force the five-sample contact readout of an already solved step."""
            kwargs["measure_contact_wrench"] = True
            return original(torque_nm, **kwargs)

        plant.step = step_with_contact_readout

    def _restore_contact_readout(self) -> None:
        """Restore the original bound plant.step, leaving outer wrappers intact."""
        original = self._plant_step_original
        self._plant_step_original = None
        if original is None:
            return
        plant = getattr(self, "plant", None)
        if plant is None:  # pragma: no cover - guard
            return
        current = plant.__dict__.get("step")
        if current is not None and getattr(current, "__name__", "") == (
            "step_with_contact_readout"
        ):
            plant.__dict__.pop("step", None)
        else:
            # An evaluation observer wrapped this instance; leave its wrapper in
            # place and only drop our own reference.
            return

    # -- task plumbing ----------------------------------------------------------------
    def _raw_command_callback(self, time_s: float):
        """Fixed raw callback: zero forward/yaw plus the scheduled height."""
        tick = int(round(float(time_s) / float(self.control_dt)))
        tick = max(0, min(JUMP_HORIZON_TICKS, tick))
        return raw_command_at_tick(self._jump_spec, tick)

    @property
    def jump_episode_spec(self) -> JumpEpisodeSpec:
        """Episode settings in force for the current or next episode."""
        return self._jump_spec

    @property
    def jump_progress(self) -> JumpProgress:
        """Private per-episode task accounting object."""
        return self._progress

    @property
    def geometry_binding_receipt(self) -> dict[str, Any]:
        """Receipt of the single wheel/plane geometry binding."""
        return copy.deepcopy(self._binding_receipt)

    def configure_next_episode(self, spec: JumpEpisodeSpec) -> None:
        """Select the episode settings applied by the *next* parent reset."""
        if not isinstance(spec, JumpEpisodeSpec):
            raise TypeError("spec must be a JumpEpisodeSpec")
        if self._active:
            raise RuntimeError("an active episode must not be reconfigured")
        self._pending_spec = spec

    def _build_heading_task_config(self) -> dict:
        """Publish the new static jump identities over the hardcoded heading ones."""
        config = dict(super()._build_heading_task_config())
        config["schema"] = JUMP_TASK_CONFIG_SCHEMA
        config["task_schema"] = JUMP_TASK_SCHEMA
        config["observation_schema"] = JUMP_OBSERVATION_SCHEMA
        config["reward_schema"] = JUMP_REWARD_SCHEMA
        config["observation_size"] = JUMP_OBSERVATION_SIZE
        config["action_size"] = JUMP_ACTION_SIZE
        config["parent_observation_prefix_size"] = HEADING_OBSERVATION_SIZE
        config["heading_prefix"] = {
            "task_schema": super()._build_heading_task_config().get("task_schema"),
            "observation_schema": super()._build_heading_task_config().get(
                "observation_schema"
            ),
            "size": JUMP_PARENT_OBSERVATION_SIZE,
        }
        config["jump"] = jump_task_definition()
        return config

    @property
    def jump_task_config(self) -> dict:
        """Full static jump task configuration used for checkpoint comparison."""
        return copy.deepcopy(self._heading_task_config)

    # -- measurement ------------------------------------------------------------------
    def _sample_geometry(self) -> dict[str, Any]:
        """Sample wheel clearance from the synchronized measurement arrays."""
        data = self.plant.measurement_data
        return dict(sample_wheel_clearance(self._binding, data.geom_xpos, data.geom_xmat))

    def _whole_robot_com(self) -> tuple[np.ndarray, np.ndarray]:
        """Whole-robot COM position and velocity (never the base origin/qvel[2])."""
        plant = self.plant
        model = plant.model
        data = plant.measurement_data
        masses = np.asarray(model.body_mass, dtype=np.float64)
        xipos = np.asarray(data.xipos, dtype=np.float64)
        weights = masses.copy()
        weights[0] = 0.0
        total = float(weights.sum())
        if abs(total - self._robot_mass_kg) > _MASS_TOLERANCE_KG:  # pragma: no cover
            raise RuntimeError("the robot mass changed between construction and readout")
        position = (weights[:, None] * xipos).sum(axis=0) / total
        jacobian = self._com_jacobian
        jacobian.fill(0.0)
        mujoco.mj_jacSubtreeCom(model, data, jacobian, int(plant.base_body_id))
        velocity = jacobian @ np.asarray(data.qvel, dtype=np.float64)
        if not (np.all(np.isfinite(position)) and np.all(np.isfinite(velocity))):
            raise RuntimeError("the whole-robot COM position and velocity must be finite")
        return position, velocity

    def _endpoint_metrics(self, geometry: dict[str, Any], info: dict) -> JumpEndpointMetrics:
        """Assemble the synchronized endpoint record for one completed interval."""
        wrench = self.plant.last_contact_wrench
        if wrench is None:
            raise RuntimeError(
                "the five-sample measured contact summary is missing; an absent readout "
                "is an exception, not a zero success signal"
            )
        state = self.plant.published_state
        com_position, com_velocity = self._whole_robot_com()
        metrics = info.get("metrics", {})
        return JumpEndpointMetrics(
            simultaneous_minimum_gap_m=geometry["simultaneous_minimum_gap_m"],
            contact_margin_m=geometry["contact_margin_m"],
            wheel_bottom_gap_m=geometry["wheel_bottom_gap_m"],
            active_sample_fraction_by_wheel=wrench.active_sample_fraction_by_wheel,
            wheel_force_world_n=wrench.wheel_force_world_n,
            physics_sample_count=int(wrench.physics_sample_count),
            base_position_m=state.base_position,
            base_rpy_rad=state.base_orientation_rpy,
            body_forward_velocity_mps=float(state.base_velocity[0]),
            com_position_m=com_position,
            com_velocity_mps=com_velocity,
            initial_origin_xy_m=self._initial_origin,
            heading_error_rad=float(metrics.get("heading_error_rad", 0.0)),
        )

    # -- gym API ----------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        """Reset task state, call the parent reset once and append the new features."""
        if self._pending_spec is not None:
            self._jump_spec = self._pending_spec
            self._pending_spec = None
        self._progress.reset()
        self._origin_recorded = False
        self._last_geometry = None
        self._last_metrics = None
        self._last_reward_breakdown = None
        self._last_progress_event = None

        observation, info = super().reset(seed=seed, options=options)
        state = self.plant.published_state
        self._initial_origin = np.array(state.base_position[:2], dtype=np.float64)
        self._origin_recorded = True
        context = self._require_context("reset")
        extended = append_jump_observation(
            observation,
            state,
            int(context.tick),
            self._jump_spec,
            self._progress,
            self._initial_origin,
        )
        info = dict(info)
        info["jump_episode"] = self._jump_spec.as_dict()
        info["jump_progress"] = self._progress.snapshot.as_dict()
        info["jump_geometry_binding"] = self.geometry_binding_receipt
        info["jump_phase"] = phase_at_tick(self._jump_spec, int(context.tick))
        info["episode_metadata"] = dict(info.get("episode_metadata", {}))
        info["episode_metadata"]["jump_episode"] = self._jump_spec.as_dict()
        self.episode_metadata["jump_episode"] = self._jump_spec.as_dict()
        self.episode_metadata["jump_task_config"] = self.jump_task_config
        return extended, info

    def step(self, action):
        """Step the parent exactly once, then replace reward/observation."""
        before_context = self._require_context("step")
        executed_tick = int(before_context.tick)
        progress_before = self._progress.snapshot

        observation, reward, terminated, truncated, info = super().step(action)

        geometry = self._sample_geometry()
        self._last_geometry = geometry
        metrics = self._endpoint_metrics(geometry, info)
        self._last_metrics = metrics
        event = self._progress.advance(
            self._jump_spec,
            executed_tick,
            float(self.control_dt),
            metrics,
            bool(terminated),
            bool(truncated),
        )
        self._last_progress_event = event
        progress_after = self._progress.snapshot
        breakdown = compose_jump_reward(
            info["reward_terms"],
            executed_tick,
            self._jump_spec,
            float(self.control_dt),
            progress_before,
            progress_after,
            metrics,
            bool(terminated),
            bool(truncated),
            executed_tick + 1,
        )
        self._last_reward_breakdown = breakdown

        cached_terminal = info.get("terminal_observation_source") == "last_executable_decision"
        context = before_context if cached_terminal else self._require_context("step")
        if not cached_terminal and int(context.tick) != executed_tick + 1:
            raise RuntimeError("the next prepared decision must follow the executed tick")
        extended = append_jump_observation(
            observation,
            self.plant.published_state,
            int(context.tick),
            self._jump_spec,
            self._progress,
            self._initial_origin,
        )
        info = dict(info)
        info["parent_reward"] = float(reward)
        info["parent_reward_terms"] = dict(breakdown.parent_reward_terms)
        info["reward_terms"] = dict(breakdown.reward_terms)
        info["jump_reward"] = breakdown.as_dict()
        info["jump_geometry"] = dict(geometry)
        info["jump_endpoint_metrics"] = metrics.as_dict()
        info["jump_progress_event"] = event.as_dict()
        info["jump_progress"] = progress_after.as_dict()
        info["jump_phase"] = phase_at_tick(self._jump_spec, int(context.tick))
        return extended, float(breakdown.reward), terminated, truncated, info

    def close(self) -> None:
        """Restore the plant.step wrapper, then close the parent."""
        try:
            self._restore_contact_readout()
        finally:
            super().close()


def load_jump_policy(model_path, metadata_path, env):
    """Validate the new static jump contract, then delegate deserialization.

    The sidecar's full static task config and the declared 95/8 actor schemas are
    compared *before* any deserialization; all old 82/85 and zero-only identities
    are rejected even when a shape field is edited.  Hashes, gains and spaces stay
    with the original :func:`load_locomotion_policy`; there is no unpickle
    fallback, shape padding or metadata repair.
    """
    payload = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("the checkpoint sidecar must be a JSON object")
    recorded = payload.get("recorded_episode")
    if not isinstance(recorded, dict):
        raise ValueError("the sidecar must record its episode metadata")
    declared = {
        "task_schema": payload.get("task_schema"),
        "observation_schema": payload.get("observation_schema"),
        "reward_schema": payload.get("reward_schema"),
        "action_schema": payload.get("action_schema"),
        "action_mode": payload.get("action_mode"),
        "observation_dim": payload.get("observation_dim"),
        "action_dim": payload.get("action_dim"),
    }
    expected = {
        "task_schema": JUMP_TASK_SCHEMA,
        "observation_schema": JUMP_OBSERVATION_SCHEMA,
        "reward_schema": JUMP_REWARD_SCHEMA,
        "action_schema": env.unwrapped.action_schema,
        "action_mode": "independent8",
        "observation_dim": JUMP_OBSERVATION_SIZE,
        "action_dim": JUMP_ACTION_SIZE,
    }
    for key, want in expected.items():
        if declared[key] != want:
            raise ValueError(
                f"incompatible checkpoint {key}: expected {want!r}, got {declared[key]!r}"
            )
    recorded_config = recorded.get("jump_task_config")
    expected_config = env.unwrapped.jump_task_config
    if _canonical(recorded_config) != _canonical(expected_config):
        raise ValueError("the recorded jump task configuration does not match this environment")
    return load_locomotion_policy(model_path, metadata_path, env)


def _canonical(payload: Any) -> str:
    """Canonical JSON text of a task configuration for exact comparison."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
