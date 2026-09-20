"""Pure task definitions for the scaffolded residual hop PPO task.

Scope (frozen ``fixed_contract.md`` / ``claude_execution_spec.md``)
------------------------------------------------------------------
This module owns the *pure* half of the new jump task:

* the frozen task/observation/reward identities and the JSON-compatible
  immutable task definition,
* :class:`JumpEpisodeSpec`, the validated per-episode request/goal/friction
  record (episode settings only, never static checkpoint semantics),
* :func:`raw_command_at_tick` / :func:`phase_at_tick`, which shift the approved
  :func:`scripts.d1_jump_readiness.fixed_height_command` schedule without
  changing any of its values or its physical horizon,
* :class:`JumpProgress`, the mutable per-episode task accounting advanced exactly
  once per successfully completed control interval,
* :func:`append_jump_observation`, the pure 85 -> 95 observation extension,
* :func:`compose_jump_reward`, the fixed named reward composition.

Nothing here constructs or touches a model, plant, controller, PPO object or
MuJoCo call.  The only imports are the immutable motion-command type and the
approved pure height/geometry helper module.  No alternative reward, gain or
action mode exists.

Oracle provenance
-----------------
The appended observation entries 87..90 (contacts) and 91..94 (task progress,
airborne progress, world planar offset) come from the published *oracle* state
and from simulator geometry/contact accounting.  The schema is therefore named
``d1-jump-oracle-task95-v1``; it is not a sensor-ready proprioceptive encoding.

A command phase label never certifies a measured contact mode, and a positive
geometric gap is never by itself a flight claim: flight is certified only by two
consecutive fully unloaded control intervals whose run began with a positive
whole-robot COM vertical velocity.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from wheel_legged_control.d1.control_loop import D1MotionCommand

try:  # the approved read-only helper lives next to this file in scripts/
    from d1_jump_readiness import (
        JUMP_READINESS_SCHEMA,
        TERMINAL_PREPARED_TICK,
        fixed_height_command,
    )
except ModuleNotFoundError:  # pragma: no cover - import layout fallback only
    try:
        from scripts.d1_jump_readiness import (
            JUMP_READINESS_SCHEMA,
            TERMINAL_PREPARED_TICK,
            fixed_height_command,
        )
    except ModuleNotFoundError:
        _SCRIPTS_DIR = str(Path(__file__).resolve().parent)
        if _SCRIPTS_DIR not in sys.path:
            sys.path.insert(0, _SCRIPTS_DIR)
        from d1_jump_readiness import (
            JUMP_READINESS_SCHEMA,
            TERMINAL_PREPARED_TICK,
            fixed_height_command,
        )

__all__ = [
    "AIRBORNE_NATIVE_SAMPLES_FOR_FULL_PROGRESS",
    "ALLOWED_FRICTION_SCALES",
    "ALLOWED_NET_CLEARANCE_M",
    "BONUS_WINDOW_TICKS",
    "CERTIFYING_UNLOADED_INTERVALS",
    "CLEARANCE_PROGRESS_WEIGHT",
    "FLIGHT_ONCE_BONUS",
    "HEIGHT_MASK_WINDOW_TICKS",
    "JUMP_ACTION_SIZE",
    "JUMP_APPENDED_OBSERVATION_FIELDS",
    "JUMP_CONTROL_DT_S",
    "JUMP_HORIZON_TICKS",
    "JUMP_LEG_ORDER",
    "JUMP_OBSERVATION_CLIP",
    "JUMP_OBSERVATION_SCHEMA",
    "JUMP_OBSERVATION_SIZE",
    "JUMP_PARENT_OBSERVATION_SIZE",
    "JUMP_PROGRESS_SCHEMA",
    "JUMP_REQUEST_PROFILE_OFFSET_TICKS",
    "JUMP_REWARD_SCHEMA",
    "JUMP_TASK_CONFIG_SCHEMA",
    "JUMP_TASK_DEFINITION",
    "JUMP_TASK_SCHEMA",
    "LANDING_ONCE_BONUS",
    "NATIVE_SAMPLES_PER_CONTROL_INTERVAL",
    "NET_CLEARANCE_OBSERVATION_SCALE_M",
    "PARENT_HEIGHT_TERM_NAME",
    "STATIONARY_OFFSET_SCALE_M",
    "JumpEndpointMetrics",
    "JumpEpisodeSpec",
    "JumpProgress",
    "JumpProgressEvent",
    "JumpProgressSnapshot",
    "JumpRewardBreakdown",
    "append_jump_observation",
    "compose_jump_reward",
    "jump_task_definition",
    "phase_at_tick",
    "raw_command_at_tick",
]


# --- frozen identities -----------------------------------------------------------------
JUMP_TASK_SCHEMA = "d1-native-plane-residual-hop-task-v1"
JUMP_OBSERVATION_SCHEMA = "d1-jump-oracle-task95-v1"
JUMP_REWARD_SCHEMA = "d1-hop-clearance-events-heading-rate-v1"
JUMP_TASK_CONFIG_SCHEMA = "d1-jump-task-config-v1"
JUMP_PROGRESS_SCHEMA = "d1-jump-clearance-progress-v1"

#: The parent heading encoding kept as an exact byte prefix.
JUMP_PARENT_OBSERVATION_SIZE = 85
JUMP_OBSERVATION_SIZE = 95
JUMP_ACTION_SIZE = 8
JUMP_CONTROL_DT_S = 0.01
JUMP_HORIZON_TICKS = 600
JUMP_OBSERVATION_CLIP = (-5.0, 5.0)

#: Canonical leg order of every four-entry task field.
JUMP_LEG_ORDER = ("FL", "FR", "RL", "RR")

#: ``fixed_height_command`` places its request at tick 200; a request at ``s`` is
#: obtained by feeding ``clip(k - s + 200, 0, 600)`` to the approved schedule.
JUMP_REQUEST_PROFILE_OFFSET_TICKS = 200

#: Requested net wheel clearances (m) and the explicit evaluation friction scales.
ALLOWED_NET_CLEARANCE_M = (0.005, 0.010, 0.020)
ALLOWED_FRICTION_SCALES = (0.95, 1.0, 1.05)

#: ``[s, s + 120)`` earns progress; ``[s + 25, s + 120)`` masks the height term.
BONUS_WINDOW_TICKS = 120
HEIGHT_MASK_WINDOW_TICKS = (25, 120)

#: Five native samples per 10 ms control interval; two fully unloaded intervals
#: conservatively certify 10 recorded native evaluations / 20 ms of flight.
NATIVE_SAMPLES_PER_CONTROL_INTERVAL = 5
CERTIFYING_UNLOADED_INTERVALS = 2
AIRBORNE_NATIVE_SAMPLES_FOR_FULL_PROGRESS = 10

#: Bounded event contributions: 2 + 1 + 2 = 5 maximum per episode.
CLEARANCE_PROGRESS_WEIGHT = 2.0
FLIGHT_ONCE_BONUS = 1.0
LANDING_ONCE_BONUS = 2.0

#: Stationary displacement incentive ``-dt * 0.5 * min((d / 0.10)^2, 4)``.
STATIONARY_OFFSET_SCALE_M = 0.10
STATIONARY_OFFSET_WEIGHT = 0.5
STATIONARY_OFFSET_SATURATION = 4.0

#: Observation scale of entry 86 (``requested net clearance / 0.04``), the leg
#: residual authority of one control interval.
NET_CLEARANCE_OBSERVATION_SCALE_M = 0.04

#: Original parent height contribution whose effective value is masked in flight.
PARENT_HEIGHT_TERM_NAME = "height"

#: Landing conditions of the single genuine final-transition bonus.
LANDING_BASE_HEIGHT_M = 0.455
LANDING_HEIGHT_TOLERANCE_M = 0.015
LANDING_BODY_VX_LIMIT_MPS = 0.03
LANDING_COM_VZ_LIMIT_MPS = 0.03
LANDING_ATTITUDE_LIMIT_RAD = math.radians(10.0)
LANDING_HEADING_LIMIT_RAD = math.radians(5.0)
LANDING_DISPLACEMENT_LIMIT_M = 0.10

JUMP_APPENDED_OBSERVATION_FIELDS = (
    "request_clock",
    "requested_net_clearance_normalized",
    "contact_FL",
    "contact_FR",
    "contact_RL",
    "contact_RR",
    "clearance_progress_fraction",
    "airborne_progress",
    "planar_offset_x_normalized",
    "planar_offset_y_normalized",
)

_APPENDED_COUNT = JUMP_OBSERVATION_SIZE - JUMP_PARENT_OBSERVATION_SIZE
if len(JUMP_APPENDED_OBSERVATION_FIELDS) != _APPENDED_COUNT:  # pragma: no cover - guard
    raise RuntimeError("the appended field list must describe exactly ten new entries")

_PLANAR_OFFSET_SCALE_M = STATIONARY_OFFSET_SCALE_M
_HOLD_CONDITION = "hold"
_PROFILE_CONDITION = "profile"
_TOLERANCE = 1e-12


def _json_ready(payload: Any) -> Any:
    """Convert nested mappings/tuples into plain JSON types (lists, dicts)."""
    if isinstance(payload, Mapping):
        return {str(key): _json_ready(value) for key, value in payload.items()}
    if isinstance(payload, (tuple, list)):
        return [_json_ready(item) for item in payload]
    return payload


def _frozen(payload: Any) -> Any:
    """Recursively freeze a definition into mapping proxies and tuples."""
    if isinstance(payload, Mapping):
        return MappingProxyType({str(key): _frozen(value) for key, value in payload.items()})
    if isinstance(payload, (tuple, list)):
        return tuple(_frozen(item) for item in payload)
    return payload


_TASK_DEFINITION = {
    "schema": JUMP_TASK_CONFIG_SCHEMA,
    "task_schema": JUMP_TASK_SCHEMA,
    "observation_schema": JUMP_OBSERVATION_SCHEMA,
    "reward_schema": JUMP_REWARD_SCHEMA,
    "progress_schema": JUMP_PROGRESS_SCHEMA,
    "readiness_helper_schema": JUMP_READINESS_SCHEMA,
    "baseline": "wheel_leg",
    "action_mode": "independent8",
    "action": {
        "size": JUMP_ACTION_SIZE,
        "low": -1.0,
        "high": 1.0,
        "dtype": "float32",
        "leg_extension_residual_scale_m": 0.04,
        "wheel_speed_residual_scale_rad_s": 4.0,
        "order": "four leg-extension residuals then four wheel-speed residuals",
        "leg_order": JUMP_LEG_ORDER,
        "projection": "none: learned actions reach the original controller unchanged",
    },
    "episode": {
        "control_dt_s": JUMP_CONTROL_DT_S,
        "horizon_ticks": JUMP_HORIZON_TICKS,
        "duration_s": JUMP_CONTROL_DT_S * JUMP_HORIZON_TICKS,
        "reset_inside_episode": False,
    },
    "profile": {
        "source": "scripts.d1_jump_readiness.fixed_height_command",
        "request_profile_offset_ticks": JUMP_REQUEST_PROFILE_OFFSET_TICKS,
        "hold_condition": _HOLD_CONDITION,
        "request_condition": _PROFILE_CONDITION,
        "raw_forward_velocity_mps": 0.0,
        "raw_yaw_rate_rps": 0.0,
        "baseline_clearance_m": 0.455,
        "relative_segments_m": (
            {"first_offset": 0, "last_offset": 24, "clearance_m": 0.405},
            {"first_offset": 25, "last_offset": 39, "clearance_m": 0.500},
            {"first_offset": 40, "last_offset": 74, "clearance_m": 0.455},
            {"first_offset": 75, "last_offset": 119, "clearance_m": 0.435},
        ),
        "allowed_net_clearance_m": ALLOWED_NET_CLEARANCE_M,
        "allowed_friction_scales": ALLOWED_FRICTION_SCALES,
        "scaffolded": True,
        "note": "scaffolded residual hop learning, not an end-to-end discovered jump",
    },
    "observation": {
        "parent_size": JUMP_PARENT_OBSERVATION_SIZE,
        "total_size": JUMP_OBSERVATION_SIZE,
        "appended_fields": JUMP_APPENDED_OBSERVATION_FIELDS,
        "clip_low": JUMP_OBSERVATION_CLIP[0],
        "clip_high": JUMP_OBSERVATION_CLIP[1],
        "dtype": "float32",
        "net_clearance_scale_m": NET_CLEARANCE_OBSERVATION_SCALE_M,
        "planar_offset_scale_m": _PLANAR_OFFSET_SCALE_M,
        "request_clock_before_request": -1.0,
        "request_clock_window_ticks": BONUS_WINDOW_TICKS,
        "contact_source": "published_oracle_state.wheel_contact",
        "progress_source": "oracle_task_geometry_and_native_contact_accounting",
        "gate_time": "prepared_decision_tick",
    },
    "progress": {
        "native_samples_per_control_interval": NATIVE_SAMPLES_PER_CONTROL_INTERVAL,
        "certifying_unloaded_intervals": CERTIFYING_UNLOADED_INTERVALS,
        "airborne_native_samples_for_full_progress": (
            AIRBORNE_NATIVE_SAMPLES_FOR_FULL_PROGRESS
        ),
        "bonus_window_ticks": BONUS_WINDOW_TICKS,
        "net_gap_definition": "c = max(0, simultaneous_minimum_gap_m - contact_margin_m)",
        "unloaded_definition": (
            "all four active_sample_fraction_by_wheel zero, all four measured wheel "
            "force vectors zero and endpoint simultaneous gap strictly above the margin"
        ),
        "upward_onset_required": True,
        "first_certified_run_uses_run_maximum": True,
        "progress_monotone": True,
        "hold_or_pre_request_progress": False,
    },
    "reward": {
        "parent_terms_retained": True,
        "parent_height_term": PARENT_HEIGHT_TERM_NAME,
        "height_mask_window_ticks": HEIGHT_MASK_WINDOW_TICKS,
        "height_mask_basis": "command_time_predeclared",
        "clearance_progress_weight": CLEARANCE_PROGRESS_WEIGHT,
        "flight_once_bonus": FLIGHT_ONCE_BONUS,
        "landing_once_bonus": LANDING_ONCE_BONUS,
        "maximum_event_total": (
            CLEARANCE_PROGRESS_WEIGHT + FLIGHT_ONCE_BONUS + LANDING_ONCE_BONUS
        ),
        "stationary_offset": {
            "weight": STATIONARY_OFFSET_WEIGHT,
            "scale_m": STATIONARY_OFFSET_SCALE_M,
            "saturation": STATIONARY_OFFSET_SATURATION,
            "formula": "-actual_dt * 0.5 * min((|base_xy - initial_xy| / 0.10)^2, 4)",
        },
        "landing_once_conditions": {
            "endpoint_tick": JUMP_HORIZON_TICKS,
            "base_height_m": LANDING_BASE_HEIGHT_M,
            "base_height_tolerance_m": LANDING_HEIGHT_TOLERANCE_M,
            "body_vx_limit_mps": LANDING_BODY_VX_LIMIT_MPS,
            "com_vz_limit_mps": LANDING_COM_VZ_LIMIT_MPS,
            "attitude_limit_rad": LANDING_ATTITUDE_LIMIT_RAD,
            "heading_limit_rad": LANDING_HEADING_LIMIT_RAD,
            "displacement_limit_m": LANDING_DISPLACEMENT_LIMIT_M,
            "requires_full_progress": True,
            "requires_flight": True,
            "requires_all_wheels_loaded": True,
        },
        "fixed_before_training": True,
    },
}

#: Immutable, JSON-compatible frozen definition of the whole new task.
JUMP_TASK_DEFINITION = _frozen(_TASK_DEFINITION)

# Fail loudly at import if any non-JSON value ever enters the definition.
json.dumps(_json_ready(JUMP_TASK_DEFINITION), sort_keys=True, allow_nan=False)


def jump_task_definition() -> dict[str, Any]:
    """Return a fresh JSON-ready copy of :data:`JUMP_TASK_DEFINITION`."""
    return _json_ready(JUMP_TASK_DEFINITION)


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------


def _finite_float(value: Any, name: str) -> float:
    """Validate a real finite non-boolean scalar and return it as a float."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real finite number, got a boolean")
    if isinstance(value, (complex, np.complexfloating)):
        raise TypeError(f"{name} must be a real finite number, got a complex value")
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real finite number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _exact_integer(value: Any, name: str) -> int:
    """Validate a non-boolean exact integer (integer-valued floats rejected)."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a non-boolean integer, got a boolean")
    if isinstance(value, np.integer) or type(value) is int:
        return int(value)
    if isinstance(value, int):  # a non-bool int subclass is still an exact integer
        return int(value)
    raise TypeError(
        f"{name} must be a Python int or numpy integer (floats, including "
        f"integer-valued floats, are rejected), got {type(value).__name__}"
    )


def _validated_tick(value: Any, name: str = "tick") -> int:
    """Validate a physical tick in ``0..600`` inclusive."""
    tick = _exact_integer(value, name)
    if tick < 0 or tick > TERMINAL_PREPARED_TICK:
        raise ValueError(
            f"{name} out of range: expected 0..{TERMINAL_PREPARED_TICK} inclusive, got {tick}"
        )
    return tick


def _read_only_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Convert to an owned finite read-only float64 array of the given shape."""
    array = value if isinstance(value, np.ndarray) else np.asarray(value)
    if array.dtype.kind not in ("f", "i", "u"):
        raise TypeError(
            f"{name} must be a real numeric array (bool, complex, string and object "
            f"arrays are rejected), got dtype {array.dtype!r}"
        )
    out = np.array(array, dtype=np.float64, copy=True)
    if out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite")
    out.setflags(write=False)
    return out


def _matching_allowed(value: float, allowed: tuple[float, ...], name: str) -> float:
    """Return the allowed constant exactly equal to ``value`` or raise."""
    for candidate in allowed:
        if value == candidate:
            return float(candidate)
    raise ValueError(f"{name} must be exactly one of {allowed}, got {value!r}")


# ---------------------------------------------------------------------------
# frozen episode specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JumpEpisodeSpec:
    """One episode's requested hop: onset tick, net clearance goal and friction.

    A no-jump hold is exactly ``JumpEpisodeSpec(None, 0, friction_scale)``.  A
    request uses a non-negative integer tick whose whole profile/recovery window
    ``[s, s + 120)`` lies inside the fixed 600-tick episode, one of the allowed
    fixed goals ``0.005 / 0.010 / 0.020`` m and one of the explicit friction
    scales ``0.95 / 1.0 / 1.05``.

    These are *episode settings*.  They are recorded in episode metadata and are
    deliberately not part of the static checkpoint task semantics: a different
    goal/onset/friction inside the declared contract stays loadable and is never
    evidence of robustness.
    """

    request_tick: int | None
    net_clearance_m: float
    friction_scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate and normalize the episode settings, rejecting hidden inputs."""
        if callable(self.request_tick) or callable(self.net_clearance_m):
            raise TypeError("episode settings must be plain numbers, not callable conditions")
        friction = _matching_allowed(
            _finite_float(self.friction_scale, "friction_scale"),
            ALLOWED_FRICTION_SCALES,
            "friction_scale",
        )
        object.__setattr__(self, "friction_scale", friction)

        if self.request_tick is None:
            clearance = _finite_float(self.net_clearance_m, "net_clearance_m")
            if clearance != 0.0:
                raise ValueError(
                    "a no-jump hold is exactly (None, 0, friction_scale); got "
                    f"net_clearance_m={self.net_clearance_m!r}"
                )
            object.__setattr__(self, "net_clearance_m", 0.0)
            return

        tick = _exact_integer(self.request_tick, "request_tick")
        latest = JUMP_HORIZON_TICKS - BONUS_WINDOW_TICKS
        if tick < 0 or tick > latest:
            raise ValueError(
                "request_tick must be a non-negative integer whose profile/recovery "
                f"window fits inside {JUMP_HORIZON_TICKS} ticks: expected 0..{latest}, got {tick}"
            )
        object.__setattr__(self, "request_tick", tick)
        object.__setattr__(
            self,
            "net_clearance_m",
            _matching_allowed(
                _finite_float(self.net_clearance_m, "net_clearance_m"),
                ALLOWED_NET_CLEARANCE_M,
                "net_clearance_m",
            ),
        )

    @property
    def is_hold(self) -> bool:
        """True when no jump is requested at any tick of the episode."""
        return self.request_tick is None

    @property
    def condition(self) -> str:
        """Approved ``fixed_height_command`` condition of this episode."""
        return _HOLD_CONDITION if self.is_hold else _PROFILE_CONDITION

    @property
    def bonus_window_ticks(self) -> tuple[int, int] | None:
        """Half-open ``[s, s + 120)`` progress window, or None for a hold."""
        if self.request_tick is None:
            return None
        return (self.request_tick, self.request_tick + BONUS_WINDOW_TICKS)

    @property
    def height_mask_window_ticks(self) -> tuple[int, int] | None:
        """Half-open ``[s + 25, s + 120)`` masked-height window, or None."""
        if self.request_tick is None:
            return None
        first, last = HEIGHT_MASK_WINDOW_TICKS
        return (self.request_tick + first, self.request_tick + last)

    def in_bonus_window(self, tick: Any) -> bool:
        """True when a validated execution tick earns progress for this episode."""
        executed = _validated_tick(tick, "tick")
        window = self.bonus_window_ticks
        return window is not None and window[0] <= executed < window[1]

    def masks_height_term(self, tick: Any) -> bool:
        """True when the original height contribution is masked at this tick."""
        executed = _validated_tick(tick, "tick")
        window = self.height_mask_window_ticks
        return window is not None and window[0] <= executed < window[1]

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready copy of the episode settings."""
        return {
            "request_tick": None if self.request_tick is None else int(self.request_tick),
            "net_clearance_m": float(self.net_clearance_m),
            "friction_scale": float(self.friction_scale),
            "condition": self.condition,
            "is_hold": bool(self.is_hold),
            "bonus_window_ticks": (
                None if self.bonus_window_ticks is None else list(self.bonus_window_ticks)
            ),
            "height_mask_window_ticks": (
                None
                if self.height_mask_window_ticks is None
                else list(self.height_mask_window_ticks)
            ),
        }


def _validated_spec(spec: Any) -> JumpEpisodeSpec:
    if not isinstance(spec, JumpEpisodeSpec):
        raise TypeError(f"spec must be a JumpEpisodeSpec, got {type(spec).__name__}")
    return spec


def _scheduled(spec: JumpEpisodeSpec, tick: Any) -> tuple[D1MotionCommand, str]:
    """Return the approved ``(command, phase_label)`` for one prepared tick."""
    validated = _validated_spec(spec)
    prepared = _validated_tick(tick, "tick")
    if validated.is_hold:
        return fixed_height_command(prepared, _HOLD_CONDITION)
    shifted = max(
        0,
        min(
            TERMINAL_PREPARED_TICK,
            prepared - int(validated.request_tick) + JUMP_REQUEST_PROFILE_OFFSET_TICKS,
        ),
    )
    return fixed_height_command(shifted, _PROFILE_CONDITION)


def raw_command_at_tick(spec: Any, tick: Any) -> D1MotionCommand:
    """Return the raw operator command prepared at ``tick`` for ``spec``.

    Forward velocity and yaw rate are exactly zero at every tick, so the stop
    latch never arms and yaw authority stays inactive by construction.  Tick 600
    is a valid, already-prepared and final-held command; it is never a new
    execution.  This function advances no state and owns no clock.
    """
    command, _ = _scheduled(spec, tick)
    return command


def phase_at_tick(spec: Any, tick: Any) -> str:
    """Return the approved command *phase label* prepared at ``tick``.

    The label describes the commanded schedule only.  It never claims measured
    flight, takeoff or landing.
    """
    _, label = _scheduled(spec, tick)
    return label


# ---------------------------------------------------------------------------
# endpoint measurement record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JumpEndpointMetrics:
    """Synchronized endpoint geometry, pose, COM and five-sample contact summary.

    ``simultaneous_minimum_gap_m`` (g) and ``contact_margin_m`` (m) come straight
    from ``sample_wheel_clearance``; the net useful gap is ``c = max(0, g - m)``.
    Base rise and the independent maxima of the four wheels are never used as c.

    The measured contact summary must carry exactly five native samples of one
    complete control interval: a missing or short summary is an exception, never
    a zero success signal.
    """

    simultaneous_minimum_gap_m: float
    contact_margin_m: float
    wheel_bottom_gap_m: np.ndarray
    active_sample_fraction_by_wheel: np.ndarray
    wheel_force_world_n: np.ndarray
    physics_sample_count: int
    base_position_m: np.ndarray
    base_rpy_rad: np.ndarray
    body_forward_velocity_mps: float
    com_position_m: np.ndarray
    com_velocity_mps: np.ndarray
    initial_origin_xy_m: np.ndarray
    heading_error_rad: float

    def __post_init__(self) -> None:
        """Validate every measured field, rejecting an incomplete interval."""
        legs = len(JUMP_LEG_ORDER)
        for name, shape in (
            ("wheel_bottom_gap_m", (legs,)),
            ("active_sample_fraction_by_wheel", (legs,)),
            ("wheel_force_world_n", (legs, 3)),
            ("base_position_m", (3,)),
            ("base_rpy_rad", (3,)),
            ("com_position_m", (3,)),
            ("com_velocity_mps", (3,)),
            ("initial_origin_xy_m", (2,)),
        ):
            object.__setattr__(self, name, _read_only_array(getattr(self, name), shape, name))
        for name in (
            "simultaneous_minimum_gap_m",
            "contact_margin_m",
            "body_forward_velocity_mps",
            "heading_error_rad",
        ):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))
        if self.contact_margin_m < 0.0:
            raise ValueError("contact_margin_m must be non-negative")
        fractions = self.active_sample_fraction_by_wheel
        if np.any(fractions < 0.0) or np.any(fractions > 1.0):
            raise ValueError("active_sample_fraction_by_wheel must lie in [0, 1]")
        count = _exact_integer(self.physics_sample_count, "physics_sample_count")
        if count != NATIVE_SAMPLES_PER_CONTROL_INTERVAL:
            raise ValueError(
                "a complete control interval must carry exactly "
                f"{NATIVE_SAMPLES_PER_CONTROL_INTERVAL} native contact samples, got {count}"
            )
        object.__setattr__(self, "physics_sample_count", count)

    @property
    def net_gap_m(self) -> float:
        """Net useful gap ``c = max(0, g - m)`` in metres."""
        return max(0.0, float(self.simultaneous_minimum_gap_m) - float(self.contact_margin_m))

    @property
    def above_contact_margin(self) -> bool:
        """True when the simultaneous minimum gap is strictly above the margin."""
        return bool(float(self.simultaneous_minimum_gap_m) > float(self.contact_margin_m))

    @property
    def fully_unloaded(self) -> bool:
        """True only for a completely unloaded interval with a positive net gap."""
        return bool(
            np.all(self.active_sample_fraction_by_wheel == 0.0)
            and np.all(self.wheel_force_world_n == 0.0)
            and self.above_contact_margin
        )

    @property
    def wheel_normal_force_n(self) -> np.ndarray:
        """World vertical component of the measured per-wheel contact force."""
        return np.array(self.wheel_force_world_n[:, 2], dtype=np.float64, copy=True)

    @property
    def all_wheels_positively_loaded(self) -> bool:
        """True when every wheel carries a positive measured normal load."""
        return bool(np.all(self.wheel_normal_force_n > 0.0))

    @property
    def com_vertical_velocity_mps(self) -> float:
        """Whole-robot COM vertical velocity (never ``qvel[2]``)."""
        return float(self.com_velocity_mps[2])

    @property
    def planar_displacement_m(self) -> float:
        """Planar distance from the recorded published reset origin."""
        offset = np.asarray(self.base_position_m[:2], dtype=np.float64) - np.asarray(
            self.initial_origin_xy_m, dtype=np.float64
        )
        return float(np.linalg.norm(offset))

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready copy of the endpoint measurement."""
        return {
            "simultaneous_minimum_gap_m": float(self.simultaneous_minimum_gap_m),
            "contact_margin_m": float(self.contact_margin_m),
            "net_gap_m": self.net_gap_m,
            "wheel_bottom_gap_m": [float(v) for v in self.wheel_bottom_gap_m],
            "active_sample_fraction_by_wheel": [
                float(v) for v in self.active_sample_fraction_by_wheel
            ],
            "wheel_force_world_n": [[float(v) for v in row] for row in self.wheel_force_world_n],
            "wheel_normal_force_n": [float(v) for v in self.wheel_normal_force_n],
            "physics_sample_count": int(self.physics_sample_count),
            "fully_unloaded": self.fully_unloaded,
            "all_wheels_positively_loaded": self.all_wheels_positively_loaded,
            "base_position_m": [float(v) for v in self.base_position_m],
            "base_rpy_rad": [float(v) for v in self.base_rpy_rad],
            "body_forward_velocity_mps": float(self.body_forward_velocity_mps),
            "com_position_m": [float(v) for v in self.com_position_m],
            "com_velocity_mps": [float(v) for v in self.com_velocity_mps],
            "com_vertical_velocity_mps": self.com_vertical_velocity_mps,
            "initial_origin_xy_m": [float(v) for v in self.initial_origin_xy_m],
            "planar_displacement_m": self.planar_displacement_m,
            "heading_error_rad": float(self.heading_error_rad),
            "leg_order": list(JUMP_LEG_ORDER),
        }


# ---------------------------------------------------------------------------
# task progress accounting
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JumpProgressSnapshot:
    """Immutable, serializable view of the task accounting at one instant."""

    schema: str
    progress: float
    flight_seen: bool
    consecutive_unloaded_intervals: int
    unloaded_native_samples: int
    airborne_progress: float
    run_upward_onset: bool
    run_maximum_net_gap_m: float
    certified_run_maximum_net_gap_m: float
    completed_intervals: int

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready copy of the snapshot."""
        return {
            "schema": str(self.schema),
            "progress": float(self.progress),
            "flight_seen": bool(self.flight_seen),
            "consecutive_unloaded_intervals": int(self.consecutive_unloaded_intervals),
            "unloaded_native_samples": int(self.unloaded_native_samples),
            "airborne_progress": float(self.airborne_progress),
            "run_upward_onset": bool(self.run_upward_onset),
            "run_maximum_net_gap_m": float(self.run_maximum_net_gap_m),
            "certified_run_maximum_net_gap_m": float(self.certified_run_maximum_net_gap_m),
            "completed_intervals": int(self.completed_intervals),
        }


@dataclass(frozen=True, slots=True)
class JumpProgressEvent:
    """Named, immutable record of exactly one completed control interval."""

    schema: str
    executed_tick: int
    dt_s: float
    in_bonus_window: bool
    fully_unloaded: bool
    net_gap_m: float
    com_vertical_velocity_mps: float
    run_onset: bool
    newly_certified_flight: bool
    credited_net_gap_m: float
    terminated: bool
    truncated: bool
    before: JumpProgressSnapshot
    after: JumpProgressSnapshot

    @property
    def clearance_progress(self) -> float:
        """Bounded incremental clearance-progress contribution of this interval."""
        return CLEARANCE_PROGRESS_WEIGHT * (
            float(self.after.progress) - float(self.before.progress)
        )

    @property
    def flight_once(self) -> float:
        """One-off flight bonus of this interval (0 or 1)."""
        return FLIGHT_ONCE_BONUS if self.newly_certified_flight else 0.0

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready copy of the event."""
        return {
            "schema": str(self.schema),
            "executed_tick": int(self.executed_tick),
            "dt_s": float(self.dt_s),
            "in_bonus_window": bool(self.in_bonus_window),
            "fully_unloaded": bool(self.fully_unloaded),
            "net_gap_m": float(self.net_gap_m),
            "com_vertical_velocity_mps": float(self.com_vertical_velocity_mps),
            "run_onset": bool(self.run_onset),
            "newly_certified_flight": bool(self.newly_certified_flight),
            "credited_net_gap_m": float(self.credited_net_gap_m),
            "clearance_progress": float(self.clearance_progress),
            "flight_once": float(self.flight_once),
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
            "before": self.before.as_dict(),
            "after": self.after.as_dict(),
        }


class JumpProgress:
    """Mutable private accounting of achieved hop progress for one episode.

    Exactly one :meth:`advance` call belongs to each successfully completed
    control interval.  Preview, metadata reads and observation encoding never
    advance it.  Progress ``p`` is persistent, non-decreasing and bounded by one;
    it is never earned while supported, below the contact margin, before the
    request, on a hold or outside ``[s, s + 120)``.
    """

    __slots__ = (
        "_certified_run_maximum_net_gap_m",
        "_completed_intervals",
        "_consecutive_unloaded_intervals",
        "_flight_seen",
        "_progress",
        "_run_maximum_net_gap_m",
        "_run_upward_onset",
        "_unloaded_native_samples",
    )

    def __init__(self) -> None:
        """Create a freshly reset accounting object."""
        self.reset()

    def reset(self) -> None:
        """Clear all progress, flight and run state for a new episode."""
        self._progress = 0.0
        self._flight_seen = False
        self._consecutive_unloaded_intervals = 0
        self._unloaded_native_samples = 0
        self._run_upward_onset = False
        self._run_maximum_net_gap_m = 0.0
        self._certified_run_maximum_net_gap_m = 0.0
        self._completed_intervals = 0

    @property
    def progress(self) -> float:
        """Persistent achieved clearance progress ``p`` in ``[0, 1]``."""
        return float(self._progress)

    @property
    def flight_seen(self) -> bool:
        """True once a valid flight has been certified in this episode."""
        return bool(self._flight_seen)

    @property
    def airborne_progress(self) -> float:
        """Airborne progress feature: unloaded native count / 10, then 1."""
        if self._flight_seen:
            return 1.0
        return min(
            1.0,
            self._unloaded_native_samples / AIRBORNE_NATIVE_SAMPLES_FOR_FULL_PROGRESS,
        )

    @property
    def snapshot(self) -> JumpProgressSnapshot:
        """Immutable serializable snapshot of the current accounting."""
        return JumpProgressSnapshot(
            schema=JUMP_PROGRESS_SCHEMA,
            progress=self.progress,
            flight_seen=self.flight_seen,
            consecutive_unloaded_intervals=int(self._consecutive_unloaded_intervals),
            unloaded_native_samples=int(self._unloaded_native_samples),
            airborne_progress=self.airborne_progress,
            run_upward_onset=bool(self._run_upward_onset),
            run_maximum_net_gap_m=float(self._run_maximum_net_gap_m),
            certified_run_maximum_net_gap_m=float(self._certified_run_maximum_net_gap_m),
            completed_intervals=int(self._completed_intervals),
        )

    def advance(
        self,
        spec: Any,
        executed_tick: Any,
        dt_s: Any,
        metrics: Any,
        terminated: Any,
        truncated: Any,
    ) -> JumpProgressEvent:
        """Account exactly one successfully completed control interval.

        ``executed_tick`` is the pre-step tick k of the interval that has just
        been executed, ``dt_s`` its actual duration and ``metrics`` the
        synchronized endpoint geometry/COM/pose plus the original five-sample
        measured-contact summary.  Returns the named event contributions with
        immutable before/after snapshots.
        """
        validated = _validated_spec(spec)
        if not isinstance(metrics, JumpEndpointMetrics):
            raise TypeError(
                f"metrics must be a JumpEndpointMetrics, got {type(metrics).__name__}"
            )
        for name, flag in (("terminated", terminated), ("truncated", truncated)):
            if not isinstance(flag, (bool, np.bool_)):
                raise TypeError(f"{name} must be a boolean")
        tick = _exact_integer(executed_tick, "executed_tick")
        if tick < 0 or tick >= JUMP_HORIZON_TICKS:
            raise ValueError(
                f"executed_tick must index one of the {JUMP_HORIZON_TICKS} executions, got {tick}"
            )
        dt = _finite_float(dt_s, "dt_s")
        if dt <= 0.0:
            raise ValueError("a completed control interval must have a positive duration")

        before = self.snapshot
        unloaded = metrics.fully_unloaded
        net_gap = metrics.net_gap_m
        com_vz = metrics.com_vertical_velocity_mps
        in_window = validated.in_bonus_window(tick)

        run_onset = False
        if unloaded:
            if self._consecutive_unloaded_intervals == 0:
                # A prospective run may only be certified when it began while the
                # whole-robot COM was actually rising.
                run_onset = True
                self._run_upward_onset = com_vz > 0.0
                self._run_maximum_net_gap_m = net_gap
            else:
                self._run_maximum_net_gap_m = max(self._run_maximum_net_gap_m, net_gap)
            self._consecutive_unloaded_intervals += 1
            self._unloaded_native_samples += NATIVE_SAMPLES_PER_CONTROL_INTERVAL
        else:
            self._consecutive_unloaded_intervals = 0
            self._run_maximum_net_gap_m = 0.0
            self._run_upward_onset = False
            if not self._flight_seen:
                self._unloaded_native_samples = 0

        newly_certified = bool(
            unloaded
            and in_window
            and not self._flight_seen
            and self._run_upward_onset
            and self._consecutive_unloaded_intervals >= CERTIFYING_UNLOADED_INTERVALS
        )
        credited = 0.0
        if newly_certified:
            self._flight_seen = True
            # Deterministic accounting of the just-certified run: the maximum net
            # gap actually observed inside that two-interval run is credited once.
            self._certified_run_maximum_net_gap_m = float(self._run_maximum_net_gap_m)
            credited = float(self._run_maximum_net_gap_m)
        elif unloaded and in_window and self._flight_seen:
            credited = net_gap

        if credited > 0.0 and not validated.is_hold:
            goal = float(validated.net_clearance_m)
            if goal <= 0.0:  # pragma: no cover - guarded by JumpEpisodeSpec
                raise ValueError("a requested episode must carry a positive net clearance goal")
            self._progress = max(self._progress, min(1.0, max(0.0, credited / goal)))

        self._completed_intervals += 1
        after = self.snapshot
        if after.progress < before.progress - _TOLERANCE:
            raise RuntimeError("achieved clearance progress must never decrease")
        return JumpProgressEvent(
            schema=JUMP_PROGRESS_SCHEMA,
            executed_tick=tick,
            dt_s=dt,
            in_bonus_window=in_window,
            fully_unloaded=unloaded,
            net_gap_m=net_gap,
            com_vertical_velocity_mps=com_vz,
            run_onset=run_onset,
            newly_certified_flight=newly_certified,
            credited_net_gap_m=credited,
            terminated=bool(terminated),
            truncated=bool(truncated),
            before=before,
            after=after,
        )


# ---------------------------------------------------------------------------
# pure observation extension
# ---------------------------------------------------------------------------


def append_jump_observation(
    base85: Any,
    published_state: Any,
    prepared_tick: Any,
    spec: Any,
    progress: Any,
    initial_origin: Any,
) -> np.ndarray:
    """Extend the parent 85 encoding to the finite 95-entry jump observation.

    The first 85 entries are preserved byte for byte.  Gate and readout time is
    the *prepared* decision tick, never the just-executed phase.  Nothing about
    a future request time or goal is exposed before the request: entry 85 is
    ``-1`` and entry 86 is ``0`` at every tick before ``s`` and throughout a
    no-jump hold.  This function is pure and advances no state.
    """
    validated = _validated_spec(spec)
    if not isinstance(progress, (JumpProgress, JumpProgressSnapshot)):
        raise TypeError(
            "progress must be a JumpProgress or JumpProgressSnapshot, got "
            f"{type(progress).__name__}"
        )
    base = base85 if isinstance(base85, np.ndarray) else np.asarray(base85)
    if base.shape != (JUMP_PARENT_OBSERVATION_SIZE,) or base.dtype != np.float32:
        raise ValueError(
            "the parent task must supply the exact "
            f"{JUMP_PARENT_OBSERVATION_SIZE}-dimensional float32 encoding"
        )
    if not np.all(np.isfinite(base)):
        raise ValueError("the parent observation prefix must be finite")
    tick = _validated_tick(prepared_tick, "prepared_tick")
    origin = _read_only_array(initial_origin, (2,), "initial_origin")

    contact = getattr(published_state, "wheel_contact", None)
    position = getattr(published_state, "base_position", None)
    if contact is None or position is None:
        raise TypeError(
            "published_state must expose wheel_contact and base_position from the "
            "published oracle state"
        )
    contact_array = np.asarray(contact)
    if contact_array.shape != (len(JUMP_LEG_ORDER),):
        raise ValueError("published wheel_contact must carry one flag per leg")
    if contact_array.dtype != np.bool_:
        raise TypeError("published wheel_contact must be a strict boolean array")
    base_position = _read_only_array(position, (3,), "published base_position")

    requested = not validated.is_hold and tick >= int(validated.request_tick or 0)
    if requested:
        elapsed = (tick - int(validated.request_tick)) / BONUS_WINDOW_TICKS
        request_clock = min(1.0, max(0.0, elapsed))
        goal_feature = float(validated.net_clearance_m) / NET_CLEARANCE_OBSERVATION_SCALE_M
        progress_feature = float(progress.progress)
    else:
        request_clock = -1.0
        goal_feature = 0.0
        progress_feature = 0.0

    offset = (np.asarray(base_position[:2], dtype=np.float64) - origin) / _PLANAR_OFFSET_SCALE_M
    appended = np.array(
        (
            request_clock,
            goal_feature,
            *(1.0 if bool(flag) else 0.0 for flag in contact_array),
            progress_feature,
            float(progress.airborne_progress),
            float(offset[0]),
            float(offset[1]),
        ),
        dtype=np.float64,
    )
    if appended.shape != (_APPENDED_COUNT,):  # pragma: no cover - guarded above
        raise RuntimeError("the appended jump observation must carry exactly ten entries")
    if not np.all(np.isfinite(appended)):
        raise ValueError("the appended jump observation must be finite")
    low, high = JUMP_OBSERVATION_CLIP
    appended = np.clip(appended, low, high)

    result = np.concatenate(
        (base.astype(np.float32, copy=True), appended.astype(np.float32))
    ).astype(np.float32, copy=False)
    if result.shape != (JUMP_OBSERVATION_SIZE,) or not np.all(np.isfinite(result)):
        raise RuntimeError(
            f"the jump observation must be a finite {JUMP_OBSERVATION_SIZE}-entry vector"
        )
    if result[:JUMP_PARENT_OBSERVATION_SIZE].tobytes() != base.tobytes():
        raise RuntimeError("the parent observation prefix must be preserved byte for byte")
    return result


# ---------------------------------------------------------------------------
# fixed reward composition
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JumpRewardBreakdown:
    """Named, immutable decomposition of the new PPO scalar for one interval."""

    schema: str
    reward: float
    reward_terms: Mapping[str, float]
    parent_reward_terms: Mapping[str, float]
    parent_reward: float
    height_term_original: float
    height_term_effective: float
    height_term_masked: bool
    clearance_progress: float
    flight_once: float
    stationary_offset: float
    landing_once: float
    landing_conditions: Mapping[str, bool]

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready copy of the reward decomposition."""
        return {
            "schema": str(self.schema),
            "reward": float(self.reward),
            "reward_terms": {str(k): float(v) for k, v in self.reward_terms.items()},
            "parent_reward_terms": {
                str(k): float(v) for k, v in self.parent_reward_terms.items()
            },
            "parent_reward": float(self.parent_reward),
            "height_term_original": float(self.height_term_original),
            "height_term_effective": float(self.height_term_effective),
            "height_term_masked": bool(self.height_term_masked),
            "clearance_progress": float(self.clearance_progress),
            "flight_once": float(self.flight_once),
            "stationary_offset": float(self.stationary_offset),
            "landing_once": float(self.landing_once),
            "landing_conditions": {str(k): bool(v) for k, v in self.landing_conditions.items()},
        }


def _validated_parent_terms(parent_terms: Any) -> dict[str, float]:
    if not isinstance(parent_terms, Mapping):
        raise TypeError(
            f"parent_terms must be a mapping of original reward terms, got "
            f"{type(parent_terms).__name__}"
        )
    if PARENT_HEIGHT_TERM_NAME not in parent_terms:
        raise ValueError(
            f"parent_terms must retain the original {PARENT_HEIGHT_TERM_NAME!r} contribution"
        )
    terms: dict[str, float] = {}
    for name, value in parent_terms.items():
        if not isinstance(name, str) or not name:
            raise TypeError("parent reward term names must be nonempty strings")
        terms[name] = _finite_float(value, f"parent_terms[{name!r}]")
    for reserved in ("clearance_progress", "flight_once", "stationary_offset", "landing_once"):
        if reserved in terms:
            raise ValueError(f"the parent reward already publishes a {reserved!r} term")
    return terms


def _validated_snapshot(value: Any, name: str) -> JumpProgressSnapshot:
    if isinstance(value, JumpProgress):
        return value.snapshot
    if not isinstance(value, JumpProgressSnapshot):
        raise TypeError(
            f"{name} must be a JumpProgressSnapshot or JumpProgress, got {type(value).__name__}"
        )
    return value


def _landing_conditions(
    spec: JumpEpisodeSpec,
    executed_tick: int,
    endpoint_tick: int,
    progress_after: JumpProgressSnapshot,
    metrics: JumpEndpointMetrics,
    terminated: bool,
    truncated: bool,
) -> dict[str, bool]:
    """Evaluate the fixed final-transition landing conditions."""
    return {
        "genuine_final_endpoint": bool(
            endpoint_tick == JUMP_HORIZON_TICKS and executed_tick == JUMP_HORIZON_TICKS - 1
        ),
        "not_terminated": bool(not terminated),
        "time_limit_truncated": bool(truncated),
        "requested_jump": bool(not spec.is_hold),
        "flight_seen": bool(progress_after.flight_seen),
        "full_progress": bool(progress_after.progress >= 1.0 - _TOLERANCE),
        "all_wheels_positively_loaded": metrics.all_wheels_positively_loaded,
        "base_height": bool(
            abs(float(metrics.base_position_m[2]) - LANDING_BASE_HEIGHT_M)
            <= LANDING_HEIGHT_TOLERANCE_M
        ),
        "body_forward_velocity": bool(
            abs(float(metrics.body_forward_velocity_mps)) <= LANDING_BODY_VX_LIMIT_MPS
        ),
        "com_vertical_velocity": bool(
            abs(metrics.com_vertical_velocity_mps) <= LANDING_COM_VZ_LIMIT_MPS
        ),
        "roll_pitch": bool(
            max(abs(float(metrics.base_rpy_rad[0])), abs(float(metrics.base_rpy_rad[1])))
            <= LANDING_ATTITUDE_LIMIT_RAD
        ),
        "heading_error": bool(
            abs(float(metrics.heading_error_rad)) <= LANDING_HEADING_LIMIT_RAD
        ),
        "planar_displacement": bool(
            metrics.planar_displacement_m <= LANDING_DISPLACEMENT_LIMIT_M
        ),
    }


def compose_jump_reward(
    parent_terms: Any,
    executed_tick: Any,
    spec: Any,
    dt: Any,
    progress_before: Any,
    progress_after: Any,
    endpoint_metrics: Any,
    terminated: Any,
    truncated: Any,
    endpoint_tick: Any,
) -> JumpRewardBreakdown:
    """Compose the fixed new PPO scalar for one executed control interval.

    Every original parent contribution is retained and separately available in
    ``parent_reward_terms``.  Only the *effective* height contribution is masked
    to zero on a requested jump's predeclared execution ticks ``[s+25, s+120)``.
    The event contributions 2 / 1 / 2 are bounded across one episode, and the
    landing bonus exists only at a genuine completed 600th transition.
    """
    validated = _validated_spec(spec)
    original = _validated_parent_terms(parent_terms)
    before = _validated_snapshot(progress_before, "progress_before")
    after = _validated_snapshot(progress_after, "progress_after")
    if not isinstance(endpoint_metrics, JumpEndpointMetrics):
        raise TypeError(
            "endpoint_metrics must be a JumpEndpointMetrics, got "
            f"{type(endpoint_metrics).__name__}"
        )
    for name, flag in (("terminated", terminated), ("truncated", truncated)):
        if not isinstance(flag, (bool, np.bool_)):
            raise TypeError(f"{name} must be a boolean")
    tick = _exact_integer(executed_tick, "executed_tick")
    if tick < 0 or tick >= JUMP_HORIZON_TICKS:
        raise ValueError(
            f"executed_tick must index one of the {JUMP_HORIZON_TICKS} executions, got {tick}"
        )
    endpoint = _exact_integer(endpoint_tick, "endpoint_tick")
    if endpoint != tick + 1:
        raise ValueError(
            f"endpoint_tick must be executed_tick + 1 = {tick + 1}, got {endpoint}"
        )
    actual_dt = _finite_float(dt, "dt")
    if actual_dt <= 0.0:
        raise ValueError("a completed control interval must have a positive duration")
    if after.progress < before.progress - _TOLERANCE:
        raise ValueError("achieved clearance progress must never decrease")

    masked = validated.masks_height_term(tick)
    height_original = original[PARENT_HEIGHT_TERM_NAME]
    height_effective = 0.0 if masked else height_original

    clearance_progress = CLEARANCE_PROGRESS_WEIGHT * (after.progress - before.progress)
    if clearance_progress < -_TOLERANCE or clearance_progress > CLEARANCE_PROGRESS_WEIGHT:
        raise RuntimeError("the clearance-progress contribution left its bounded range")
    flight_once = (
        FLIGHT_ONCE_BONUS if after.flight_seen and not before.flight_seen else 0.0
    )
    displacement = endpoint_metrics.planar_displacement_m / STATIONARY_OFFSET_SCALE_M
    stationary_offset = (
        -actual_dt
        * STATIONARY_OFFSET_WEIGHT
        * min(displacement * displacement, STATIONARY_OFFSET_SATURATION)
    )
    conditions = _landing_conditions(
        validated, tick, endpoint, after, endpoint_metrics, bool(terminated), bool(truncated)
    )
    landing_once = LANDING_ONCE_BONUS if all(conditions.values()) else 0.0

    effective = {
        **original,
        PARENT_HEIGHT_TERM_NAME: height_effective,
        "clearance_progress": clearance_progress,
        "flight_once": flight_once,
        "stationary_offset": stationary_offset,
        "landing_once": landing_once,
    }
    reward = math.fsum(effective.values())
    if not math.isfinite(reward):
        raise RuntimeError("the composed jump reward must be finite")
    return JumpRewardBreakdown(
        schema=JUMP_REWARD_SCHEMA,
        reward=float(reward),
        reward_terms=MappingProxyType(dict(effective)),
        parent_reward_terms=MappingProxyType(dict(original)),
        parent_reward=float(math.fsum(original.values())),
        height_term_original=float(height_original),
        height_term_effective=float(height_effective),
        height_term_masked=bool(masked),
        clearance_progress=float(clearance_progress),
        flight_once=float(flight_once),
        stationary_offset=float(stationary_offset),
        landing_once=float(landing_once),
        landing_conditions=MappingProxyType(dict(conditions)),
    )
