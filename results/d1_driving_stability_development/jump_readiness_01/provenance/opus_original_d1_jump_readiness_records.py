"""Native records, non-integrating geometry reconstruction and the pure
readiness scorer for the D1 stationary height-profile hop diagnostic.

Three responsibilities are kept strictly separate:

* :class:`NativeStepObserver` is a narrow context-manager seam around the
  already-existing ``mujoco.mj_step`` entry.  It records one entry per actual
  native call of the target model/data, calls the original step exactly once and
  restores the original binding on every exit path, including entry/exit
  failures.  It never adds a step and never solves a contact force: contact
  numbers come from the already-solved cache and are tagged accordingly.
* :class:`ScratchKinematics` reconstructs collision geometry and whole-robot COM
  state from saved poses on one scratch ``MjData`` per episode, using only
  ``mj_kinematics``, ``mj_comPos`` and ``mj_jacBodyCom``.  No integration, no
  force solve, no collision call, no reset and no write to the real plant.
* :func:`score_readiness` is pure: it reads a stored record payload and never
  touches an environment, model, plant or reconstruction.

Sampled runs describe saved native evaluations only; they are not a claim about
unsampled continuous time.  A readiness result is a new development screen and
is never an old G1 pass/fail.
"""
from __future__ import annotations

import hashlib
import json
import math
from contextlib import AbstractContextManager
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

from scripts.d1_jump_readiness import sampled_true_runs

READINESS_RECORDS_SCHEMA = "d1-height-profile-hop-readiness-records-v1"
READINESS_SCORE_SCHEMA = "d1-height-profile-hop-readiness-score-v1"
NATIVE_CACHE_TAG = "native_step_solved_cache_not_synchronized_endpoint"
SCRATCH_GEOMETRY_TAG = "scratch_kinematics_no_integration_no_force_solve"

CONTROL_DT_S = 0.01
NATIVE_DT_S = 0.002
SUBSTEPS_PER_TICK = 5
CONTROL_INTERVALS = 600
NATIVE_INTERVALS = CONTROL_INTERVALS * SUBSTEPS_PER_TICK
ENDPOINT_COUNT = CONTROL_INTERVALS + 1
#: Fixed absolute horizon of one episode: the native clock must run 0..6 s.
EPISODE_HORIZON_S = CONTROL_INTERVALS * CONTROL_DT_S

#: Frozen screening numbers copied from the accepted planning package; no value
#: is tuned, searched or relaxed here.
REQUEST_TICK = 200
LIFT_MIN_INTERVALS = 10
LIFT_MIN_DURATION_S = 0.02
TIME_TOL_S = 1.0e-10
FIRST_HURDLE_M = 0.02
RETURN_TIME_MAX_S = 4.0
LATE_FIRST_ENDPOINT = 400
LATE_LAST_ENDPOINT = 600
LATE_HEIGHT_RMSE_MAX_M = 0.015
LATE_ABS_BODY_VX_MAX_MPS = 0.03
LATE_ABS_COM_VZ_MAX_MPS = 0.03
LATE_PLANAR_PATH_MAX_M = 0.05
LATE_NATIVE_INTERVALS = 1000
PER_WHEEL_POSITIVE_LOAD_FRACTION_MIN = 0.95
GLOBAL_ABS_ROLL_PITCH_MAX_RAD = 0.17453292519943295
GLOBAL_ABS_HEADING_ERROR_MAX_RAD = 0.08726646259971647
GLOBAL_PLANAR_DISPLACEMENT_MAX_M = 0.1

#: First native returned endpoint that belongs to the requested height profile.
REQUEST_NATIVE_INDEX = REQUEST_TICK * SUBSTEPS_PER_TICK
#: Number of 2 ms native returned endpoints scored for useful clearance.
REQUEST_NATIVE_ENDPOINTS = NATIVE_INTERVALS - REQUEST_NATIVE_INDEX

_DT_TOL_S = 1.0e-9
#: Tolerance of a recorded absolute native clock against its fixed grid slot.
_ABS_TIME_TOL_S = 1.0e-9


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def array_sha256(*arrays: Any) -> str:
    """Stable digest of reconstruction inputs (exact bytes, signed zero kept)."""
    digest = hashlib.sha256()
    for item in arrays:
        arr = np.ascontiguousarray(np.asarray(item, dtype=np.float64))
        digest.update(str(arr.shape).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _finite_list(values: Any) -> bool:
    try:
        items = list(values)
    except TypeError:
        return False
    return bool(items) and all(_finite(v) for v in items)


def _exact_int(value: Any) -> int | None:
    """Exact non-boolean integer, or ``None`` for anything else.

    Used for record identity fields (native indices, endpoint ticks) where a
    float, string, boolean or missing value must reject the record instead of
    being coerced into a passing index.
    """
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return None


def _nonneg_counts(values: Any) -> List[int] | None:
    """Four exact non-negative contact counts, or ``None`` when malformed."""
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return None
    counts = [_exact_int(v) for v in values]
    if any(c is None or c < 0 for c in counts):
        return None
    return [int(c) for c in counts]


def _nonneg_loads(values: Any) -> List[float] | None:
    """Four finite non-negative actual normal loads, or ``None`` when malformed."""
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return None
    loads: List[float] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not _finite(value):
            return None
        number = float(value)
        if number < 0.0:
            return None
        loads.append(number)
    return loads


def _copy(value: Any) -> np.ndarray:
    return np.array(value, dtype=np.float64, copy=True)


# ---------------------------------------------------------------------------
# Native observer (records only; never adds a step or solves a force)
# ---------------------------------------------------------------------------


class NativeStepObserver(AbstractContextManager):
    """Record one entry per actual native step of ``(model, data)``.

    The original binding is captured on entry and restored on every exit path.
    Foreign models/data pass through untouched and unrecorded.  ``contact_reader``
    is called on the already-solved cache after the step; the observer verifies
    that this read changes neither pose nor warm-start data.
    """

    def __init__(self, module: Any, model: Any, data: Any, *, contact_reader=None,
                 plant: Any = None, attribute: str = "mj_step") -> None:
        self.module = module
        self.model = model
        self.data = data
        self.contact_reader = contact_reader
        self.plant = plant
        self.attribute = attribute
        self.entries: List[Dict[str, Any]] = []
        self.attempted_calls = 0
        self.returned_calls = 0
        self.foreign_calls = 0
        self.failed_calls = 0
        self.bound = False
        self._original: Any = None
        self._captured = False

    def __enter__(self) -> "NativeStepObserver":
        self._original = getattr(self.module, self.attribute)
        self._captured = True
        try:
            setattr(self.module, self.attribute, self._observed)
        except BaseException:
            setattr(self.module, self.attribute, self._original)
            raise
        self.bound = True
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # The original binding is restored on every exit path, including a step
        # failure, a contact-reader failure and a body exception.
        if self._captured:
            setattr(self.module, self.attribute, self._original)
        self.bound = False
        return False

    @property
    def original_step(self) -> Any:
        return self._original

    def _warm_snapshot(self) -> Dict[str, Any]:
        return {
            "qpos": _copy(self.data.qpos),
            "qvel": _copy(self.data.qvel),
            "qacc_warmstart": _copy(getattr(self.data, "qacc_warmstart", ())),
            "time": float(self.data.time),
        }

    def _observed(self, model: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
        if model is not self.model or data is not self.data:
            self.foreign_calls += 1
            return self._original(model, data, *args, **kwargs)

        entry: Dict[str, Any] = {
            "index": len(self.entries),
            "start_time_s": float(data.time),
            "qpos_before": _copy(data.qpos),
            "qvel_before": _copy(data.qvel),
            "ctrl_nm": _copy(data.ctrl),
            "xfrc_applied": _copy(getattr(data, "xfrc_applied", ())),
            "contact_sample_tag": NATIVE_CACHE_TAG,
            "returned": False,
            "error": None,
        }
        self.entries.append(entry)
        self.attempted_calls += 1
        try:
            result = self._original(model, data, *args, **kwargs)
        except BaseException as error:
            self.failed_calls += 1
            entry["error"] = {"type": type(error).__name__, "message": str(error)}
            entry["qpos_after_error"] = _copy(data.qpos)
            entry["qvel_after_error"] = _copy(data.qvel)
            entry["end_time_s"] = float(data.time)
            entry["actual_dt_s"] = float(data.time) - entry["start_time_s"]
            raise
        # Returned endpoint is copied immediately from the stepped data, not from
        # any measurement cache.
        entry["returned"] = True
        entry["qpos_returned"] = _copy(data.qpos)
        entry["qvel_returned"] = _copy(data.qvel)
        entry["end_time_s"] = float(data.time)
        entry["actual_dt_s"] = float(data.time) - entry["start_time_s"]
        self.returned_calls += 1
        if self.contact_reader is not None:
            before = self._warm_snapshot()
            sample = self.contact_reader(self.plant if self.plant is not None else data)
            after = self._warm_snapshot()
            for key in ("qpos", "qvel", "qacc_warmstart"):
                if before[key].tobytes() != after[key].tobytes():
                    raise RuntimeError(
                        "native contact observer changed integrator or warm start")
            if before["time"] != after["time"]:
                raise RuntimeError("native contact observer changed native time")
            entry["contacts"] = sample
        return result

    def receipt(self) -> Dict[str, Any]:
        return {
            "schema": READINESS_RECORDS_SCHEMA,
            "attempted_native_calls": int(self.attempted_calls),
            "returned_native_calls": int(self.returned_calls),
            "failed_native_calls": int(self.failed_calls),
            "foreign_pass_through_calls": int(self.foreign_calls),
            "recorded_entries": len(self.entries),
            "observer_bound": bool(self.bound),
            "contact_sample_tag": NATIVE_CACHE_TAG,
        }


def per_wheel_contact_detail(model: Any, data: Any, binding: Any, *,
                             force_reader: Callable[..., Any]) -> Dict[str, Any]:
    """Read-only per-wheel geometric/active contact counts and normal loads.

    ``force_reader(model, data, index, buffer)`` must fill an existing solved
    force buffer; no collision detection or force solve is performed here.  An
    active solver contact (``efc_address >= 0``) with zero force is reported
    separately from the absence of a geometric contact, and invalid (negative or
    non-finite) magnitudes are flagged rather than clamped to zero.
    """
    wheel_geoms = list(binding.wheel_geom_ids)
    plane_geom = int(binding.plane_geom_id)
    geometric = [0, 0, 0, 0]
    active = [0, 0, 0, 0]
    load = [0.0, 0.0, 0.0, 0.0]
    invalid: List[Dict[str, Any]] = []
    nonwheel = 0
    ncon = int(data.ncon)
    buffer = np.zeros(6, dtype=np.float64)
    for index in range(ncon):
        contact = data.contact[index]
        pair = (int(contact.geom1), int(contact.geom2))
        wheel = [w for w in pair if w in wheel_geoms]
        if len(wheel) != 1 or plane_geom not in pair:
            nonwheel += 1
            continue
        slot = wheel_geoms.index(wheel[0])
        geometric[slot] += 1
        address = int(getattr(contact, "efc_address", -1))
        if address < 0:
            continue
        active[slot] += 1
        buffer[:] = 0.0
        force_reader(model, data, index, buffer)
        normal = float(buffer[0])
        if not math.isfinite(normal) or normal < 0.0:
            invalid.append({"contact_index": index, "wheel": slot, "normal_n": normal})
            continue
        load[slot] += normal
    return {
        "schema": READINESS_RECORDS_SCHEMA,
        "contact_sample_tag": NATIVE_CACHE_TAG,
        "geometric_contact_count": geometric,
        "active_contact_count": active,
        "wheel_normal_load_n": load,
        "invalid_normal_values": invalid,
        # Explicit per-sample verdict: a sample carrying a negative or non-finite
        # solved magnitude is rejected by the scorer instead of reading its
        # remaining wheels as a valid zero load.
        "invalid_load": bool(invalid),
        "nonwheel_contact_count": int(nonwheel),
        "total_contacts": ncon,
        "load_source": "solved cache mj_contactForce; never the controller support request",
    }


# ---------------------------------------------------------------------------
# Non-integrating geometry / COM reconstruction
# ---------------------------------------------------------------------------


def whole_body_com(masses: Sequence[float], body_com_world_m: Any) -> np.ndarray:
    """Mass-weighted COM of the supplied bodies (never a base-origin value)."""
    weights = _copy(masses)
    points = _copy(body_com_world_m)
    if weights.ndim != 1 or points.shape != (weights.size, 3):
        raise ValueError("masses and body COM positions must describe the same bodies")
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("total body mass must be finite and positive")
    return (weights @ points) / total


def whole_body_com_velocity(masses: Sequence[float], com_jacobians: Any,
                            qvel: Any) -> np.ndarray:
    """Mass-weighted COM velocity from per-body COM Jacobians and one qvel."""
    weights = _copy(masses)
    jacobians = _copy(com_jacobians)
    velocity = _copy(qvel)
    if weights.ndim != 1 or jacobians.ndim != 3 or jacobians.shape[0] != weights.size:
        raise ValueError("com_jacobians must be (nbody, 3, nv) for the supplied masses")
    if jacobians.shape[1] != 3 or jacobians.shape[2] != velocity.size:
        raise ValueError("com_jacobians must map the supplied qvel")
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("total body mass must be finite and positive")
    stacked = jacobians @ velocity
    return (weights @ stacked) / total


def subtree_body_ids(body_parentid: Sequence[int], root_body_id: int) -> List[int]:
    """Real robot bodies of the base subtree, in compiled order."""
    parents = [int(p) for p in body_parentid]
    root = int(root_body_id)
    members = [root]
    for body in range(len(parents)):
        if body <= root:
            continue
        parent = parents[body]
        if parent in members:
            members.append(body)
    return members


class KinematicsApi:
    """Explicit allow-list of MuJoCo entries used for reconstruction."""

    def __init__(self, make_data, mj_kinematics, mj_com_pos, mj_jac_body_com) -> None:
        self.make_data = make_data
        self.mj_kinematics = mj_kinematics
        self.mj_comPos = mj_com_pos
        self.mj_jacBodyCom = mj_jac_body_com


def default_kinematics_api() -> KinematicsApi:
    import mujoco  # local import: kinematics only, never an integration entry

    return KinematicsApi(mujoco.MjData, mujoco.mj_kinematics, mujoco.mj_comPos,
                         mujoco.mj_jacBodyCom)


class ScratchKinematics:
    """One reusable scratch ``MjData`` for pose-preserving reconstruction."""

    allowed_api_names = ("mj_kinematics", "mj_comPos", "mj_jacBodyCom")

    def __init__(self, model: Any, *, body_ids: Sequence[int], masses: Sequence[float],
                 base_body_id: int, api: KinematicsApi | None = None) -> None:
        self.model = model
        self.api = api if api is not None else default_kinematics_api()
        self.body_ids = [int(b) for b in body_ids]
        self.masses = _copy(masses)
        if self.masses.shape != (len(self.body_ids),):
            raise ValueError("masses must match the supplied body ids")
        self.base_body_id = int(base_body_id)
        self.scratch = self.api.make_data(model)
        self.api_calls = {name: 0 for name in self.allowed_api_names}
        self.reconstructions = 0

    def reconstruct(self, qpos: Any, qvel: Any, *, label: str,
                    time_s: float | None = None) -> Dict[str, Any]:
        position = _copy(qpos)
        velocity = _copy(qvel)
        data = self.scratch
        data.qpos[:] = position
        data.qvel[:] = velocity
        self.api.mj_kinematics(self.model, data)
        self.api_calls["mj_kinematics"] += 1
        self.api.mj_comPos(self.model, data)
        self.api_calls["mj_comPos"] += 1
        jacobians = np.zeros((len(self.body_ids), 3, velocity.size), dtype=np.float64)
        for slot, body in enumerate(self.body_ids):
            jacp = np.zeros((3, velocity.size), dtype=np.float64)
            self.api.mj_jacBodyCom(self.model, data, jacp, None, body)
            self.api_calls["mj_jacBodyCom"] += 1
            jacobians[slot] = jacp
        xipos = _copy(data.xipos)[self.body_ids, :]
        com = whole_body_com(self.masses, xipos)
        com_velocity = whole_body_com_velocity(self.masses, jacobians, velocity)
        self.reconstructions += 1
        return {
            "schema": READINESS_RECORDS_SCHEMA,
            "tag": SCRATCH_GEOMETRY_TAG,
            "label": str(label),
            "time_s": None if time_s is None else float(time_s),
            "input_sha256": array_sha256(position, velocity),
            "geom_xpos": _copy(data.geom_xpos),
            "geom_xmat": _copy(data.geom_xmat),
            "com_position_m": [float(v) for v in com],
            "com_velocity_mps": [float(v) for v in com_velocity],
            "base_origin_height_m": float(position[2]),
            "base_quaternion": [float(v) for v in position[3:7]],
            "qpos": position,
            "qvel": velocity,
        }

    def receipt(self) -> Dict[str, Any]:
        return {
            "schema": READINESS_RECORDS_SCHEMA,
            "tag": SCRATCH_GEOMETRY_TAG,
            "api_calls": dict(self.api_calls),
            "reconstructions": int(self.reconstructions),
            "body_ids": list(self.body_ids),
            "base_body_id": int(self.base_body_id),
            "total_mass_kg": float(np.sum(self.masses)),
            "forbidden_entries": ["mj_step", "mj_step1", "mj_step2", "mj_forward",
                                  "mj_inverse", "mj_collision", "mj_resetData"],
        }


# ---------------------------------------------------------------------------
# Pure scoring
# ---------------------------------------------------------------------------


def _endpoints(episode: Dict[str, Any]) -> List[Dict[str, Any]]:
    value = episode.get("endpoints")
    return list(value) if isinstance(value, list) else []


def _intervals(episode: Dict[str, Any]) -> List[Dict[str, Any]]:
    value = episode.get("intervals")
    return list(value) if isinstance(value, list) else []


def _native_record_timeline(intervals: List[Dict[str, Any]]) -> tuple[bool, str]:
    """Strict native record identity, clock and horizon validation.

    Rejects a missing, duplicated or out-of-order native index, a call that did
    not report ``returned is True``, a stored ``actual_dt_s`` that disagrees with
    the recorded start/end clock, a substep that is not the fixed 2 ms and any
    departure from the fixed absolute 0..6 s horizon.
    """
    if len(intervals) != NATIVE_INTERVALS:
        return False, (f"expected {NATIVE_INTERVALS} recorded native intervals, "
                       f"got {len(intervals)}")
    previous_end: float | None = None
    for slot, interval in enumerate(intervals):
        if not isinstance(interval, dict):
            return False, f"native interval at position {slot} is not a record"
        index = _exact_int(interval.get("index"))
        if index != slot:
            return False, (f"native index {interval.get('index')!r} at position {slot} is "
                           "missing, duplicated or out of order")
        if interval.get("returned") is not True:
            return False, f"native interval {slot} does not report returned is True"
        start = interval.get("start_time_s")
        end = interval.get("end_time_s")
        dt = interval.get("actual_dt_s")
        if not (_finite(start) and _finite(end) and _finite(dt)):
            return False, f"native interval {slot} has a non-finite recorded clock"
        start, end, dt = float(start), float(end), float(dt)
        if end <= start:
            return False, f"native interval {slot} end_time_s must exceed start_time_s"
        if abs((end - start) - dt) > _DT_TOL_S:
            return False, (f"native interval {slot} actual_dt_s disagrees with its recorded "
                           "start/end clock")
        if abs(dt - NATIVE_DT_S) > _DT_TOL_S:
            return False, (f"native interval {slot} actual_dt_s is not the fixed "
                           f"{NATIVE_DT_S} s substep")
        if abs(start - slot * NATIVE_DT_S) > _ABS_TIME_TOL_S:
            return False, (f"native interval {slot} start_time_s leaves the fixed "
                           f"0..{EPISODE_HORIZON_S} s horizon")
        if abs(end - (slot + 1) * NATIVE_DT_S) > _ABS_TIME_TOL_S:
            return False, (f"native interval {slot} end_time_s leaves the fixed "
                           f"0..{EPISODE_HORIZON_S} s horizon")
        if previous_end is not None and abs(start - previous_end) > TIME_TOL_S:
            return False, (f"native interval {slot} does not continue the previous returned "
                           "endpoint clock")
        previous_end = end
    if previous_end is None or abs(previous_end - EPISODE_HORIZON_S) > _ABS_TIME_TOL_S:
        return False, f"recorded native clock does not end at the fixed {EPISODE_HORIZON_S} s"
    return True, ""


def _native_contact_fields(intervals: List[Dict[str, Any]]) -> tuple[bool, str]:
    """Four-wheel contact/load field validation of every native record.

    Actual normal load must be finite and non-negative; a record flagged
    ``invalid_load`` -- or one that never declares the flag -- is rejected rather
    than read as a valid zero load.
    """
    if len(intervals) != NATIVE_INTERVALS:
        return False, (f"expected {NATIVE_INTERVALS} recorded native intervals, "
                       f"got {len(intervals)}")
    for slot, interval in enumerate(intervals):
        if not isinstance(interval, dict):
            return False, f"native interval at position {slot} is not a record"
        if _nonneg_counts(interval.get("active_wheel_contacts")) is None:
            return False, f"native interval {slot} lacks four active wheel contact counts"
        geometric = interval.get("geometric_wheel_contacts")
        if geometric is not None and _nonneg_counts(geometric) is None:
            return False, f"native interval {slot} has malformed geometric contact counts"
        if _nonneg_loads(interval.get("wheel_normal_load_n")) is None:
            return False, (f"native interval {slot} lacks four finite non-negative actual "
                           "wheel normal loads")
        if interval.get("invalid_load") is not False:
            return False, (f"native interval {slot} is flagged invalid_load or never declares "
                           "a valid actual load")
        invalid_values = interval.get("invalid_normal_values")
        if invalid_values is not None and list(invalid_values):
            return False, f"native interval {slot} carries invalid solved normal values"
    return True, ""


def _endpoint_tick_sequence(endpoints: List[Dict[str, Any]]) -> tuple[bool, str]:
    """Unique, ordered endpoint ticks 0..600 with an agreeing control clock."""
    if len(endpoints) != ENDPOINT_COUNT:
        return False, f"expected {ENDPOINT_COUNT} endpoint records, got {len(endpoints)}"
    for slot, endpoint in enumerate(endpoints):
        if not isinstance(endpoint, dict):
            return False, f"endpoint at position {slot} is not a record"
        tick = _exact_int(endpoint.get("tick"))
        if tick != slot:
            return False, (f"endpoint tick {endpoint.get('tick')!r} at position {slot} is "
                           "missing, duplicated or out of order")
        time_s = endpoint.get("time_s")
        if time_s is not None:
            if not _finite(time_s) or abs(float(time_s) - slot * CONTROL_DT_S) > _ABS_TIME_TOL_S:
                return False, f"endpoint {slot} time_s disagrees with the fixed control clock"
    return True, ""


def _execution_valid(episode: Dict[str, Any]) -> Dict[str, Any]:
    flags = dict(episode.get("execution") or {})
    endpoints = _endpoints(episode)
    intervals = _intervals(episode)
    completed = _exact_int(episode.get("completed_control_intervals"))
    checks: Dict[str, Any] = {
        "completed_control_intervals": completed == CONTROL_INTERVALS,
        "native_interval_count": len(intervals) == NATIVE_INTERVALS,
        "endpoint_count": len(endpoints) == ENDPOINT_COUNT,
        "native_endpoints_per_control_interval": completed is not None
        and len(intervals) == SUBSTEPS_PER_TICK * completed
        and len(endpoints) == completed + 1,
        "single_reset": _exact_int(flags.get("reset_count")) == 1,
        "no_unknown_wrench": flags.get("unknown_wrench") is False,
        "no_nonwheel_contact": flags.get("nonwheel_contact") is False,
        "no_fall": flags.get("fall") is False,
        "no_domain_exit": flags.get("domain_exit") is False,
        "finite_state": flags.get("nonfinite_state") is False,
        "plane_normals_valid": flags.get("plane_normals_valid") is True,
        "protections_intact": flags.get("protections_intact") is True,
        "rated_torque_within_envelope": flags.get("rated_torque_exceeded") is False,
        "position_within_envelope": flags.get("position_envelope_exceeded") is False,
        "velocity_within_envelope": flags.get("velocity_envelope_exceeded") is False,
    }

    reasons: List[str] = []
    timeline_ok, timeline_reason = _native_record_timeline(intervals)
    contacts_ok, contacts_reason = _native_contact_fields(intervals)
    ticks_ok, ticks_reason = _endpoint_tick_sequence(endpoints)
    checks["native_time_continuous"] = timeline_ok
    checks["native_contact_fields_valid"] = contacts_ok
    checks["endpoint_tick_sequence"] = ticks_ok
    for reason in (timeline_reason, contacts_reason, ticks_reason):
        if reason:
            reasons.append(reason)

    roll_pitch_ok = bool(endpoints)
    heading_ok = bool(endpoints)
    displacement_ok = bool(endpoints)
    for endpoint in endpoints:
        roll, pitch = endpoint.get("roll_rad"), endpoint.get("pitch_rad")
        heading = endpoint.get("heading_error_rad")
        planar = endpoint.get("planar_displacement_m")
        if not (_finite(roll) and _finite(pitch) and _finite(heading) and _finite(planar)):
            roll_pitch_ok = heading_ok = displacement_ok = False
            reasons.append("non-finite endpoint attitude, heading or displacement")
            break
        roll_pitch_ok = roll_pitch_ok and max(abs(float(roll)), abs(float(pitch))) <= (
            GLOBAL_ABS_ROLL_PITCH_MAX_RAD)
        heading_ok = heading_ok and abs(float(heading)) <= GLOBAL_ABS_HEADING_ERROR_MAX_RAD
        displacement_ok = displacement_ok and float(planar) <= GLOBAL_PLANAR_DISPLACEMENT_MAX_M
    checks["global_roll_pitch"] = roll_pitch_ok
    checks["global_heading_error"] = heading_ok
    checks["global_planar_displacement"] = displacement_ok

    return {
        "checks": checks,
        "passed": all(bool(v) for v in checks.values()),
        "reasons": reasons,
        "protection_occupancy": episode.get("protection_occupancy"),
    }


def _lift_mask(intervals: List[Dict[str, Any]]) -> np.ndarray:
    mask = np.zeros(len(intervals), dtype=bool)
    for slot, interval in enumerate(intervals):
        if slot < REQUEST_NATIVE_INDEX:
            continue
        if not isinstance(interval, dict):
            continue
        if interval.get("returned") is not True or interval.get("invalid_load") is not False:
            continue
        active = _nonneg_counts(interval.get("active_wheel_contacts"))
        load = _nonneg_loads(interval.get("wheel_normal_load_n"))
        gap = interval.get("endpoint_min_gap_m")
        margin = interval.get("contact_margin_m")
        if active is None or load is None:
            continue
        if not (_finite(gap) and _finite(margin)):
            continue
        if any(count != 0 for count in active):
            continue
        if any(value != 0.0 for value in load):
            continue
        if not float(gap) > float(margin):
            continue
        mask[slot] = True
    return mask


def _physical_lift(episode: Dict[str, Any]) -> Dict[str, Any]:
    intervals = _intervals(episode)
    detail: Dict[str, Any] = {
        "evidence": "sampled recorded native intervals and returned endpoints; "
                    "not a continuous-time flight proof",
        "qualifying_run": None,
        "com_vz_at_run_start_mps": None,
    }
    if len(intervals) != NATIVE_INTERVALS:
        detail["reason"] = "missing or incomplete native intervals"
        return {"detail": detail, "passed": False}
    starts = np.array([float(i.get("start_time_s", np.nan)) for i in intervals])
    ends = np.array([float(i.get("end_time_s", np.nan)) for i in intervals])
    if not (np.all(np.isfinite(starts)) and np.all(np.isfinite(ends))):
        detail["reason"] = "non-finite native interval times"
        return {"detail": detail, "passed": False}
    mask = _lift_mask(intervals)
    try:
        runs = sampled_true_runs(mask, starts, ends)
    except (TypeError, ValueError) as error:
        detail["reason"] = f"run reduction rejected the stored intervals: {error}"
        return {"detail": detail, "passed": False}
    detail["runs"] = [dict(run) for run in runs]
    for run in runs:
        if int(run["intervals"]) < LIFT_MIN_INTERVALS:
            continue
        if float(run["summed_interval_duration_s"]) + TIME_TOL_S < LIFT_MIN_DURATION_S:
            continue
        entry = intervals[int(run["first_index"])]
        vz = entry.get("start_com_vz_mps")
        detail["qualifying_run"] = dict(run)
        detail["com_vz_at_run_start_mps"] = float(vz) if _finite(vz) else None
        if not _finite(vz) or float(vz) <= 0.0:
            detail["reason"] = "COM vertical velocity at the run start is not positive"
            return {"detail": detail, "passed": False}
        return {"detail": detail, "passed": True}
    detail["reason"] = "no run of >=10 contiguous unloaded intervals spanning >=0.020 s"
    return {"detail": detail, "passed": False}


def _control_endpoint_clearance(endpoints: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Separate 10 ms control-endpoint clearance diagnostic.

    This is reported alongside the scored 2 ms native sampling and never decides
    the gate: a peak that falls between two control endpoints must not fail the
    screen, and a control endpoint must not stand in for a native measurement.
    """
    detail: Dict[str, Any] = {
        "sampling": "10 ms control endpoints; diagnostic only, never the scored gate",
    }
    if len(endpoints) != ENDPOINT_COUNT:
        return {"detail": detail, "malformed": True, "reason": "missing control endpoints"}
    best: float | None = None
    best_tick: int | None = None
    per_wheel_peaks: List[float | None] = [None, None, None, None]
    for endpoint in endpoints:
        tick = _exact_int(endpoint.get("tick"))
        if tick is None or tick < REQUEST_TICK:
            continue
        gap = endpoint.get("min_gap_m")
        if not _finite(gap):
            return {"detail": detail, "malformed": True,
                    "reason": "non-finite control endpoint clearance"}
        if best is None or float(gap) > best:
            best, best_tick = float(gap), tick
        wheels = endpoint.get("wheel_gap_m")
        if isinstance(wheels, (list, tuple)) and len(wheels) == 4:
            for slot, value in enumerate(wheels):
                if _finite(value) and (per_wheel_peaks[slot] is None
                                       or float(value) > per_wheel_peaks[slot]):
                    per_wheel_peaks[slot] = float(value)
    if best is None:
        return {"detail": detail, "malformed": True,
                "reason": "no control endpoint at or after the request tick"}
    detail.update({
        "max_simultaneous_minimum_gap_m": best,
        "at_tick": best_tick,
        "per_wheel_peak_gap_m": per_wheel_peaks,
    })
    return {"detail": detail, "malformed": False, "reason": ""}


def _useful_clearance(episode: Dict[str, Any]) -> Dict[str, Any]:
    """Score the useful clearance screen on actual native returned endpoints.

    Every recorded 2 ms native returned endpoint from the request onwards is
    scored, so a valid simultaneous four-wheel peak between two control
    endpoints is no longer falsely failed.  The 10 ms control-endpoint peak and
    the individual wheel peaks are reported separately and never replace the
    simultaneous minimum.  The strict ``> 0.020 m + compiled margin`` comparison
    is unchanged, and a missing or non-finite native endpoint rejects the screen
    instead of scoring a convenient subset.
    """
    intervals = _intervals(episode)
    detail: Dict[str, Any] = {
        "threshold_note": "0.020 m first course hurdle + compiled margin",
        "scored_sampling": "all actual 2 ms native returned endpoints at or after the "
                           "request; a requested phase never certifies clearance",
    }
    control = _control_endpoint_clearance(_endpoints(episode))
    detail["control_endpoint_peak"] = control["detail"]
    if control["malformed"]:
        detail["reason"] = control["reason"]
        return {"detail": detail, "passed": False}
    if len(intervals) != NATIVE_INTERVALS:
        detail["reason"] = "missing native returned endpoints"
        return {"detail": detail, "passed": False}

    best: float | None = None
    best_index: int | None = None
    best_time: float | None = None
    margin: float | None = None
    per_wheel_peaks: List[float | None] = [None, None, None, None]
    scored = 0
    for slot in range(REQUEST_NATIVE_INDEX, NATIVE_INTERVALS):
        interval = intervals[slot]
        if not isinstance(interval, dict) or interval.get("returned") is not True:
            detail["reason"] = f"native returned endpoint {slot} is missing or did not return"
            return {"detail": detail, "passed": False}
        gap = interval.get("endpoint_min_gap_m")
        endpoint_margin = interval.get("contact_margin_m")
        time_s = interval.get("end_time_s")
        if not (_finite(gap) and _finite(endpoint_margin) and _finite(time_s)):
            detail["reason"] = (f"native returned endpoint {slot} has non-finite clearance "
                                "or endpoint time")
            return {"detail": detail, "passed": False}
        scored += 1
        margin = (float(endpoint_margin) if margin is None
                  else max(margin, float(endpoint_margin)))
        if best is None or float(gap) > best:
            best, best_index, best_time = float(gap), slot, float(time_s)
        wheels = interval.get("endpoint_wheel_gap_m")
        if isinstance(wheels, (list, tuple)) and len(wheels) == 4:
            for wheel, value in enumerate(wheels):
                if _finite(value) and (per_wheel_peaks[wheel] is None
                                       or float(value) > per_wheel_peaks[wheel]):
                    per_wheel_peaks[wheel] = float(value)
    if scored != REQUEST_NATIVE_ENDPOINTS or best is None or margin is None:
        detail["reason"] = "incomplete native returned endpoints after the request"
        return {"detail": detail, "passed": False}

    threshold = FIRST_HURDLE_M + float(margin)
    detail.update({
        "max_simultaneous_minimum_gap_m": best,
        "at_native_index": best_index,
        "at_native_time_s": best_time,
        "scored_native_endpoints": scored,
        "contact_margin_m": float(margin),
        "threshold_m": threshold,
        "per_wheel_peak_gap_m": per_wheel_peaks,
        "per_wheel_peaks_note": "individual peaks are reported only; they never replace "
                                "the simultaneous minimum",
    })
    return {"detail": detail, "passed": bool(best > threshold)}


def _late_settled(episode: Dict[str, Any]) -> Dict[str, Any]:
    endpoints = _endpoints(episode)
    intervals = _intervals(episode)
    detail: Dict[str, Any] = {}
    if len(endpoints) != ENDPOINT_COUNT or len(intervals) != NATIVE_INTERVALS:
        detail["reason"] = "missing endpoints or native intervals"
        return {"detail": detail, "passed": False}
    window = [e for e in endpoints
              if isinstance(e, dict)
              and _exact_int(e.get("tick")) is not None
              and LATE_FIRST_ENDPOINT <= _exact_int(e["tick"]) < LATE_LAST_ENDPOINT]
    if len(window) != LATE_LAST_ENDPOINT - LATE_FIRST_ENDPOINT:
        detail["reason"] = "late endpoint window is incomplete"
        return {"detail": detail, "passed": False}
    errors: List[float] = []
    vx_max = 0.0
    vz_max = 0.0
    for endpoint in window:
        achieved = endpoint.get("achieved_clearance_m")
        commanded = endpoint.get("commanded_clearance_m")
        vx = endpoint.get("body_vx_mps")
        vz = endpoint.get("com_vz_mps")
        if not (_finite(achieved) and _finite(commanded) and _finite(vx) and _finite(vz)):
            detail["reason"] = "non-finite late endpoint state"
            return {"detail": detail, "passed": False}
        errors.append(float(achieved) - float(commanded))
        vx_max = max(vx_max, abs(float(vx)))
        vz_max = max(vz_max, abs(float(vz)))
    rmse = float(math.sqrt(sum(e * e for e in errors) / len(errors)))

    path_points = [e for e in endpoints
                   if isinstance(e.get("tick"), int)
                   and LATE_FIRST_ENDPOINT <= int(e["tick"]) <= LATE_LAST_ENDPOINT]
    if len(path_points) != LATE_LAST_ENDPOINT - LATE_FIRST_ENDPOINT + 1:
        detail["reason"] = "late path endpoints S400..S600 are incomplete"
        return {"detail": detail, "passed": False}
    path = 0.0
    for slot in range(1, len(path_points)):
        first = path_points[slot - 1].get("planar_position_m")
        second = path_points[slot].get("planar_position_m")
        if not (_finite_list(first) and _finite_list(second)):
            detail["reason"] = "non-finite late planar positions"
            return {"detail": detail, "passed": False}
        path += float(math.dist([float(v) for v in first], [float(v) for v in second]))

    tail = intervals[-LATE_NATIVE_INTERVALS:]
    positive = [0, 0, 0, 0]
    for interval in tail:
        if not isinstance(interval, dict) or interval.get("returned") is not True:
            detail["reason"] = "missing or unreturned late native interval"
            return {"detail": detail, "passed": False}
        if interval.get("invalid_load") is not False:
            detail["reason"] = "invalid or undeclared actual normal load in the late window"
            return {"detail": detail, "passed": False}
        load = _nonneg_loads(interval.get("wheel_normal_load_n"))
        if load is None:
            detail["reason"] = "non-finite or negative late wheel normal load"
            return {"detail": detail, "passed": False}
        for wheel in range(4):
            if load[wheel] > 0.0:
                positive[wheel] += 1
    fractions = [count / float(LATE_NATIVE_INTERVALS) for count in positive]

    checks = {
        "height_rmse": rmse <= LATE_HEIGHT_RMSE_MAX_M,
        "body_vx": vx_max <= LATE_ABS_BODY_VX_MAX_MPS,
        "com_vz": vz_max <= LATE_ABS_COM_VZ_MAX_MPS,
        "planar_path": path <= LATE_PLANAR_PATH_MAX_M,
        "per_wheel_positive_load": all(
            f >= PER_WHEEL_POSITIVE_LOAD_FRACTION_MIN for f in fractions),
    }
    detail.update({
        "height_rmse_m": rmse,
        "rmse_reference": "per-tick raw held height command, never a constant 0.455 m",
        "max_abs_body_vx_mps": vx_max,
        "max_abs_com_vz_mps": vz_max,
        "planar_path_m": path,
        "per_wheel_positive_load_fraction": fractions,
        "checks": checks,
    })
    return {"detail": detail, "passed": all(checks.values())}


def _loaded_return(episode: Dict[str, Any], lift: Dict[str, Any]) -> Dict[str, Any]:
    intervals = _intervals(episode)
    detail: Dict[str, Any] = {"first_return": None}
    run = lift["detail"].get("qualifying_run")
    if not lift["passed"] or not run:
        detail["reason"] = "no qualifying airborne run to return from"
        return {"detail": detail, "passed": False}
    for interval in intervals[int(run["end_index_exclusive"]):]:
        if not isinstance(interval, dict) or interval.get("returned") is not True:
            continue
        if interval.get("invalid_load") is not False:
            # An invalid solved magnitude never certifies a measured landing.
            continue
        active = _nonneg_counts(interval.get("active_wheel_contacts"))
        load = _nonneg_loads(interval.get("wheel_normal_load_n"))
        if active is None or load is None:
            continue
        loaded = [slot for slot in range(4) if active[slot] >= 1 and load[slot] > 0.0]
        if not loaded:
            continue
        start = interval.get("start_time_s")
        detail["first_return"] = {
            "index": interval.get("index"),
            "start_time_s": float(start) if _finite(start) else None,
            "loaded_wheels": loaded,
            "active_wheel_contacts": list(active),
            "wheel_normal_load_n": list(load),
            "note": "measured loaded contact return; a requested phase is never a landing",
        }
        passed = _finite(start) and float(start) < RETURN_TIME_MAX_S
        detail["return_before_s"] = RETURN_TIME_MAX_S
        return {"detail": detail, "passed": bool(passed)}
    detail["reason"] = "no loaded wheel contact after the airborne run"
    return {"detail": detail, "passed": False}


def score_readiness(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Pure readiness score of stored records; inputs are never mutated.

    No environment, model, plant or reconstruction is touched, and no missing
    sample is treated as a zero that could pass a gate.
    """
    episodes = dict((payload or {}).get("episodes") or {})
    result: Dict[str, Any] = {
        "schema": READINESS_SCORE_SCHEMA,
        "naming": "new height-profile readiness screens; not an old G1 pass/fail",
        "episodes": {},
    }
    execution_all = bool(episodes)
    for name in ("stationary_height_hold", "stationary_height_profile"):
        episode = episodes.get(name)
        if not isinstance(episode, dict):
            result["episodes"][name] = {
                "execution_valid": False,
                "reason": "episode records missing",
                "screens": {},
                "readiness_passed": False,
            }
            execution_all = False
            continue
        execution = _execution_valid(episode)
        late = _late_settled(episode)
        screens: Dict[str, Any] = {"late_settled": late}
        if name == "stationary_height_profile":
            lift = _physical_lift(episode)
            clearance = _useful_clearance(episode)
            landing = _loaded_return(episode, lift)
            screens.update({"physical_lift": lift, "useful_clearance": clearance,
                            "loaded_return": landing})
            required = [execution["passed"], lift["passed"], clearance["passed"],
                        landing["passed"], late["passed"]]
        else:
            screens["airborne_required"] = {"passed": True,
                                            "detail": {"note": "hold requires no flight"}}
            required = [execution["passed"], late["passed"]]
        execution_all = execution_all and execution["passed"]
        result["episodes"][name] = {
            "execution_valid": execution["passed"],
            "execution_detail": execution,
            "screens": screens,
            "readiness_passed": all(bool(v) for v in required),
        }
    result["execution_valid"] = execution_all
    result["readiness_passed"] = bool(
        execution_all
        and all(e.get("readiness_passed") for e in result["episodes"].values())
        and len(result["episodes"]) == 2
    )
    return result


def score_digest(score: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(score, sort_keys=True, default=str).encode()).hexdigest()
