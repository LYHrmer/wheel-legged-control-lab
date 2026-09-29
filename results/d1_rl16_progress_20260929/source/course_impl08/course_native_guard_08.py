"""Read-only native contact guard for the fixed 92-primitive D1 course.

The root runtime owns the five-step budget and calls ``before``/``after`` around
its *actual* mj_step. This module never constructs, resets, forwards or steps a
model. A train interval is read once after its fifth native return, before the
environment computes task termination and reward. The built-in exclusive
recorder writes full E/heldout native/contact/force rows and bounded train
numeric blocks; train writes full contacts on task/hard failure. Only sampled
native forces are claimed.
"""

from __future__ import annotations

import gzip
import json
import math
import os
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "d1-course-native-contact-guard-08-v1"
MODES = frozenset(("authority_probe", "baseline_ladder", "train", "heldout"))
FAMILY_PREFIXES = {
    "bump": "terrain_bump_", "rough": "terrain_rough_",
    "ramp": "terrain_ramp_", "stair": "terrain_stair_",
    "jump": "terrain_jump_",
}
EXPECTED_FAMILY_COUNTS = {
    "floor": 1, "rough": 63, "ramp": 3, "stair": 9,
    "bump": 13, "jump": 3,
}


def _finite(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"nonfinite {name}")
    return array


def _pose_degrees(qpos: Any) -> tuple[float, float]:
    values = _finite(qpos, "qpos")
    if values.ndim != 1 or values.size < 7:
        raise ValueError("native qpos lacks free-base quaternion")
    w, x, y, z = map(float, values[3:7])
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if abs(norm - 1.0) > 1e-6:
        raise ValueError("native free-base quaternion is not unit length")
    return (
        math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))),
        math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))),
    )


def _family(name: str) -> str:
    if name == "floor":
        return "floor"
    for family, prefix in FAMILY_PREFIXES.items():
        if name.startswith(prefix):
            return family
    raise ValueError(f"unknown course terrain family: {name}")


def _json_safe(value: Any) -> Any:
    """Preserve a failed nonfinite value visibly without emitting invalid JSON."""
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "+Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


class CourseNativeGuard:
    """One model/data, one mode, actual contact cache and force provenance."""

    def __init__(
        self,
        runtime: Any,
        *,
        mode: str | None = None,
        native_sink: Callable[[dict[str, Any]], None] | None = None,
        failure_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if mode is not None and mode not in MODES:
            raise ValueError(f"unsupported native guard mode: {mode}")
        self.runtime = runtime
        self.mode = mode
        self.native_sink = native_sink
        self.failure_sink = failure_sink
        self.plant: Any | None = None
        self._identity: tuple[int, int] | None = None
        self._terrain: dict[int, tuple[str, str]] = {}
        self._interval: list[dict[str, Any]] = []
        self._last_summary: dict[str, Any] | None = None
        self._last_read = 0
        self._interval_open = False
        self._segment: dict[str, Any] | None = None
        self.segments: list[dict[str, Any]] = []
        self._native_block: list[dict[str, Any]] = []
        self._train_arrays: dict[str, list[np.ndarray]] = {}
        self._block_index = 0
        self._controls_in_block = 0
        self._contact_failure_index = 0
        self.attempted = 0
        self.returned = 0
        self.checked = 0
        self.contacts_checked = 0
        self.nonwheel_contacts = 0
        self.positive_wheel_load_by_family: Counter[str] = Counter()
        self.max_abs_roll_deg = 0.0
        self.max_abs_pitch_deg = 0.0
        self.failure: dict[str, Any] | None = None

    def __enter__(self) -> CourseNativeGuard:  # noqa: PYI034 - Python 3.10 has no Self
        if self.runtime.native_monitor is not None:
            raise RuntimeError("another native monitor is installed")
        self.runtime.native_monitor = self
        return self

    def __exit__(self, *_: object) -> None:
        self.runtime.native_monitor = None

    def bind(self, plant: Any) -> None:
        """Bind one actual compiled course pair before any native call."""
        if self.plant is not None or self.attempted:
            raise RuntimeError("course native guard may bind only once")
        rows = tuple(plant.collision_terrain_metadata["world_collision_geoms"])
        if len(rows) != 92:
            raise ValueError("actual compiled course does not have 92 terrain geoms")
        terrain: dict[int, tuple[str, str]] = {}
        families: Counter[str] = Counter()
        model = plant.model
        for row in rows:
            gid = row["geom_id"]
            name = row["name"]
            if type(gid) is not int or not 0 <= gid < int(model.ngeom) or gid in terrain:
                raise ValueError("duplicate or invalid compiled course geom ID")
            family = _family(name)
            families[family] += 1
            if (int(model.geom_bodyid[gid]) != 0 or row["body_id"] != 0
                    or row["collision"] is not True
                    or row["type"] != ("plane" if family == "floor" else "box")):
                raise ValueError("course terrain binding differs from actual compiled geom")
            for actual, saved, label in (
                (model.geom_pos[gid], row["position_m"], "position"),
                (model.geom_size[gid], row["size_m"], "size"),
                (model.geom_quat[gid], row["quaternion_wxyz"], "quaternion"),
            ):
                if not np.array_equal(np.asarray(actual), np.asarray(saved)):
                    raise ValueError(f"compiled course geom {label} changed since binding")
            terrain[gid] = (name, family)
        if dict(families) != EXPECTED_FAMILY_COUNTS:
            raise ValueError("compiled course terrain families differ from frozen 92 layout")
        if set(terrain) != set(plant.terrain_geom_ids):
            raise ValueError("course terrain IDs differ from plant contact binding")
        if int(plant.physics_steps) != 5 or float(model.opt.timestep) != 0.002:
            raise ValueError("course native cadence must be five times 0.002 seconds")
        self._identity = (int(model._address), int(plant.data._address))
        self._terrain = terrain
        self.plant = plant

    def start_segment(
        self,
        plant: Any,
        folder: Any,
        name: str,
        control_limit: int,
        mode: str,
        *,
        native_sink: Callable[[dict[str, Any]], None] | None = None,
        failure_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Reserve a caller-owned phase; the root runtime still enforces C caps."""
        if self._segment is not None or self._interval_open or self._interval:
            raise RuntimeError("previous course segment is not closed")
        if self.failure is not None:
            raise RuntimeError("fatal native guard failure forbids a new segment")
        if mode not in MODES or type(control_limit) is not int or control_limit < 1:
            raise ValueError("invalid course segment mode or control limit")
        if not Path(folder).is_dir():
            raise ValueError("root must create a unique course segment folder first")
        if self.plant is None:
            self.bind(plant)
        elif plant is not self.plant:
            raise RuntimeError("course segment changed the actual model/data pair")
        selected_sink = native_sink if native_sink is not None else self.native_sink
        self.mode = mode
        self.native_sink = selected_sink
        if failure_sink is not None:
            self.failure_sink = failure_sink
        self._segment = {
            "name": name, "folder": str(folder), "mode": mode,
            "control_limit": control_limit, "native_limit": 5 * control_limit,
            "attempted_start": self.attempted, "returned_start": self.returned,
            "native_files": [], "train_array_files": [],
            "task_contact_failure_files": [],
            "model_address": self._identity[0], "data_address": self._identity[1],
        }
        self._native_block = []
        self._train_arrays = {}
        self._block_index = 0
        self._controls_in_block = 0
        self._contact_failure_index = 0

    def _write_gzip_rows(self, path: Path, rows: list[dict[str, Any]]) -> None:
        """One exclusive file per bounded block; no append or per-native fsync."""
        with path.open("xb") as raw, gzip.GzipFile(
            fileobj=raw, mode="wb", mtime=0
        ) as compressed:
            for row in rows:
                compressed.write(
                    (json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                    .encode("utf-8")
                )
        # An exclusive block is durable before its filename enters a receipt.
        with path.open("rb") as persisted:
            os.fsync(persisted.fileno())

    def _flush_block(self) -> None:
        if not self._native_block:
            return
        if self._segment is None:
            raise RuntimeError("cannot flush native block without a segment")
        path = Path(self._segment["folder"]) / f"native_block_{self._block_index:04d}.jsonl.gz"
        self._write_gzip_rows(path, self._native_block)
        self._segment["native_files"].append(path.name)
        if self._segment["mode"] == "train":
            array_path = (Path(self._segment["folder"])
                          / f"native_arrays_{self._block_index:04d}.npz")
            with array_path.open("xb") as stream:
                np.savez(stream, **{
                    key: np.stack(values) for key, values in self._train_arrays.items()
                })
                stream.flush()
                os.fsync(stream.fileno())
            self._segment["train_array_files"].append(array_path.name)
            self._train_arrays = {}
        self._native_block = []
        self._block_index += 1
        self._controls_in_block = 0

    def begin_interval(self, plant: Any, *, start_time_s: float) -> None:
        """Env calls once after prepare and immediately before its physical step."""
        if plant is not self.plant or self.runtime.active_plant is not plant:
            raise RuntimeError("course interval start changed the bound actual plant")
        if self._segment is None or self._interval_open or self._interval:
            raise RuntimeError("course interval is unbound or prior interval is unread")
        start = float(start_time_s)
        if not math.isfinite(start) or abs(start - float(plant.data.time)) > 1e-12:
            raise RuntimeError("prepared course time differs from actual integrator clock")
        completed = self.returned - self._segment["returned_start"]
        if completed >= self._segment["native_limit"]:
            raise RuntimeError("course segment native reservation exhausted")
        self._interval_open = True

    def _target(self, model: Any, data: Any) -> None:
        plant = self.plant
        if (plant is None or model is not plant.model or data is not plant.data
                or self.runtime.active_plant is not plant
                or (int(model._address), int(data._address)) != self._identity):
            raise RuntimeError("native call did not use bound actual course model/data")

    def before(self, model: Any, data: Any) -> dict[str, Any]:
        self._target(model, data)
        if not self._interval_open or self._segment is None:
            raise RuntimeError("root must open one course control interval before native calls")
        if len(self._interval) >= 5:
            raise RuntimeError("previous five-native interval was not consumed")
        if self.attempted - self._segment["attempted_start"] >= self._segment["native_limit"]:
            raise RuntimeError("course segment attempted-native reservation exhausted")
        for key in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
            _finite(getattr(data, key), f"pre-native {key}")
        start = float(data.time)
        if not math.isfinite(start):
            raise ValueError("nonfinite pre-native clock")
        if self._interval and abs(start - self._interval[-1]["end_time_s"]) > 1e-12:
            raise ValueError("noncontiguous native interval clock")
        self.attempted += 1
        token: dict[str, Any] = {"index": self.attempted - 1, "start_time_s": start}
        if self.mode != "train":
            token["before"] = {
                key: np.asarray(getattr(data, key)).copy().tolist()
                for key in ("qpos", "qvel", "ctrl", "qacc_warmstart")
            }
        else:
            token["train_before"] = {
                key: np.asarray(getattr(data, key)).copy()
                for key in ("qpos", "qvel", "ctrl", "qacc_warmstart")
            }
        return token

    def _contact(self, model: Any, data: Any, index: int, contact: Any) -> dict[str, Any]:
        from scripts.d1_single_step_geometry import box_contact_feature

        g1, g2 = int(contact.geom1), int(contact.geom2)
        if not (0 <= g1 < int(model.ngeom) and 0 <= g2 < int(model.ngeom)):
            raise ValueError("contact geom ID outside actual compiled model")
        frame = _finite(contact.frame, "contact frame").reshape(3, 3)
        if (np.max(np.abs(frame @ frame.T - np.eye(3))) > 1e-10
                or abs(float(np.linalg.det(frame)) - 1.0) > 1e-10):
            raise ValueError("native contact frame is not right-handed orthonormal")
        for key in ("pos", "friction", "solref", "solreffriction", "solimp"):
            _finite(getattr(contact, key), f"contact {key}")
        for key in ("dist", "includemargin"):
            if not math.isfinite(float(getattr(contact, key))):
                raise ValueError(f"nonfinite contact {key}")
        for gid in (g1, g2):
            _finite(data.geom_xpos[gid], "actual geom_xpos cache")
            rot = _finite(data.geom_xmat[gid], "actual geom_xmat cache").reshape(3, 3)
            if (np.max(np.abs(rot @ rot.T - np.eye(3))) > 1e-10
                    or abs(float(np.linalg.det(rot)) - 1.0) > 1e-10):
                raise ValueError("invalid actual geom orientation cache")
        t1, t2 = g1 in self._terrain, g2 in self._terrain
        row: dict[str, Any] = {
            "contact_index": index, "geom1": g1, "geom2": g2,
            "efc_address": int(contact.efc_address),
            "terrain_family": None, "robot_wheel_index": None,
            "nonwheel_ground_contact": False,
            "nonwheel_ground_candidate": False,
            "nonwheel_active_solver_contact": False,
            "positive_wheel_load": False,
        }
        if self.mode != "train":
            row.update({
                "dim": int(contact.dim),
                "position_world_m": np.asarray(contact.pos).tolist(),
                "frame_world": frame.tolist(),
                "distance_m": float(contact.dist),
                "includemargin_m": float(contact.includemargin),
                "friction": np.asarray(contact.friction).tolist(),
                "solref": np.asarray(contact.solref).tolist(),
                "solreffriction": np.asarray(contact.solreffriction).tolist(),
                "solimp": np.asarray(contact.solimp).tolist(),
                "geom1_xpos": np.asarray(data.geom_xpos[g1]).tolist(),
                "geom2_xpos": np.asarray(data.geom_xpos[g2]).tolist(),
                "geom1_xmat": np.asarray(data.geom_xmat[g1]).reshape(3, 3).tolist(),
                "geom2_xmat": np.asarray(data.geom_xmat[g2]).reshape(3, 3).tolist(),
                "geom1_compiled_size": np.asarray(model.geom_size[g1]).tolist(),
                "geom2_compiled_size": np.asarray(model.geom_size[g2]).tolist(),
                "geom1_compiled_margin": float(model.geom_margin[g1]),
                "geom2_compiled_margin": float(model.geom_margin[g2]),
            })
        wrench = None
        if self.mode != "train" and int(contact.efc_address) >= 0:
            import mujoco

            wrench = np.empty(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, index, wrench)
            _finite(wrench, "actual mj_contactForce wrench")
            row["contact_force_local"] = wrench.tolist()
            row["force_source"] = "mj_contactForce_actual_native_cache"
        if t1 and t2:
            raise ValueError("invalid terrain-to-terrain solver contact")
        if t1 == t2:
            return row
        terrain = g1 if t1 else g2
        robot = g2 if t1 else g1
        name, family = self._terrain[terrain]
        body = int(model.geom_bodyid[robot])
        wheel = self.plant._wheel_index_by_body_id.get(body)
        normal_world = frame[0] if t1 else -frame[0]
        tolerance = (abs(float(contact.dist))
                     + max(float(model.geom_margin[g1]),
                           float(model.geom_margin[g2]), float(contact.includemargin))
                     + 1e-7)
        if family == "floor":
            if np.max(np.abs(normal_world - np.asarray((0.0, 0.0, 1.0)))) > 1e-10:
                raise ValueError("invalid actual plane contact normal")
            if abs(float(contact.pos[2])) > tolerance:
                raise ValueError("actual plane contact point outside compiled plane")
            feature = "plane"
        else:
            center = _finite(data.geom_xpos[terrain], "box center cache")
            rotation = _finite(data.geom_xmat[terrain], "box rotation cache").reshape(3, 3)
            local_position = rotation.T @ (_finite(contact.pos, "contact position") - center)
            local_normal = rotation.T @ normal_world
            support = box_contact_feature(
                local_position, local_normal, (0.0, 0.0, 0.0),
                model.geom_size[terrain], tolerance_m=tolerance,
            )
            if not support["geometric_support_valid"]:
                raise ValueError("invalid rotated-box local normal cone/contact feature")
            feature = support["feature"]
        row.update({
            "terrain_geom_id": terrain, "terrain_geom_name": name,
            "terrain_family": family, "robot_geom_id": robot,
            "robot_body_id": body, "robot_wheel_index": wheel,
            # The frozen E/T gate counts every native nonwheel terrain
            # candidate, including an inactive margin contact. The separate
            # active flag does not reinterpret a candidate as positive load.
            "nonwheel_ground_contact": wheel is None,
            "nonwheel_ground_candidate": wheel is None,
            "nonwheel_active_solver_contact": wheel is None and int(contact.efc_address) >= 0,
        })
        if self.mode != "train":
            row.update({
                "terrain_to_robot_normal_world": normal_world.tolist(),
                "geometric_feature": feature, "geometry_support_valid": True,
            })
        # Native forces are sampled only in qualification modes. The solver
        # cache and force are both from this actual returned native call.
        if wrench is not None:
            force_world = frame.T @ wrench[:3]
            if robot == g1:
                force_world = -force_world
            normal_load = float(force_world @ normal_world)
            row.update({
                "force_on_robot_world_n": force_world.tolist(),
                "normal_load_on_robot_n": normal_load,
                "positive_wheel_load": bool(wheel is not None and normal_load > 0.0),
            })
        return row

    def after(self, model: Any, data: Any, token: dict[str, Any]) -> None:
        # The runtime reaches this hook only after the actual mj_step returns.
        # Preserve that fact even if validation or record I/O subsequently fails.
        self.returned += 1
        try:
            self._target(model, data)
            if token.get("index") != self.attempted - 1:
                raise RuntimeError("native after token differs from last attempted call")
            for key in ("qpos", "qvel", "ctrl", "qacc", "qacc_warmstart"):
                _finite(getattr(data, key), f"post-native {key}")
            end = float(data.time)
            if not math.isfinite(end) or abs(end - token["start_time_s"] - 0.002) > 1e-12:
                raise ValueError("actual native clock did not advance 0.002 seconds")
            roll, pitch = _pose_degrees(data.qpos)
            self.max_abs_roll_deg = max(self.max_abs_roll_deg, abs(roll))
            self.max_abs_pitch_deg = max(self.max_abs_pitch_deg, abs(pitch))
            if max(abs(roll), abs(pitch)) > 45.0:
                raise ValueError("actual post-native posture exceeds hard 45 degree bound")
            contacts = [
                self._contact(model, data, index, contact)
                for index, contact in enumerate(data.contact)
            ]
            nonwheel = sum(int(row["nonwheel_ground_contact"]) for row in contacts)
            loads = Counter(
                row["terrain_family"] for row in contacts if row["positive_wheel_load"]
            )
            self.contacts_checked += len(contacts)
            self.nonwheel_contacts += nonwheel
            self.positive_wheel_load_by_family.update(loads)
            row = {
                "native_index": token["index"],
                "start_time_s": token["start_time_s"], "end_time_s": end,
                "roll_deg": roll, "pitch_deg": pitch,
                "contact_count": len(contacts),
                "nonwheel_contact_count": nonwheel,
                "terrain_family_positive_wheel_load": dict(loads),
            }
            if self.mode != "train":
                row["before"] = token["before"]
                row["after"] = {
                    key: np.asarray(getattr(data, key)).copy().tolist()
                    for key in ("qpos", "qvel", "ctrl", "qacc", "qacc_warmstart")
                }
                row["after"]["actuator_force"] = np.asarray(data.actuator_force).copy().tolist()
                row["after"]["qfrc_actuator"] = np.asarray(data.qfrc_actuator).copy().tolist()
                row["contacts"] = contacts
            elif nonwheel:
                self._record_task_contact_failure({
                    **row, "contacts": self._raw_contacts(data),
                    "reason": "train_task_terminal_nonwheel",
                })
            if self.mode == "train":
                arrays = {
                    **{f"{key}_before": value for key, value in token["train_before"].items()},
                    **{f"{key}_after": np.asarray(getattr(data, key)).copy()
                       for key in ("qpos", "qvel", "ctrl", "qacc_warmstart")},
                    "qacc_after": np.asarray(data.qacc).copy(),
                    "actuator_force_after": np.asarray(data.actuator_force).copy(),
                    "qfrc_actuator_after": np.asarray(data.qfrc_actuator).copy(),
                    "start_time_s": np.asarray(token["start_time_s"]),
                    "end_time_s": np.asarray(end),
                }
                for key, value in arrays.items():
                    self._train_arrays.setdefault(key, []).append(value)
            self._native_block.append(row)
            if self.native_sink is not None:
                self.native_sink(row)
            self._interval.append(row)
            self.checked += 1
            if self.mode in ("authority_probe", "baseline_ladder"):
                if max(abs(roll), abs(pitch)) > 10.0:
                    raise ValueError("E qualification posture exceeds 10 degrees")
                if nonwheel:
                    raise ValueError("E qualification nonwheel terrain contact")
        except BaseException as error:
            self.failure = {
                "type": type(error).__name__, "message": str(error),
                "native_index": token.get("index"),
                "time_s": _json_safe(float(data.time)), "mode": self.mode,
            }
            if self._segment is not None:
                try:
                    post = {
                        key: _json_safe(np.asarray(getattr(data, key)))
                        for key in ("qpos", "qvel", "ctrl", "qacc", "qacc_warmstart",
                                    "actuator_force", "qfrc_actuator")
                    }
                    evidence = {
                        "reason": "native_guard_hard_failure", "failure": self.failure,
                        "model_address": int(model._address),
                        "data_address": int(data._address),
                        "before": _json_safe(token.get("before", token.get("train_before"))),
                        "after": post,
                        "start_time_s": _json_safe(token.get("start_time_s")),
                        "end_time_s": _json_safe(float(data.time)),
                        "raw_contacts": _json_safe(self._raw_contacts(data)),
                    }
                except Exception as snapshot_error:  # noqa: BLE001 - preserve physics error
                    evidence = {
                        "reason": "native_guard_hard_failure", "failure": self.failure,
                        "evidence_snapshot_error": repr(snapshot_error),
                    }
                try:
                    self._record_task_contact_failure(evidence)
                except Exception as save_error:  # noqa: BLE001 - preserve physics error
                    self.failure["evidence_save_error"] = repr(save_error)
            raise

    def _raw_contacts(self, data: Any) -> list[dict[str, Any]]:
        """Copy full actual cache only on a train task/hard failure."""
        return [
            {
                "geom1": int(c.geom1), "geom2": int(c.geom2),
                "dim": int(c.dim), "efc_address": int(c.efc_address),
                "pos": np.asarray(c.pos).tolist(),
                "frame": np.asarray(c.frame).tolist(),
                "dist": float(c.dist), "includemargin": float(c.includemargin),
                "friction": np.asarray(c.friction).tolist(),
                "solref": np.asarray(c.solref).tolist(),
                "solreffriction": np.asarray(c.solreffriction).tolist(),
                "solimp": np.asarray(c.solimp).tolist(),
                "geom1_xpos": np.asarray(data.geom_xpos[int(c.geom1)]).tolist(),
                "geom2_xpos": np.asarray(data.geom_xpos[int(c.geom2)]).tolist(),
                "geom1_xmat": np.asarray(data.geom_xmat[int(c.geom1)]).reshape(3, 3).tolist(),
                "geom2_xmat": np.asarray(data.geom_xmat[int(c.geom2)]).reshape(3, 3).tolist(),
            }
            for c in data.contact
        ]

    def _record_task_contact_failure(self, row: dict[str, Any]) -> None:
        if self._segment is None:
            raise RuntimeError("contact failure has no course segment")
        path = (Path(self._segment["folder"])
                / f"native_contact_failure_{self._contact_failure_index:04d}.json")
        with path.open("x", encoding="utf-8") as stream:
            json.dump(_json_safe(row), stream, allow_nan=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        self._segment["task_contact_failure_files"].append(path.name)
        self._contact_failure_index += 1
        if self.failure_sink is not None:
            self.failure_sink(row)

    def interval_summary(self, plant: Any, *, end_time_s: float) -> dict[str, Any]:
        """Consume exactly the just-returned five-native interval; no engine call."""
        if plant is not self.plant or len(self._interval) != 5:
            raise RuntimeError("interval reader needs five native returns on bound plant")
        if self.returned == self._last_read or self.returned - self._last_read != 5:
            raise RuntimeError("native interval was already read or is not five returns")
        expected_end = float(end_time_s)
        if not math.isfinite(expected_end) or abs(self._interval[-1]["end_time_s"] - expected_end) > 1e-12:
            raise RuntimeError("native interval end time differs from actual plant")
        if abs(float(plant.data.time) - expected_end) > 1e-12:
            raise RuntimeError("plant data clock differs from native interval")
        loads: Counter[str] = Counter()
        for row in self._interval:
            loads.update(row["terrain_family_positive_wheel_load"])
        summary = {
            "schema": SCHEMA, "mode": self.mode,
            "start_time_s": self._interval[0]["start_time_s"],
            "end_time_s": expected_end, "native_returns": 5,
            "nonwheel_contact_count": sum(row["nonwheel_contact_count"] for row in self._interval),
            "max_abs_roll_deg": max(abs(row["roll_deg"]) for row in self._interval),
            "max_abs_pitch_deg": max(abs(row["pitch_deg"]) for row in self._interval),
            "terrain_family_positive_wheel_load": dict(loads),
        }
        if abs(summary["end_time_s"] - summary["start_time_s"] - 0.01) > 1e-12:
            raise RuntimeError("native interval did not span exactly five substeps")
        self._last_summary = summary
        self._last_read = self.returned
        self._interval.clear()
        self._interval_open = False
        self._controls_in_block += 1
        if self._controls_in_block == 1024:
            self._flush_block()
        return dict(summary)

    def finish_segment(self) -> dict[str, Any]:
        """Close and report an actual phase, including any partial failed interval."""
        if self._segment is None:
            raise RuntimeError("no course segment to close")
        self._flush_block()
        segment = self._segment
        row = {
            "name": segment["name"], "folder": segment["folder"],
            "mode": segment["mode"], "control_limit": segment["control_limit"],
            "native_limit": segment["native_limit"],
            "native_attempted": self.attempted - segment["attempted_start"],
            "native_returned": self.returned - segment["returned_start"],
            "partial_native_interval": bool(self._interval_open or self._interval),
            "failure": self.failure,
            "native_files": list(segment["native_files"]),
            "train_array_files": list(segment["train_array_files"]),
            "task_contact_failure_files": list(segment["task_contact_failure_files"]),
            "force_sampling_performed": segment["mode"] != "train",
            "full_contact_qualification_recorded": segment["mode"] != "train",
        }
        if row["native_attempted"] > row["native_limit"]:
            raise RuntimeError("course native segment exceeded its reservation")
        self.segments.append(row)
        self._segment = None
        self._interval = []
        self._interval_open = False
        return dict(row)

    def report(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA, "mode": self.mode,
            "record_mode": "compact_train" if self.mode == "train" else "full_qualification",
            "native_attempted": self.attempted, "native_returned": self.returned,
            "native_checked": self.checked,
            "contacts_checked": self.contacts_checked,
            "nonwheel_ground_contacts": self.nonwheel_contacts,
            "positive_wheel_load_by_family": dict(self.positive_wheel_load_by_family),
            "max_abs_roll_deg": self.max_abs_roll_deg,
            "max_abs_pitch_deg": self.max_abs_pitch_deg,
            "force_sampling_performed": any(
                segment["mode"] != "train" for segment in self.segments
            ),
            "full_contact_qualification_claimed": bool(self.segments) and all(
                segment["mode"] != "train" for segment in self.segments
            ),
            "failure": self.failure,
            "segments": list(self.segments),
        }
