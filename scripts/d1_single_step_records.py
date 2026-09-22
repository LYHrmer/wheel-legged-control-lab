"""Count native entries and read full solved contacts without recomputing them."""
from __future__ import annotations

import gzip
import json
from contextlib import AbstractContextManager

import mujoco
import numpy as np

from scripts.d1_jump_readiness_records import NativeStepObserver
from scripts.d1_single_step_geometry import box_contact_feature, collision_identity


class PhysicsCallLedger(AbstractContextManager):
    """Single-threaded root-owned entry guard, including before construction.

    Static forward calls are counted separately. Only the one explicitly bound
    model/data may integrate, and bulk/split stepping is never accepted.
    """

    def __init__(self, *, native_limit=12000):
        self.native_limit = native_limit
        self.allowed = None
        self.attempted = self.returned = self.failed = 0
        self.forward_calls = 0
        self.setconst_calls = 0
        self.forward_by_model_data = {}
        self.advanced_substeps = 0
        self.forbidden_calls = 0
        self.control_attempted = self.control_completed = 0
        self.originals = {}

    def __enter__(self):
        for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_setConst"):
            self.originals[name] = getattr(mujoco, name)
        mujoco.mj_step = self._step
        mujoco.mj_step1 = mujoco.mj_step2 = self._forbidden
        mujoco.mj_forward = self._forward
        mujoco.mj_setConst = self._setconst
        return self

    def __exit__(self, *_):
        for name, entry in self.originals.items():
            setattr(mujoco, name, entry)

    def _forbidden(self, *args, **kwargs):
        self.forbidden_calls += 1
        raise RuntimeError("split or unauthorized integration forbidden")

    def _step(self, model, data, *args, **kwargs):
        if (self.allowed is None or model is not self.allowed[0] or data is not self.allowed[1]
                or args or kwargs or self.attempted >= self.native_limit):
            return self._forbidden()
        self.attempted += 1
        before = float(data.time)
        try:
            result = self.originals["mj_step"](model, data)
        except BaseException:
            self.failed += 1
            raise
        finally:
            self.advanced_substeps += round((float(data.time)-before)/float(model.opt.timestep))
        self.returned += 1
        return result

    def _forward(self, *args, **kwargs):
        self.forward_calls += 1
        pair = str((id(args[0]), id(args[1])))
        self.forward_by_model_data[pair] = self.forward_by_model_data.get(pair, 0)+1
        return self.originals["mj_forward"](*args, **kwargs)

    def _setconst(self, *args, **kwargs):
        self.setconst_calls += 1
        return self.originals["mj_setConst"](*args, **kwargs)

    def receipt(self):
        return {"native_attempted": self.attempted, "native_returned": self.returned,
                "native_failed": self.failed, "native_limit": self.native_limit,
                "clock_advanced_substeps": self.advanced_substeps,
                "mj_forward_calls": self.forward_calls, "forbidden_entries": self.forbidden_calls,
                "mj_setConst_calls": self.setconst_calls,
                "forward_counts_by_runtime_model_data_identity": self.forward_by_model_data,
                "control_attempted": self.control_attempted,
                "control_completed": self.control_completed,
                "static_forward_is_not_dynamic_load_evidence": True}


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite_float": repr(value)}
    return value


class StreamingNativeObserver(NativeStepObserver):
    """Write entry event before delegation and full outcome even on exception.

    A single batch owns the outer ledger. This recorder additionally caps its
    one case at 6000 calls. Failed attempts remain present, without padding.
    """

    def __init__(self, plant, directory, *, baseline=None):
        super().__init__(mujoco, plant.model, plant.data, plant=plant,
                         contact_reader=sample_step_contacts)
        self.directory = directory
        self.stream = self.events = None
        self.baseline = baseline
        self.box_contact_seen = False
        self.prefix_compared_native = 0
        self.prefix_errors = []

    def __enter__(self):
        self.stream = gzip.open(self.directory/"native.jsonl.gz", "xt")
        self.events = gzip.open(self.directory/"native_entry_events.jsonl.gz", "xt")
        return super().__enter__()

    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            self.stream.close()
            self.events.close()

    def _observed(self, model, data, *args, **kwargs):
        if model is not self.model or data is not self.data or args or kwargs or len(self.entries) >= 6000:
            raise RuntimeError("foreign, bulk or over-budget native call")
        if self.baseline is not None and not self.box_contact_seen:
            other = self.baseline[len(self.entries)]
            for field, actual in (("qpos_before", data.qpos), ("qvel_before", data.qvel),
                                  ("ctrl_nm", data.ctrl)):
                if not np.array_equal(actual, other[field]):
                    self.prefix_errors.append({"native_index": len(self.entries), "field": field})
            if self.prefix_errors:
                raise RuntimeError("unexplained pre-contact dynamics divergence; no further integration")
        self.events.write(json.dumps({"attempt": len(self.entries), "time_s": float(data.time)})+"\n")
        self.events.flush()
        before = {"qacc_warmstart_before": data.qacc_warmstart.copy(), "act_before": data.act.copy()}
        try:
            result = super()._observed(model, data)
        finally:
            row = self.entries[-1]
            row.update(before, qacc_warmstart_after=data.qacc_warmstart.copy(), act_after=data.act.copy())
            self.stream.write(json.dumps(jsonable(row), allow_nan=False)+"\n")
            self.stream.flush()
        if self.baseline is not None and not self.box_contact_seen:
            self.box_contact_seen = row["contacts"]["geometric_box_contact"]
            if not self.box_contact_seen:
                other = self.baseline[row["index"]]
                for field in ("qpos_returned", "qvel_returned", "ctrl_nm"):
                    if not np.array_equal(row[field], other[field]):
                        self.prefix_errors.append({"native_index": row["index"], "field": field})
                self.prefix_compared_native += 1
                if self.prefix_errors:
                    raise RuntimeError("unexplained pre-contact dynamics divergence; no further integration")
        return result


def sample_step_contacts(plant):
    """Read every contact from the existing native solver cache, in geom order.

    ``frame[0]`` points geom1 to geom2. Solved forces and geometry describe the
    native solver evaluation; they are not refreshed post-integration contacts.
    """
    model, data = plant.model, plant.data
    rows, loads, box_loads = [], np.zeros(4), np.zeros(4)
    nonwheel = 0
    for index, contact in enumerate(data.contact):
        g1, g2 = int(contact.geom1), int(contact.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        t1, t2 = g1 in plant.terrain_geom_ids, g2 in plant.terrain_geom_ids
        terrain = (g1 if t1 else g2) if t1 != t2 else None
        robot = (g2 if t1 else g1) if terrain is not None else None
        wheel = (plant._wheel_index_by_body_id.get(int(model.geom_bodyid[robot]))
                 if robot is not None else None)
        frame = np.asarray(contact.frame).reshape(3, 3)
        local = np.zeros(6)
        active = int(contact.efc_address) >= 0
        if active:
            mujoco.mj_contactForce(model, data, index, local)
        sign = 1. if t1 else -1.
        normal = sign*frame[0] if terrain is not None else None
        if wheel is not None and active and local[0] > 0.:
            loads[wheel] += local[0]
            if terrain == plant.box_geom_id:
                box_loads[wheel] += local[0]
        if terrain is not None and wheel is None:
            nonwheel += 1  # Any geometric nonwheel terrain contact fails the task.
        feature = None
        if terrain == plant.box_geom_id and terrain is not None:
            feature = box_contact_feature(contact.pos, normal,
                model.geom_pos[terrain], model.geom_size[terrain],
                tolerance_m=float(abs(contact.dist)+max(model.geom_margin[g1],
                    model.geom_margin[g2], contact.includemargin)+1e-7))
        plane_valid = (terrain != plant.floor_geom_id or np.allclose(
            normal, (0., 0., 1.), rtol=0., atol=1e-10))
        rows.append({"index": index, "geom1": g1, "geom2": g2,
            "geom1_identity": collision_identity(model, g1),
            "geom2_identity": collision_identity(model, g2),
            "body1": b1, "body2": b2,
            "body1_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1),
            "body2_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2),
            "frame_geom1_to_geom2": frame.tolist(), "position_world_m": contact.pos.tolist(),
            "distance_m": float(contact.dist), "inclusion_margin_m": float(contact.includemargin),
            "efc_address": int(contact.efc_address), "dimension": int(contact.dim),
            "friction": contact.friction.tolist(), "active": active,
            "local_force_torque": local.tolist(), "normal_load_n": float(local[0]),
            "terrain_geom_id": terrain, "robot_geom_id": robot, "wheel_index": wheel,
            "normal_terrain_to_robot_world": None if normal is None else normal.tolist(),
            "force_on_robot_world_n": None if normal is None else (sign*(frame.T@local[:3])).tolist(),
            "frame_orthonormal": bool(np.allclose(frame@frame.T, np.eye(3), rtol=0., atol=1e-10)
                                      and abs(np.linalg.det(frame)-1.) <= 1e-10),
            "plane_normal_valid": bool(plane_valid), "box_feature": feature})
    return {"sampling": "native_step_solved_cache_not_synchronized_endpoint",
            "contacts": rows, "wheel_positive_normal_load_n": loads.tolist(),
            "wheel_box_positive_normal_load_n": box_loads.tolist(),
            "nonwheel_terrain_contacts": nonwheel,
            "geometric_box_contact": any(r["terrain_geom_id"] == plant.box_geom_id
                                         and plant.box_geom_id >= 0 for r in rows)}


def compiled_geometry_manifest(plant):
    model = plant.model
    rows = []
    for gid in range(model.ngeom):
        body = int(model.geom_bodyid[gid])
        rows.append({"geom_id": gid, "identity": collision_identity(model, gid),
            "body_id": body, "body_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body),
            "geom_type": int(model.geom_type[gid]), "margin_m": float(model.geom_margin[gid]),
            "wheel_index": plant._wheel_index_by_body_id.get(body),
            "collision": bool(model.geom_contype[gid] or model.geom_conaffinity[gid]),
            "terrain_kind": "plane" if gid == plant.floor_geom_id else "box" if gid == plant.box_geom_id else None,
            "position_local_m": model.geom_pos[gid].tolist(), "size_m": model.geom_size[gid].tolist()})
    return {"schema": "compiled-single-step-geom-bindings-v1", "geoms": rows,
            "base_body_ipos_local_m": model.body_ipos[plant.base_body_id].tolist(),
            "body_vx_semantics": "base inertial COM velocity projected into visible base_link frame"}
