"""Compact solver-cache geometry audit after each real normal mj_step.

The digest and compact checks do not replace held-out full contact records.
No forward, collision recomputation, model allocation or integration is added.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.d1_single_step_geometry import box_contact_feature
from scripts.d1_single_step_records import compiled_geometry_manifest, jsonable


class NativeGeometryGuard:
    def __init__(self, runtime, output):
        self.runtime = runtime
        self.output = Path(output)
        self.checked = self.passed = self.contact_count = self.box_contact_count = 0
        self.digest = hashlib.sha256()
        self.bindings = {}
        self.segments = {}
        self.failure = None

    def __enter__(self):
        if self.runtime.native_monitor is not None:
            raise RuntimeError("native geometry guard already installed")
        self.runtime.native_monitor = self
        return self

    def __exit__(self, *_):
        self.runtime.native_monitor = None
        self._save("native_monitor_receipt.json", self.report())

    def _save(self, name, value):
        with (self.output / name).open("x") as stream:
            json.dump(jsonable(value), stream, indent=2, allow_nan=False)
            stream.write("\n")

    def before(self, model, data):
        before = {"qpos": data.qpos.copy(), "qvel": data.qvel.copy(),
                  "ctrl": data.ctrl.copy(), "time_s": float(data.time)}
        try:
            plant = self.runtime.active_plant
            if plant is None or model is not plant.model or data is not plant.data:
                raise RuntimeError("native monitor observed an unbound model/data")
            if any(not np.isfinite(value).all() for value in before.values()):
                raise ValueError("nonfinite state before a native call")
            key = int(model._address)
            if key not in self.bindings:
                plant.validate_single_step()
                manifest = compiled_geometry_manifest(plant)
                ids = [row["geom_id"] for row in manifest["geoms"]]
                if ids != list(range(model.ngeom)):
                    raise RuntimeError("compiled geometry binding is incomplete")
                self.bindings[key] = manifest
            return before
        except BaseException as exc:
            self._record_failure(exc, data, before, [], phase="before native delegation")
            raise

    @staticmethod
    def _contact_row(contact):
        return {"geom1": int(contact.geom1), "geom2": int(contact.geom2),
                "dim": int(contact.dim), "efc_address": int(contact.efc_address),
                "pos": contact.pos.copy(), "frame": contact.frame.copy(),
                "dist": float(contact.dist), "includemargin": float(contact.includemargin),
                "friction": contact.friction.copy(), "solref": contact.solref.copy(),
                "solreffriction": contact.solreffriction.copy(), "solimp": contact.solimp.copy()}

    def after(self, model, data, before):
        self.checked += 1
        label = self.runtime._segment["name"]
        segment = self.segments.setdefault(label, {"checked": 0, "passed": 0})
        segment["checked"] += 1
        plant = self.runtime.active_plant
        raw = []
        try:
            for key in ("qpos", "qvel", "ctrl"):
                if not np.isfinite(before[key]).all() or not np.isfinite(getattr(data, key)).all():
                    raise ValueError("nonfinite integrator state: " + key)
                self.digest.update(before[key].tobytes())
                self.digest.update(getattr(data, key).tobytes())
            if not np.isfinite(data.qacc).all() or not np.isfinite(data.qacc_warmstart).all():
                raise ValueError("nonfinite acceleration")
            if not np.isfinite(data.time) or abs(float(data.time) - before["time_s"] - .002) > 1e-12:
                raise ValueError("normal native clock increment differs from .002")
            self.digest.update(np.asarray([self.checked, data.ncon], dtype=np.int64).tobytes())
            for contact in data.contact:
                row = self._contact_row(contact)
                raw.append(row)
                g1, g2 = row["geom1"], row["geom2"]
                if not (0 <= g1 < model.ngeom and 0 <= g2 < model.ngeom):
                    raise ValueError("contact geom ID is outside the compiled binding")
                values = np.concatenate([np.asarray(value).ravel() for value in row.values()])
                if not np.isfinite(values).all():
                    raise ValueError("nonfinite contact field")
                if (not np.isfinite(data.geom_xpos[[g1, g2]]).all()
                        or not np.isfinite(data.geom_xmat[[g1, g2]]).all()):
                    raise ValueError("nonfinite native solver-cache geom transform")
                frame = row["frame"].reshape(3, 3)
                if (not np.allclose(frame @ frame.T, np.eye(3), rtol=0., atol=1e-10)
                        or abs(np.linalg.det(frame) - 1.) > 1e-10):
                    raise ValueError("contact frame is not orthonormal")
                t1, t2 = g1 in plant.terrain_geom_ids, g2 in plant.terrain_geom_ids
                if t1 != t2:
                    terrain = g1 if t1 else g2
                    normal = frame[0] if t1 else -frame[0]
                    if terrain == plant.floor_geom_id:
                        if not np.allclose(normal, (0., 0., 1.), rtol=0., atol=1e-10):
                            raise ValueError("invalid native plane normal")
                    elif terrain == plant.box_geom_id:
                        feature = box_contact_feature(
                            row["pos"], normal, model.geom_pos[terrain], model.geom_size[terrain],
                            tolerance_m=float(abs(row["dist"]) + max(model.geom_margin[g1],
                                model.geom_margin[g2], row["includemargin"]) + 1e-7))
                        if not feature["geometric_support_valid"]:
                            raise ValueError("invalid native box normal cone or contact feature")
                        self.box_contact_count += 1
                    else:
                        raise ValueError("unknown terrain contact binding")
                self.digest.update(values.tobytes())
                self.contact_count += 1
            self.passed += 1
            segment["passed"] += 1
        except BaseException as exc:
            self._record_failure(exc, data, before, raw,
                phase="native solver contact/geom cache, returned qpos is post-integration")
            raise

    def _record_failure(self, exc, data, before, raw, *, phase):
        self.failure = {"type": type(exc).__name__, "message": str(exc),
                        "checked_native_returns": self.checked,
                        "segment": self.runtime._segment["name"], "phase": phase}
        self._save("native_geometry_failure.json", {
            **self.failure, "before": before,
            "returned_qpos": data.qpos.copy(), "returned_qvel": data.qvel.copy(),
            "returned_time_s": float(data.time),
            "native_solver_geom_xpos": data.geom_xpos.copy(),
            "native_solver_geom_xmat": data.geom_xmat.copy(),
            "all_emitted_contacts": [self._contact_row(c) for c in data.contact],
            "contacts_examined_before_failure": raw,
            "nonfinite_encoding": "recursive explicit nonfinite_float tags; no replacement with finite values"})

    def report(self):
        return {"checked_native_returns": self.checked, "passed_native_returns": self.passed,
                "contacts_checked": self.contact_count, "box_contacts_checked": self.box_contact_count,
                "sha256_rolling_digest": self.digest.hexdigest(), "failure": self.failure,
                "segments": self.segments, "compiled_bindings": self.bindings,
                "no_added_engine_step_forward_collision": True,
                "scope": "all emitted normal-step contacts, including inactive candidates; compact audit, not a full raw force archive or internal CCD-candidate proof"}
