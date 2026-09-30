"""Single-pass compact native audit; no force sampling or per-step file writes."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


class NativeBuffer:
    """Prepaid arrays retain attempted entries, including an unreturned call."""

    def __init__(self, capacity: int, nq: int, nv: int, nu: int):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("positive native capacity required")
        self.capacity = capacity
        self.attempted = self.returned = 0
        widths = {
            "qpos_before": nq,
            "qpos_returned": nq,
            "qvel_before": nv,
            "qvel_returned": nv,
            "qacc_warmstart_before": nv,
            "qacc_warmstart_after": nv,
            "ctrl_nm": nu,
        }
        self.arrays = {
            key: np.zeros((capacity, size), dtype=np.float64)
            for key, size in widths.items()
        }
        self.arrays.update(
            start_time_s=np.zeros(capacity),
            end_time_s=np.zeros(capacity),
            returned=np.zeros(capacity, dtype=np.bool_),
        )

    def entry(self, data: Any) -> int:
        if self.attempted >= self.capacity or self.attempted != self.returned:
            raise RuntimeError(
                "native compact capacity exhausted or prior call unreturned"
            )
        index = self.attempted
        self.attempted += 1
        for field, source in (
            ("qpos_before", "qpos"),
            ("qvel_before", "qvel"),
            ("qacc_warmstart_before", "qacc_warmstart"),
            ("ctrl_nm", "ctrl"),
        ):
            self.arrays[field][index] = getattr(data, source)
        self.arrays["start_time_s"][index] = data.time
        return index

    def exit(self, data: Any, index: int) -> None:
        if index != self.returned or self.attempted != index + 1:
            raise RuntimeError("native return does not match reserved entry")
        for field, source in (
            ("qpos_returned", "qpos"),
            ("qvel_returned", "qvel"),
            ("qacc_warmstart_after", "qacc_warmstart"),
        ):
            self.arrays[field][index] = getattr(data, source)
        self.arrays["end_time_s"][index] = data.time
        self.arrays["returned"][index] = True
        self.returned += 1

    def save(self, path: Path) -> None:
        with path.open("xb") as stream:
            np.savez_compressed(
                stream,
                **{key: value[: self.attempted] for key, value in self.arrays.items()},
            )


def posture_degrees(qpos: Any) -> tuple[float, float]:
    w, x, y, z = map(float, qpos[3:7])
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("invalid native quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return (
        math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))),
        math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))),
    )


def audit_native_contact(plant: Any, contact: Any) -> bool:
    """One true cache contact; return whether it belongs to the box."""
    model, data = plant.model, plant.data
    g1, g2 = int(contact.geom1), int(contact.geom2)
    if not (0 <= g1 < model.ngeom and 0 <= g2 < model.ngeom):
        raise ValueError("contact geom ID outside compiled model")
    frame = np.asarray(contact.frame).reshape(3, 3)
    fields = (
        frame,
        contact.pos,
        contact.friction,
        contact.solref,
        contact.solreffriction,
        contact.solimp,
        (contact.dist, contact.includemargin),
        data.geom_xpos[[g1, g2]],
        data.geom_xmat[[g1, g2]],
    )
    if any(not np.isfinite(value).all() for value in fields):
        raise ValueError("nonfinite native contact/cache field")
    if (
        np.max(np.abs(frame @ frame.T - np.eye(3))) > 1e-10
        or abs(float(np.linalg.det(frame)) - 1.0) > 1e-10
    ):
        raise ValueError("native contact frame not right-handed orthonormal")
    t1, t2 = g1 in plant.terrain_geom_ids, g2 in plant.terrain_geom_ids
    if t1 == t2:
        return False
    terrain, robot = (g1, g2) if t1 else (g2, g1)
    if plant._wheel_index_by_body_id.get(int(model.geom_bodyid[robot])) is None:
        raise ValueError("nonwheel geometric terrain contact")
    normal = frame[0] if t1 else -frame[0]
    if terrain == plant.floor_geom_id:
        if np.max(np.abs(normal - np.asarray((0.0, 0.0, 1.0)))) > 1e-10:
            raise ValueError("invalid native plane normal")
        return False
    if terrain != plant.box_geom_id:
        raise ValueError("unknown terrain contact binding")
    from scripts.d1_single_step_geometry import box_contact_feature

    feature = box_contact_feature(
        contact.pos,
        normal,
        model.geom_pos[terrain],
        model.geom_size[terrain],
        tolerance_m=float(
            abs(contact.dist)
            + max(model.geom_margin[g1], model.geom_margin[g2], contact.includemargin)
            + 1e-7
        ),
    )
    if not feature["geometric_support_valid"]:
        raise ValueError("invalid box normal cone/contact feature")
    return True


class CompactNativeGuard:
    """Keep C/Python native accounting; inspect every solved cache only once."""

    def __init__(self, runtime: Any, output: Path):
        self.runtime, self.output = runtime, output
        self.buffer = None
        self.folder = None
        self.label = None
        self.checked = self.contacts = self.box_contacts = 0
        self.max_abs_rp_deg = 0.0
        self.failure = None
        self.segments: list[dict] = []
        self._bound_identity = None

    def __enter__(self):
        if self.runtime.native_monitor is not None:
            raise RuntimeError("another native monitor is installed")
        self.runtime.native_monitor = self
        return self

    def __exit__(self, *_):
        self.runtime.native_monitor = None

    def start_segment(self, plant: Any, folder: Path, label: str, controls: int):
        if self.buffer is not None:
            raise RuntimeError("previous compact segment was not closed")
        identity = (int(plant.model._address), int(plant.data._address))
        if self._bound_identity is None:
            plant.validate_single_step()
            self._bound_identity = identity
        if identity != self._bound_identity:
            raise RuntimeError("compact run changed model/data")
        self.folder, self.label = folder, label
        self.buffer = NativeBuffer(
            5 * controls, plant.model.nq, plant.model.nv, plant.model.nu
        )

    def before(self, model: Any, data: Any) -> int:
        plant = self.runtime.active_plant
        if (
            self.buffer is None
            or plant is None
            or model is not plant.model
            or data is not plant.data
        ):
            raise RuntimeError("unbound compact native target")
        if any(
            not np.isfinite(getattr(data, key)).all()
            for key in ("qpos", "qvel", "ctrl", "qacc_warmstart")
        ) or not math.isfinite(float(data.time)):
            raise ValueError("nonfinite integrator before native call")
        return self.buffer.entry(data)

    def after(self, model: Any, data: Any, index: int):
        self.buffer.exit(data, index)
        try:
            if any(
                not np.isfinite(getattr(data, key)).all()
                for key in ("qpos", "qvel", "ctrl", "qacc", "qacc_warmstart")
            ):
                raise ValueError("nonfinite integrator after native call")
            if (
                not math.isfinite(float(data.time))
                or abs(
                    float(data.time) - self.buffer.arrays["start_time_s"][index] - 0.002
                )
                > 1e-12
            ):
                raise ValueError("native clock does not advance .002 seconds")
            roll, pitch = posture_degrees(data.qpos)
            self.max_abs_rp_deg = max(self.max_abs_rp_deg, abs(roll), abs(pitch))
            if max(abs(roll), abs(pitch)) > 10.0:
                raise ValueError("native posture exceeds 10 degree profile gate")
            plant = self.runtime.active_plant
            for contact in data.contact:
                self.box_contacts += int(audit_native_contact(plant, contact))
                self.contacts += 1
            self.checked += 1
        except BaseException as error:
            self.failure = {
                "type": type(error).__name__,
                "message": str(error),
                "segment": self.label,
                "native_index": index,
                "time_s": float(data.time),
            }
            # Raw failed cache dump only on failure. No force query or recomputation.
            raw = [
                {
                    key: np.asarray(getattr(c, key)).tolist()
                    for key in (
                        "geom1",
                        "geom2",
                        "dim",
                        "efc_address",
                        "pos",
                        "frame",
                        "dist",
                        "includemargin",
                        "friction",
                        "solref",
                        "solreffriction",
                        "solimp",
                    )
                }
                for c in data.contact
            ]
            with (self.output / "compact_native_failure.json").open("x") as stream:
                json.dump(
                    {"failure": self.failure, "raw_contacts": raw}, stream, indent=2
                )
            raise

    def finish_segment(self):
        if self.buffer is None:
            return None
        row = {
            "segment": self.label,
            "attempted_native": self.buffer.attempted,
            "returned_native": self.buffer.returned,
            "capacity": self.buffer.capacity,
        }
        self.buffer.save(self.folder / "native_arrays.npz")
        self.segments.append(row)
        self.buffer, self.folder, self.label = None, None, None
        return row

    def report(self):
        return {
            "record_mode": "compact_equivalence_probe",
            "checked_native_returns": self.checked,
            "contacts_checked": self.contacts,
            "box_contacts_checked": self.box_contacts,
            "max_abs_roll_pitch_deg": self.max_abs_rp_deg,
            "failure": self.failure,
            "segments": self.segments,
            "force_sampling_performed": False,
            "full_contact_qualification_claimed": False,
        }
