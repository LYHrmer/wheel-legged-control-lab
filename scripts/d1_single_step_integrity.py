"""Pure archive cross-link checks for compiled identities and native summaries."""
from __future__ import annotations

import numpy as np


def validate_record_links(payload):
    """Reject lost collision geoms, forged aggregate loads and inconsistent pairs."""
    manifest = payload["geometry_manifest"]
    geoms = {row["geom_id"]: row for row in manifest["geoms"]}
    if len(geoms) != len(manifest["geoms"]) or set(geoms) != set(range(len(geoms))):
        raise ValueError("compiled geom manifest is not a unique contiguous mapping")
    robot = {row["identity"]: row for row in geoms.values()
             if row["body_id"] > 0 and row["collision"]}
    if len(robot) != sum(row["body_id"] > 0 and row["collision"] for row in geoms.values()):
        raise ValueError("duplicate robot geom identities")
    for ep in payload["endpoints"]:
        bounds = ep["collision_bounds"]
        if len(bounds) != len(robot) or {r["identity"] for r in bounds} != set(robot):
            raise ValueError("endpoint omitted or duplicated physical collision geometry")
        for row in bounds:
            expected = robot[row["identity"]]
            for field in ("geom_id", "body_name", "geom_type", "wheel_index", "margin_m"):
                if row[field] != expected[field]:
                    raise ValueError("collision bound identity differs from compiled manifest: "+field)
        tick = ep["tick"]
        native = payload["native"][0 if tick == 0 else 5*tick-1]
        q = np.asarray(native["qpos_before" if tick == 0 else "qpos_returned"])
        v = np.asarray(native["qvel_before" if tick == 0 else "qvel_returned"])
        w, x, y, z = q[3:7]
        rpy = [np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)), np.arcsin(np.clip(2*(w*y-z*x), -1, 1)),
               np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))]
        forward_axis = np.array([1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)])
        base_com_vx = float(forward_axis@v[:3]+np.cross(v[3:6], manifest["base_body_ipos_local_m"])[0])
        if (not np.array_equal(q[:3], ep["base_position_m"])
                or not np.allclose(rpy, ep["rpy_rad"], rtol=0., atol=1e-12)
                or abs(base_com_vx-ep["body_vx_mps"]) > 1e-12
                or ep["com_vz_mps"] != ep["com_velocity_mps"][2]):
            raise ValueError("endpoint state disagrees with corresponding native return")
    previous = None
    for index, entry in enumerate(payload["native"]):
        cdata = entry["contacts"]
        loads, box_loads = np.zeros(4), np.zeros(4)
        nonwheel, geometric_box = 0, False
        for contact in cdata["contacts"]:
            for field in ("distance_m", "inclusion_margin_m", "friction", "position_world_m"):
                if not np.isfinite(np.asarray(contact[field], dtype=float)).all():
                    raise ValueError("nonfinite contact field: "+field)
            first, second = geoms[contact["geom1"]], geoms[contact["geom2"]]
            for num, actual in ((1, first), (2, second)):
                if (contact[f"geom{num}_identity"] != actual["identity"]
                        or contact[f"body{num}"] != actual["body_id"]
                        or contact[f"body{num}_name"] != actual["body_name"]):
                    raise ValueError("native geom/body identity mismatch")
            frame = np.asarray(contact["frame_geom1_to_geom2"], dtype=float)
            if (frame.shape != (3, 3) or not np.isfinite(frame).all()
                    or not np.allclose(frame@frame.T, np.eye(3), rtol=0., atol=1e-10)
                    or abs(np.linalg.det(frame)-1.) > 1e-10):
                raise ValueError("native frame is not orthonormal")
            local = np.asarray(contact["local_force_torque"], dtype=float)
            if local.shape != (6,) or not np.isfinite(local).all() or local[0] != contact["normal_load_n"]:
                raise ValueError("native local force and reported normal load disagree")
            active = contact["efc_address"] >= 0
            if active != contact["active"] or (not active and np.any(local)):
                raise ValueError("native active/load fields disagree")
            t1, t2 = first["terrain_kind"] is not None, second["terrain_kind"] is not None
            terrain = (first if t1 else second) if t1 != t2 else None
            other = (second if t1 else first) if terrain else None
            wheel = other["wheel_index"] if other else None
            if (contact["terrain_geom_id"] != (terrain["geom_id"] if terrain else None)
                    or contact["robot_geom_id"] != (other["geom_id"] if other else None)
                    or contact["wheel_index"] != wheel):
                raise ValueError("native terrain/wheel classification differs from compiled model")
            if terrain:
                normal = (1. if t1 else -1.)*frame[0]
                if not np.array_equal(normal, contact["normal_terrain_to_robot_world"]):
                    raise ValueError("native terrain normal has wrong geom-order sign")
                if not np.array_equal((1. if t1 else -1.)*(frame.T@local[:3]), contact["force_on_robot_world_n"]):
                    raise ValueError("native signed world force disagrees with local force")
                if terrain["terrain_kind"] == "plane" and not np.allclose(normal, (0., 0., 1.), rtol=0., atol=1e-10):
                    raise ValueError("native plane normal invalid")
                if terrain["terrain_kind"] == "box":
                    geometric_box = True
                    pos = np.asarray(contact["position_world_m"])
                    center, half = np.asarray(terrain["position_local_m"]), np.asarray(terrain["size_m"])
                    tol = abs(contact["distance_m"])+max(first["margin_m"], second["margin_m"], contact["inclusion_margin_m"])+1e-7
                    support = center+np.sign(normal)*half
                    candidate = (np.abs(pos-support) <= tol) & (np.abs(normal) > 1e-6)
                    residual = np.linalg.norm(np.where(candidate, 0., normal))
                    if not candidate.any() or residual > .0021 or np.any(np.abs(pos-center) > half+tol):
                        raise ValueError("native box normal inconsistent with compiled position feature")
                    if contact["box_feature"] is None or not contact["box_feature"]["geometric_support_valid"]:
                        raise ValueError("native box feature missing or invalid")
                elif contact["box_feature"] is not None:
                    raise ValueError("box feature assigned to a non-box geom")
                if wheel is None:
                    nonwheel += 1
                elif active and local[0] > 0.:
                    loads[wheel] += local[0]
                    if terrain["terrain_kind"] == "box":
                        box_loads[wheel] += local[0]
        if (not np.array_equal(loads, cdata["wheel_positive_normal_load_n"])
                or not np.array_equal(box_loads, cdata["wheel_box_positive_normal_load_n"])
                or nonwheel != cdata["nonwheel_terrain_contacts"]
                or geometric_box != cdata["geometric_box_contact"]):
            raise ValueError("native contact aggregates disagree with actual contact rows")
        k = index//5
        if k < len(payload["trace"]) and not np.array_equal(entry["ctrl_nm"], payload["trace"][k]["torque_nm"]):
            raise ValueError("native held ctrl disagrees with executed protected control")
        if previous is not None and not np.array_equal(previous, entry["qpos_before"]):
            raise ValueError("native pose chain broken")
        previous = entry["qpos_returned"]
        if (index % 5 == 4 and k+1 < len(payload["endpoints"])
                and not np.array_equal(np.asarray(previous)[:3], payload["endpoints"][k+1]["base_position_m"])):
            raise ValueError("control endpoint position differs from returned native state")
