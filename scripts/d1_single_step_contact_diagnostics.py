"""Pure inspection of saved raw contacts, keeping geometry failure separate."""
from __future__ import annotations

import numpy as np


def audit_contact(contact, geoms):
    """Report every inspected inconsistency without filtering unloaded contacts."""
    out = {"raw_issues": [], "geometry_issues": [], "is_box": False,
           "wheel_index": None, "positive_load": False, "normal_load_n": None,
           "box_feature": None}
    raw, geometry = out["raw_issues"], out["geometry_issues"]

    def match(condition, code):
        if not condition:
            raw.append(code)

    def number(value, field):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value):
            raise ValueError("nonfinite_or_nonnumeric:"+field)
        return float(value)

    def array(value, shape, field):
        result = np.asarray(value)
        if result.shape != shape or result.dtype.kind not in "iuf" or not np.isfinite(result).all():
            raise ValueError("nonfinite_or_bad_shape:"+field)
        return result.astype(float)

    try:
        if not isinstance(contact, dict):
            raise TypeError("contact_not_dict")
        c = contact
        for field in ("geom1", "geom2", "body1", "body2", "efc_address", "index"):
            if type(c[field]) is not int:
                raise TypeError("noninteger:"+field)
        first, second = geoms[c["geom1"]], geoms[c["geom2"]]
        for i, geom in ((1, first), (2, second)):
            match(c[f"geom{i}_identity"] == geom["identity"], f"geom{i}_identity")
            match(c[f"body{i}"] == geom["body_id"], f"body{i}_id")
            match(c[f"body{i}_name"] == geom["body_name"], f"body{i}_name")
        distance = number(c["distance_m"], "distance_m")
        margin = number(c["inclusion_margin_m"], "inclusion_margin_m")
        match(type(c["dimension"]) is int and 1 <= c["dimension"] <= 6,
              "contact_dimension")
        array(c["friction"], (5,), "friction")
        pos = array(c["position_world_m"], (3,), "position_world_m")
        frame = array(c["frame_geom1_to_geom2"], (3, 3), "contact_frame")
        frame_ok = bool(np.allclose(frame@frame.T, np.eye(3), rtol=0., atol=1e-10)
                        and abs(np.linalg.det(frame)-1.) <= 1e-10)
        match(frame_ok, "frame_not_orthonormal_right_handed")
        match(type(c["frame_orthonormal"]) is bool and c["frame_orthonormal"] == frame_ok,
              "frame_orthonormal_flag")
        local = array(c["local_force_torque"], (6,), "local_force_torque")
        load = number(c["normal_load_n"], "normal_load_n")
        match(load == local[0], "normal_load_local_force_link")
        match(load >= 0., "negative_normal_load")
        active = c["efc_address"] >= 0
        match(type(c["active"]) is bool and c["active"] == active, "active_efc_link")
        match(active or not np.any(local), "inactive_force_nonzero")
        out["normal_load_n"] = float(local[0])
        out["positive_load"] = bool(active and local[0] > 0.)
        t1, t2 = first["terrain_kind"] is not None, second["terrain_kind"] is not None
        terrain = (first if t1 else second) if t1 != t2 else None
        other = (second if t1 else first) if terrain is not None else None
        wheel = other["wheel_index"] if other else None
        match(wheel is None or type(wheel) is int and 0 <= wheel < 4, "compiled_wheel_index")
        out["wheel_index"] = wheel
        match(c["terrain_geom_id"] == (terrain["geom_id"] if terrain else None), "terrain_geom_link")
        match(c["robot_geom_id"] == (other["geom_id"] if other else None), "robot_geom_link")
        match(c["wheel_index"] == wheel, "wheel_index_link")
        if terrain is None:
            match(c["normal_terrain_to_robot_world"] is None, "unexpected_terrain_normal")
            match(c["force_on_robot_world_n"] is None, "unexpected_terrain_force")
            match(c["box_feature"] is None, "unexpected_box_feature")
            return out
        normal = (1. if t1 else -1.)*frame[0]
        recorded_normal = array(c["normal_terrain_to_robot_world"], (3,), "terrain_normal")
        match(np.array_equal(normal, recorded_normal), "normal_geom_order_sign_link")
        force = (1. if t1 else -1.)*(frame.T@local[:3])
        match(np.array_equal(force, array(c["force_on_robot_world_n"], (3,), "world_force")),
              "world_force_local_frame_link")
        plane_ok = terrain["terrain_kind"] != "plane" or np.allclose(normal, (0., 0., 1.), rtol=0., atol=1e-10)
        match(type(c["plane_normal_valid"]) is bool and c["plane_normal_valid"] == bool(plane_ok),
              "plane_normal_flag")
        if not plane_ok:
            geometry.append("plane_normal_not_upward")
        if terrain["terrain_kind"] == "box":
            out["is_box"] = True
            center = array(terrain["position_local_m"], (3,), "box_center")
            half = array(terrain["size_m"], (3,), "box_half_size")
            if np.any(half <= 0.):
                raise ValueError("invalid_box_half_size")
            tol = abs(distance)+max(number(first["margin_m"], "margin1"),
                number(second["margin_m"], "margin2"), margin)+1e-7
            candidate = (abs(pos-(center+np.sign(normal)*half)) <= tol) & (abs(normal) > 1e-6)
            residual = float(np.linalg.norm(np.where(candidate, 0., normal)))
            valid = bool(candidate.any() and residual <= .0021 and np.all(abs(pos-center) <= half+tol))
            faces = {(0, -1): "front", (0, 1): "back", (1, -1): "right",
                     (1, 1): "left", (2, -1): "bottom", (2, 1): "top"}
            axes = np.flatnonzero(candidate).tolist()
            supporting_faces = [faces[i, int(np.sign(normal[i]))] for i in axes]
            offsets = [float(abs(pos[i] - center[i] - np.sign(normal[i])*half[i])) for i in axes]
            label = ("unresolved" if not supporting_faces else supporting_faces[0]
                     if len(supporting_faces) == 1 else
                     "_".join(supporting_faces) + ("_edge" if len(supporting_faces) == 2 else "_corner"))
            out["box_feature"] = {"candidate_axes": axes,
                "normal_cone_residual": residual, "normal_cone_residual_limit": .0021,
                "position_tolerance_m": tol, "geometric_support_valid": valid}
            if not valid:
                geometry.append("box_normal_outside_support_cone")
            feature = c["box_feature"]
            if not isinstance(feature, dict):
                raise TypeError("box_feature_missing")
            match(type(feature["geometric_support_valid"]) is bool
                  and feature["geometric_support_valid"] == valid, "saved_feature_validity_link")
            match(number(feature["normal_cone_residual_limit"], "saved_cone_limit") == .0021,
                  "saved_cone_limit_link")
            match(abs(number(feature["normal_cone_residual"], "saved_cone_residual")-residual) <= 1e-12,
                  "saved_cone_residual_link")
            match(feature["feature"] == label, "saved_feature_name_link")
            match(feature["supporting_faces"] == supporting_faces, "saved_supporting_faces_link")
            saved_offsets = array(feature["support_offsets_m"], (len(offsets),), "saved_support_offsets")
            match(np.allclose(saved_offsets, offsets, rtol=0., atol=1e-12), "saved_support_offsets_link")
            saved_normal = array(feature["normal_box_to_robot_world"], (3,), "saved_box_normal")
            match(np.allclose(saved_normal, normal / np.linalg.norm(normal), rtol=0., atol=1e-12),
                  "saved_box_normal_link")
        else:
            match(c["box_feature"] is None, "unexpected_box_feature")
    except Exception as exc:  # noqa: BLE001 -- malformed input must be reported, never accepted
        raw.append("malformed_contact:"+type(exc).__name__+":"+str(exc)[:180])
        geometry.append("geometry_not_evaluable")
    return out
