"""Independent two-primitive binding/contact verification; saved data only."""
from pathlib import Path
from collections import Counter
import math
import numpy as np
from verify_course_e_08_03 import require, close, _rotation, read
from geometry24 import validate_geometry24


def family(name):
    if name == 'floor':
        return 'floor'
    if name == 'terrain_single_15mm_box':
        return 'step'
    raise ValueError('unregistered actual single-step terrain')


def check_binding(saved: dict) -> dict:
    geom_bodyid = np.asarray(saved["geom_bodyid"], dtype=np.int64)
    geom_type = np.asarray(saved["geom_type"], dtype=np.int64)
    geom_size = np.asarray(saved["geom_size"], dtype=np.float64)
    geom_contype = np.asarray(saved["geom_contype"], dtype=np.int64)
    geom_conaffinity = np.asarray(saved["geom_conaffinity"], dtype=np.int64)
    n = len(geom_bodyid)
    require(n > 2 and geom_type.shape == (n,)
            and geom_size.shape == (n, 3)
            and geom_contype.shape == geom_conaffinity.shape == (n,),
            "E actual compiled robot/terrain geom arrays differ")
    wheel_map = {int(key): value for key, value in saved["wheel_index_by_body_id"].items()}
    require(len(wheel_map) == 4 and set(wheel_map.values()) == set(range(4))
            and all(body > 0 for body in wheel_map),
            "E actual wheel body binding is not four distinct compiled bodies")
    qpos = np.asarray(saved["joint_qpos_addresses"], dtype=np.int64)
    dof = np.asarray(saved["joint_dof_addresses"], dtype=np.int64)
    joint_ids = np.asarray(saved["joint_ids"], dtype=np.int64)
    actuator_ids = np.asarray(saved["actuator_ids"], dtype=np.int64)
    trnid = np.asarray(saved["actuator_trnid"], dtype=np.int64)
    gear = np.asarray(saved["actuator_gear"], dtype=np.float64)
    require(qpos.shape == dof.shape == joint_ids.shape == actuator_ids.shape == (16,)
            and len(set(qpos)) == len(set(dof)) == 16
            and np.all(qpos >= 7) and np.all(dof >= 6)
            and np.array_equal(trnid[actuator_ids, 0], joint_ids)
            and np.all(gear[actuator_ids, 0] == 1.0)
            and np.all(gear[actuator_ids, 1:] == 0.0),
            "E actual joint/actuator direct-torque binding differs")
    ipos = np.asarray(saved["base_body_ipos"], dtype=np.float64)
    mass = np.asarray(saved["body_mass"], dtype=np.float64)
    require(ipos.shape == (3,) and np.isfinite(ipos).all()
            and 0 < int(saved["base_body_id"]) < len(mass)
            and np.isfinite(mass).all() and close(float(np.sum(mass)),
                                                   saved["nominal_total_mass_kg"], atol=1e-8),
            "E compiled base COM offset/body mass differs")
    result = {
        "geom_bodyid": geom_bodyid, "geom_type": geom_type,
        "geom_size": geom_size, "geom_contype": geom_contype,
        "geom_conaffinity": geom_conaffinity, "wheel_map": wheel_map,
        "qpos_addresses": qpos, "dof_addresses": dof,
        "actuator_ids": actuator_ids,
        "base_body_ipos": ipos, "base_body_id": int(saved["base_body_id"]),
    }
    kin = saved['kinematics24']
    for key in ('geom_bodyid','geom_type','geom_size','geom_contype',
                'geom_conaffinity','body_mass'):
        require(close(kin[key],saved[key]),'C24 kinematic/physical compiled binding differs: '+key)
    require(close(kin['body_ipos'][int(saved['base_body_id'])],ipos)
            and len(kin['body_parentid'])==len(mass)
            and len(kin['geom_pos'])==n and len(kin['geom_margin'])==n,
            'C24 full-system kinematic binding dimensions differ')
    result['kinematics24'] = kin
    return result


def geom_map(folder: Path, binding: dict) -> dict[int, dict]:
    manifest = read(folder / "geometry_manifest.json")
    validate_geometry24(manifest)
    rows = manifest["world_collision_geoms"]
    require(len(rows) == 2 and manifest["control_dt_s"] == .01
            and manifest["native_dt_s"] == .002,
            "actual compiled course layout/timing differs")
    require(Counter(family(row["name"]) for row in rows) == {"floor":1,"step":1},
            "compiled course terrain family counts differ")
    mapping = {}
    kinds: dict[str, set[int]] = {"plane": set(), "box": set()}
    for row in rows:
        gid = row["geom_id"]
        require(type(gid) is int and gid not in mapping
                and 0 <= gid < len(binding["geom_bodyid"])
                and row["body_id"] == 0 and row["collision"] is True
                and binding["geom_bodyid"][gid] == 0
                and (binding["geom_contype"][gid] != 0
                     or binding["geom_conaffinity"][gid] != 0)
                and close(binding["geom_size"][gid], row["size_m"])
                and row["type"] == ("plane" if row["name"] == "floor" else "box"),
                "compiled course terrain identity invalid")
        quat = np.asarray(row["quaternion_wxyz"], dtype=np.float64)
        require(close(quat @ quat, 1.0, atol=1e-12)
                and np.isfinite(row["position_m"]).all()
                and np.isfinite(row["size_m"]).all(),
                "compiled course primitive has invalid transform")
        kinds[row["type"]].add(int(binding["geom_type"][gid]))
        mapping[gid] = row
    require(len(kinds["plane"]) == len(kinds["box"]) == 1
            and kinds["plane"] != kinds["box"],
            "actual compiled terrain plane/box geom types are not distinct")
    actual_world = {i for i,b in enumerate(binding["geom_bodyid"]) if b==0 and (binding["geom_contype"][i] or binding["geom_conaffinity"][i])}
    require(set(mapping)==actual_world, "hidden/unbound actual world collision geom")
    return mapping


def check_contact(contact: dict, geoms: dict[int, dict], binding: dict) -> tuple[bool, str | None]:
    """Independent local rotated-box cone check; never calls engine geometry."""
    g1, g2 = contact["geom1"], contact["geom2"]
    require(0 <= g1 < len(binding["geom_bodyid"])
            and 0 <= g2 < len(binding["geom_bodyid"]),
            "native contact geom outside actual compiled model")
    t1, t2 = g1 in geoms, g2 in geoms
    require(not (t1 and t2), "native terrain-terrain contact")
    frame = np.asarray(contact["frame_world"], dtype=np.float64)
    require(frame.shape == (3, 3) and np.isfinite(frame).all()
            and close(frame @ frame.T, np.eye(3))
            and abs(float(np.linalg.det(frame))-1) <= 1e-10,
            "native contact frame invalid")
    for key in ("position_world_m", "friction", "solref", "solreffriction", "solimp",
                "geom1_xpos", "geom2_xpos", "geom1_xmat", "geom2_xmat"):
        require(np.isfinite(np.asarray(contact[key], dtype=np.float64)).all(),
                f"native contact cache nonfinite: {key}")
    if contact["efc_address"] >= 0:
        wrench = np.asarray(contact["contact_force_local"], dtype=np.float64)
        require(wrench.shape == (6,) and np.isfinite(wrench).all()
                and contact["force_source"] == "mj_contactForce_actual_native_cache",
                "actual native contact force record absent")
    if not (t1 or t2):
        return False, None
    terrain = g1 if t1 else g2
    robot = g2 if t1 else g1
    row = geoms[terrain]
    actual_body = int(binding["geom_bodyid"][robot])
    require(contact["robot_geom_id"] == robot
            and contact["robot_body_id"] == actual_body
            and contact["robot_wheel_index"] == binding["wheel_map"].get(actual_body),
            "native contact robot/wheel binding differs from compiled model")
    outward = frame[0] if t1 else -frame[0]
    pos = np.asarray(contact["position_world_m"], dtype=np.float64)
    tol = (abs(float(contact["distance_m"]))
           + max(float(contact["geom1_compiled_margin"]),
                 float(contact["geom2_compiled_margin"]),
                 float(contact["includemargin_m"])) + 1e-7)
    if row["name"] == "floor":
        require(close(outward, (0.0, 0.0, 1.0)) and abs(float(pos[2])) <= tol,
                "native floor contact normal/point invalid")
    else:
        rotation = _rotation(row["quaternion_wxyz"])
        # Native world cache is checked against the actual compiled static geom.
        require(close(contact[f"geom{1 if t1 else 2}_xpos"], row["position_m"])
                and close(contact[f"geom{1 if t1 else 2}_xmat"], rotation),
                "native rotated terrain cache differs from compiled primitive")
        local = rotation.T @ (pos - np.asarray(row["position_m"], dtype=np.float64))
        normal = rotation.T @ outward
        half = np.asarray(row["size_m"], dtype=np.float64)
        require(np.isfinite(local).all() and np.isfinite(normal).all()
                and np.all(local >= -half-tol) and np.all(local <= half+tol),
                "native rotated box contact is outside actual extents")
        support = np.zeros(3)
        faces = []
        face_names = {(0,-1):"front",(0,1):"back",(1,-1):"right",(1,1):"left",(2,-1):"bottom",(2,1):"top"}
        for axis in range(3):
            if abs(float(normal[axis])) > 1e-6:
                face = math.copysign(float(half[axis]), float(normal[axis]))
                if abs(float(local[axis])-face) <= tol:
                    support[axis] = normal[axis]
                    faces.append(face_names[(axis,1 if normal[axis]>0 else -1)])
        require(np.any(support) and np.linalg.norm(normal-support) <= .0021,
                "native rotated box normal cone invalid")
        feature = faces[0] if len(faces)==1 else "_".join(faces)+("_edge" if len(faces)==2 else "_corner")
        require(contact["geometric_feature"]==feature and contact["geometry_support_valid"] is True, "C24 actual box contact feature differs")
    require(close(contact["terrain_to_robot_normal_world"],outward), "C24 terrain-to-robot normal differs")
    require(contact["terrain_geom_id"] == terrain
            and contact["terrain_geom_name"] == row["name"]
            and contact["terrain_family"] == family(row["name"]),
            "native contact terrain identity differs")
    load = False
    if contact["efc_address"] >= 0:
        force = frame.T @ np.asarray(contact["contact_force_local"][:3], dtype=np.float64)
        if not t1:
            force = -force
        normal_load = float(force @ outward)
        load = contact["robot_wheel_index"] in range(4) and normal_load > 0.0
        require(close(contact["force_on_robot_world_n"], force)
                and close(contact["normal_load_on_robot_n"], normal_load),
                "native contact force transform differs from recorded solver force")
    require(contact["positive_wheel_load"] is load,
            "positive wheel load flag differs from actual bound contact force")
    return load, family(row["name"])

