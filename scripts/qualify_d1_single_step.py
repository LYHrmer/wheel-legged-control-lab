"""Zero-integration geometry/contact qualification for the frozen 15 mm box."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from scripts.d1_single_step_geometry import (
    collision_identity,
    robot_collision_bounds,
    robot_collision_ids,
)
from scripts.d1_single_step_plant import BOX_CENTER_M, BOX_HALF_SIZE_M
from scripts.d1_single_step_records import PhysicsCallLedger, jsonable, sample_step_contacts

ROOT = Path(__file__).resolve().parents[1]


def source_manifest():
    paths = list((ROOT/"src").rglob("*.py"))+list((ROOT/"scripts").glob("*.py"))
    paths += list((ROOT/"tests").glob("test_d1_single_step*.py"))
    paths += list((ROOT/"src/wheel_legged_control/d1/assets").rglob("*"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths)) if p.is_file() and "__pycache__" not in p.parts}


def model_comparison(first, second):
    """Compare physical arrays by structural geom identity, with explicit ID map."""
    mismatches, checked = [], []
    for prefix, names in {
        "body": "parentid mass inertia ipos iquat pos quat jntnum jntadr dofnum dofadr gravcomp mocapid",
        "jnt": "type bodyid pos axis limited range margin stiffness solref solimp qposadr dofadr",
        "dof": "bodyid jntid parentid damping armature frictionloss solref solimp",
        "actuator": "trntype trnid gear ctrllimited ctrlrange forcelimited forcerange dyntype dynprm gaintype gainprm biastype biasprm actlimited actrange",
        "mesh": "vert face normal texcoord scale pos quat",
        "mat": "rgba friction texrepeat reflectance shininess specular emission",
    }.items():
        for name in names.split():
            field = prefix+"_"+name
            if not hasattr(first, field):
                continue
            checked.append(field)
            if not np.array_equal(getattr(first, field), getattr(second, field)):
                mismatches.append(field)
    for field in dir(first.opt):
        if field.startswith("_"):
            continue
        left, right = getattr(first.opt, field), getattr(second.opt, field)
        if isinstance(left, (int, float, np.number, np.ndarray)):
            checked.append("opt."+field)
            if not np.array_equal(left, right):
                mismatches.append("opt."+field)
    maps = [{collision_identity(m, g): g for g in range(m.ngeom)} for m in (first, second)]
    mapping = []
    for key, g in maps[0].items():
        if key not in maps[1]:
            mismatches.append("missing geom "+key)
            continue
        other = maps[1][key]
        mapping.append({"identity": key, "first_id": g, "second_id": other})
        for field in ["type", "size", "pos", "quat", "contype", "conaffinity", "condim", "friction", "margin", "gap", "solref", "solimp", "priority", "solmix", "rgba", "group", "dataid", "matid", "bodyid", "rbound", "aabb"]:
            attr = "geom_"+field
            if hasattr(first, attr):
                checked.append(key+":"+attr)
                if not np.array_equal(getattr(first, attr)[g], getattr(second, attr)[other]):
                    mismatches.append(key+":"+attr)
    return {"passed": not mismatches, "mismatches": mismatches,
            "checked_fields": checked, "geom_mapping": mapping,
            "additional_geoms": sorted(set(maps[1])-set(maps[0]))}


def query_checks(plant):
    center, half = np.asarray(BOX_CENTER_M), np.asarray(BOX_HALF_SIZE_M)
    points = [(center[0], center[1])]
    # Exact closed faces/corners, both sides of each edge, and separated plane.
    for x in (center[0]-half[0], center[0], center[0]+half[0]):
        for y in (center[1]-half[1], center[1], center[1]+half[1]):
            points.append((x, y))
    for axis in range(2):
        for sign in (-1, 1):
            for delta in (-1e-8, 1e-8):
                xy = center[:2].copy(); xy[axis] += sign*(half[axis]+delta)
                points.append(tuple(xy))
    rows = []
    for xy in points:
        pnt = np.array([*xy, 1.]); vec = np.array([0., 0., -1.])
        hits = []
        for g in plant.terrain_geom_ids:
            dist = mujoco.mju_rayGeom(plant.model.geom_pos[g], np.eye(3).ravel(),
                plant.model.geom_size[g], pnt, vec, int(plant.model.geom_type[g]))
            if dist >= 0.:
                hits.append((dist, int(g)))
        nearest = min(hits)
        queried = plant.locomotion_ground_reference(*xy)
        # Exact world boundary can subtract to half_size + one ULP inside the
        # ray routine. Keep the raw result and independently measure distance
        # to the compiled closed box, without widening the height query.
        signed_distance = None
        boundary_verified = False
        interior_ray = None
        outward_queries = []
        if plant.obstacle_enabled and queried.height_m == .015:
            g = plant.box_geom_id
            point_on_top = np.array([*xy, queried.height_m])
            q = np.abs(point_on_top-plant.model.geom_pos[g])-plant.model.geom_size[g]
            signed_distance = float(np.linalg.norm(np.maximum(q, 0.))+min(float(np.max(q)), 0.))
            tolerance = 8*np.finfo(float).eps*max(1., np.max(np.abs(point_on_top)))
            exact_boundary = any(xy[i] in (center[i]-half[i], center[i]+half[i]) for i in range(2))
            if exact_boundary:
                interior = np.array([*xy, 1.])
                for axis in range(2):
                    if xy[axis] in (center[axis]-half[axis], center[axis]+half[axis]):
                        interior[axis] = np.nextafter(xy[axis], center[axis])
                        outward = list(xy)
                        outward[axis] = np.nextafter(xy[axis], np.inf if xy[axis] > center[axis] else -np.inf)
                        outward_queries.append(plant.locomotion_ground_reference(*outward).height_m)
                interior_ray = mujoco.mju_rayGeom(plant.model.geom_pos[g], np.eye(3).ravel(),
                    plant.model.geom_size[g], interior, vec, int(plant.model.geom_type[g]))
                boundary_verified = bool(abs(signed_distance) <= tolerance
                    and abs(1.-interior_ray-.015) < 1e-12 and all(v == 0. for v in outward_queries))
        rows.append({"xy": xy, "height_query_m": queried.height_m,
                     "ray_height_m": 1.-nearest[0], "hit_geom_id": nearest[1],
                     "compiled_box_point_signed_distance_m": signed_distance,
                     "closed_boundary_verified_by_distance": boundary_verified,
                     "nearest_interior_ray_distance_m": interior_ray,
                     "outward_nextafter_query_heights_m": outward_queries,
                     "passed": abs(queried.height_m-(1.-nearest[0])) < 1e-12 or boundary_verified})
    return rows


def static_contact_checks(plant):
    """Declared synthetic poses on separate data; no dynamic load claim."""
    model = plant.model
    nominal = plant.data.qpos.copy()
    geom = next(g for g in robot_collision_ids(model)
                if model.geom_bodyid[g] == plant.wheel_body_ids_by_leg[0])
    d = mujoco.MjData(model)
    d.qpos[:] = nominal
    mujoco.mj_kinematics(model, d)
    initial_center = d.geom_xpos[geom].copy()
    r = float(model.geom_size[geom, 0])
    c, h = np.asarray(BOX_CENTER_M), np.asarray(BOX_HALF_SIZE_M)
    front, top = c[0]-h[0], c[2]+h[2]
    edge_distance = (r-.0002)/np.sqrt(2.)
    targets = {
        "plane": [c[0]-.8, 0., r-.0002],
        "top": [c[0], 0., top+r-.0002],
        "front": [front-r+.0002, 0., c[2]],
        "front_top_edge": [front-edge_distance, 0., top+edge_distance],
        "separated": [c[0]-.8, 0., .5],
    }
    rows = []
    from types import SimpleNamespace
    for name, target in targets.items():
        d = mujoco.MjData(model); d.qpos[:] = nominal
        d.qpos[:3] += np.asarray(target)-initial_center
        mujoco.mj_kinematics(model, d); mujoco.mj_collision(model, d)
        sample = sample_step_contacts(SimpleNamespace(model=model, data=d, floor_geom_id=plant.floor_geom_id, box_geom_id=plant.box_geom_id, terrain_geom_ids=plant.terrain_geom_ids, _wheel_index_by_body_id=plant._wheel_index_by_body_id))
        wheel = [row for row in sample["contacts"] if row["wheel_index"] == 0]
        relevant = [row for row in wheel if row["terrain_geom_id"] == (
            plant.floor_geom_id if name == "plane" else plant.box_geom_id)]
        valid = all(row["frame_orthonormal"] and row["plane_normal_valid"]
                    and (row["box_feature"] is None or row["box_feature"]["geometric_support_valid"])
                    for row in relevant)
        if name == "separated":
            valid &= not wheel
        else:
            valid &= bool(relevant)
            if name in ("top", "front", "front_top_edge"):
                valid &= any(row["box_feature"]["feature"] == name for row in relevant)
        rows.append({"pose": name, "wheel_center_world_m": target,
                     "contacts": sample["contacts"], "passed": bool(valid),
                     "time_s": float(d.time), "sampling": "static_kinematics_collision_no_solve",
                     "dynamic_load_qualification": False})
    # Real body collision geometry at plane and box. No wheel-load criterion.
    base_geom = next(g for g in robot_collision_ids(model) if model.geom_bodyid[g] == plant.base_body_id)
    d = mujoco.MjData(model); d.qpos[:] = nominal; mujoco.mj_kinematics(model, d)
    base_center = d.geom_xpos[base_geom].copy()
    for name, target in (("nonwheel_plane", [c[0]-.8, 0., 0.]), ("nonwheel_box", c)):
        d = mujoco.MjData(model); d.qpos[:] = nominal; d.qpos[:3] += np.asarray(target)-base_center
        mujoco.mj_kinematics(model, d); mujoco.mj_collision(model, d)
        wanted = plant.floor_geom_id if name.endswith("plane") else plant.box_geom_id
        contacts = [{"geom1": int(v.geom1), "geom2": int(v.geom2), "distance_m": float(v.dist)}
                    for v in d.contact if base_geom in (v.geom1, v.geom2) and wanted in (v.geom1, v.geom2)]
        rows.append({"pose": name, "contacts": contacts, "passed": bool(contacts),
                     "time_s": float(d.time), "sampling": "static_kinematics_collision_no_solve"})
    return rows


def qualify():
    from scripts.d1_flat_plane_env import D1FlatPlanePlant
    from scripts.d1_single_step_env import D1SingleStepEnv
    from wheel_legged_control.d1.locomotion_env import LOCOMOTION_SPAWN_POSITION_M
    from wheel_legged_control.d1.model import D1Plant

    envs = [D1SingleStepEnv(obstacle_enabled=value) for value in (False, True)]
    original = D1FlatPlanePlant()
    original.reset(base_position=LOCOMOTION_SPAWN_POSITION_M)
    for env in envs:
        env.reset(seed=77301)
    flat, box = [env.plant for env in envs]
    old_comparison = model_comparison(original.model, flat.model)
    paired_comparison = model_comparison(flat.model, box.model)
    initial_equal = all(np.array_equal(getattr(original.data, f), getattr(p.data, f))
                        for p in (flat, box) for f in ("qpos", "qvel", "ctrl", "qacc_warmstart"))
    checks = {"original_flat_physics_equal": old_comparison["passed"],
              "paired_robot_physics_equal": paired_comparison["passed"],
              "only_added_box": paired_comparison["additional_geoms"] == ["terrain_single_15mm_box"],
              "initial_state_bitwise_equal": initial_equal,
              "inherited_step_and_reset": D1SingleStepEnv.__mro__[1].__name__ == "D1HeadingTrackingEnv"
                  and type(box).step is D1Plant.step and type(box).reset is D1Plant.reset,
              "separate_data": all(p.data is not p._measurement_data for p in (flat, box)),
              "no_mocap_hfield": all(p.model.nmocap == p.model.nhfield == 0 for p in (flat, box))}
    bounds = robot_collision_bounds(box.model, box._measurement_data)
    checks["initial_box_separation"] = all(row["maximum_world_m"][0] < BOX_CENTER_M[0]-BOX_HALF_SIZE_M[0] for row in bounds)
    queries = {str(p.obstacle_enabled): query_checks(p) for p in (flat, box)}
    checks["query_ray_agreement"] = all(row["passed"] for rows in queries.values() for row in rows)
    contacts = static_contact_checks(box)
    checks["static_contact_geometry"] = all(row["passed"] for row in contacts)
    velocity_checks = []
    for quaternion in ((1., 0., 0., 0.), (1., .1, .2, .3)):
        scratch = mujoco.MjData(box.model)
        scratch.qpos[:] = box.data.qpos
        scratch.qpos[3:7] = np.asarray(quaternion)/np.linalg.norm(quaternion)
        scratch.qvel[:6] = [.1, .2, .3, .4, .5, .6]
        mujoco.mj_kinematics(box.model, scratch); mujoco.mj_comPos(box.model, scratch)
        jacobian = np.zeros((3, box.model.nv))
        mujoco.mj_jacBodyCom(box.model, scratch, jacobian, None, box.base_body_id)
        rotation = scratch.xmat[box.base_body_id].reshape(3, 3)
        from_jacobian = rotation.T@(jacobian@scratch.qvel)
        from_free_joint = rotation.T@scratch.qvel[:3]+np.cross(scratch.qvel[3:6], box.model.body_ipos[box.base_body_id])
        velocity_checks.append({"jacobian_body_com_velocity_mps": from_jacobian.tolist(),
            "free_joint_offset_formula_mps": from_free_joint.tolist(),
            "passed": bool(np.allclose(from_jacobian, from_free_joint, rtol=0., atol=1e-12))})
    checks["base_com_velocity_semantics"] = all(row["passed"] for row in velocity_checks)
    result = {"passed": all(checks.values()), "checks": checks,
              "original_flat_comparison": old_comparison, "paired_comparison": paired_comparison,
              "initial_collision_bounds": bounds, "query_ray_checks": queries,
              "static_contact_checks": contacts,
              "base_com_velocity_checks": velocity_checks,
              "nominal_parameters": [asdict(env.wheel_leg_control) for env in envs],
              "terrain_metadata": [p.collision_terrain_metadata for p in (flat, box)]}
    for env in envs:
        env.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {}
    with PhysicsCallLedger(native_limit=0) as ledger:
        try:
            result = qualify()
        except BaseException as error:  # noqa: BLE001 -- persist partial static qualification
            result = {"passed": False, "error": {"type": type(error).__name__, "message": str(error)}}
        finally:
            result["ledger"] = ledger.receipt()
            result["input_sha256"] = source_manifest()
            (args.output/"qualification.json").write_text(json.dumps(jsonable(result), indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: result.get(k) for k in ("passed", "checks", "error", "ledger")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
