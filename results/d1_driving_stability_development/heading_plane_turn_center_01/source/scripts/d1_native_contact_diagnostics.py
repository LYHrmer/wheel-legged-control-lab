"""Read solved contact forces from the native step cache without refreshing it."""
from __future__ import annotations

import mujoco
import numpy as np


def sample_native_contacts(plant):
    """Sample the existing solver cache, not the synchronized endpoint cache.

    After implicitfast mj_step returns, data.time/qpos have advanced while these
    contact positions and solved forces belong to the native solver evaluation.
    Never interpret this as a newly computed endpoint contact measurement.
    """
    model, data = plant.model, plant.data
    reference = data.xpos[plant.base_body_id].copy()
    force, moment, normal_sum, tangent_sum = (np.zeros(3) for _ in range(4))
    contacts, counts = [], [0]*4
    nonwheel, geometric_wheel_count = 0, 0
    local = np.empty(6)
    max_horizontal, max_vertical_error = 0., 0.
    for index, c in enumerate(data.contact):
        g1, g2 = int(c.geom1), int(c.geom2)
        terrain1, terrain2 = g1 in plant.terrain_geom_ids, g2 in plant.terrain_geom_ids
        if terrain1 == terrain2:
            continue
        robot_geom = g2 if terrain1 else g1
        wheel = plant._wheel_index_by_body_id.get(int(model.geom_bodyid[robot_geom]))
        if wheel is None:
            nonwheel += int(c.efc_address >= 0)
            continue
        geometric_wheel_count += 1
        frame = np.asarray(c.frame).reshape(3, 3)
        normal = frame[0] * (1. if terrain1 else -1.)
        max_horizontal = max(max_horizontal, float(np.linalg.norm(normal[:2])))
        max_vertical_error = max(max_vertical_error, abs(float(abs(normal[2])-1.)))
        if c.efc_address < 0:
            continue
        mujoco.mj_contactForce(model, data, index, local)
        sign = 1. if terrain1 else -1.
        world_force, world_torque = sign*(frame.T@local[:3]), sign*(frame.T@local[3:])
        normal_force = float(np.dot(world_force, normal))*normal
        tangent_force = world_force-normal_force
        counts[wheel] += 1
        normal_sum += normal_force
        tangent_sum += tangent_force
        force += world_force
        moment += np.cross(c.pos-reference, world_force)+world_torque
        contacts.append({"wheel": wheel, "robot_geom_id": robot_geom,
            "pos_world_m": c.pos.tolist(), "normal_terrain_to_robot_world": normal.tolist(),
            "normal_force_world_n": normal_force.tolist(),
            "tangent_force_world_n": tangent_force.tolist(),
            "force_world_n": world_force.tolist(), "torque_world_nm": world_torque.tolist()})
    return {"sampling": "native_step_solved_cache_not_synchronized_endpoint",
        "reference_world_m": reference.tolist(), "contacts": contacts,
        "active_contacts_by_wheel": counts, "geometric_wheel_contact_count": geometric_wheel_count,
        "active_nonwheel_contacts": nonwheel, "max_horizontal_normal": max_horizontal,
        "max_vertical_normal_error": max_vertical_error,
        "summed_normal_force_world_n": normal_sum.tolist(),
        "summed_tangent_force_world_n": tangent_sum.tolist(),
        "total_wrench_world_6": np.r_[force, moment].tolist()}
