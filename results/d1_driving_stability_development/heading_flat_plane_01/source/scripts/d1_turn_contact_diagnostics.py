"""Read-only wheel/terrain contact telemetry for the D1 yaw-limit turn study.

This module is diagnostic only: it never steps physics, never refreshes or
resets the measurement cache, never builds a model or data object, and never
calls ``mj_forward``/``mj_step``.  It reads the already-synchronized endpoint
measurement cache (``plant.measurement_data``) and reports contact forces plus
contact-point kinematics using exactly the force/frame conventions used by
``D1Plant._measure_wheel_contact_wrench``.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

TURN_CONTACT_SCHEMA = "d1-turn-contact-kinematics-v1"

_WHEEL_JOINT_LOCAL_INDICES = (3, 7, 11, 15)
_WHEEL_COUNT = 4
_JOINT_DOF_COUNT = 16


def _tangent_speed(vector: np.ndarray, normal_world: np.ndarray) -> float:
    """Magnitude of ``vector`` after removing its component along the normal."""

    tangential = vector - float(np.dot(vector, normal_world)) * normal_world
    return float(np.linalg.norm(tangential))


def sample_turn_contacts(plant) -> dict[str, Any]:
    """Sample active wheel-terrain contacts from the synchronized measurement cache.

    Returns a JSON-serializable dict.  Wheels without an active contact report
    zero counts and zero force and contribute no per-contact velocity sample;
    absence of contact is never reported as a zero-slip measurement.
    """

    model = plant.model
    data = plant.measurement_data
    reference = np.asarray(plant.base_position.copy(), dtype=np.float64).reshape(3)

    dof_addresses = np.asarray(plant.dof_addresses, dtype=np.int64).reshape(-1)
    if dof_addresses.size != _JOINT_DOF_COUNT:
        raise ValueError(f"expected {_JOINT_DOF_COUNT} joint dof addresses")

    nv = int(model.nv)
    wheel_dof_addresses = dof_addresses[list(_WHEEL_JOINT_LOCAL_INDICES)]
    wheel_mask = np.zeros(nv, dtype=bool)
    wheel_mask[wheel_dof_addresses] = True
    joint_mask = np.zeros(nv, dtype=bool)
    joint_mask[dof_addresses] = True
    leg_mask = joint_mask & ~wheel_mask          # non-wheel leg dofs
    base_mask = ~joint_mask                      # remainder: free base dofs

    qvel = np.array(data.qvel, dtype=np.float64).reshape(nv)
    qvel_leg = np.where(leg_mask, qvel, 0.0)
    qvel_wheel = np.where(wheel_mask, qvel, 0.0)
    qvel_base = np.where(base_mask, qvel, 0.0)

    terrain_geom_ids = plant.terrain_geom_ids
    wheel_index_by_body_id = plant._wheel_index_by_body_id

    contacts: list[dict[str, Any]] = []
    wheel_forces = np.zeros((_WHEEL_COUNT, 3), dtype=np.float64)
    contact_counts = np.zeros(_WHEEL_COUNT, dtype=np.int64)
    total_force = np.zeros(3, dtype=np.float64)
    total_moment = np.zeros(3, dtype=np.float64)

    local_wrench = np.empty(6, dtype=np.float64)
    jacp_robot = np.empty((3, nv), dtype=np.float64)
    jacp_terrain = np.empty((3, nv), dtype=np.float64)

    for contact_id, contact in enumerate(data.contact):
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        terrain1 = geom1 in terrain_geom_ids
        terrain2 = geom2 in terrain_geom_ids
        if terrain1 == terrain2 or int(contact.efc_address) < 0:
            continue
        robot_geom = geom2 if terrain1 else geom1
        terrain_geom = geom1 if terrain1 else geom2
        robot_body = int(model.geom_bodyid[robot_geom])
        wheel_index = wheel_index_by_body_id.get(robot_body)
        if wheel_index is None:
            continue
        terrain_body = int(model.geom_bodyid[terrain_geom])

        mujoco.mj_contactForce(model, data, contact_id, local_wrench)
        contact_frame = np.array(contact.frame, dtype=np.float64).reshape(3, 3)
        force_world = contact_frame.T @ local_wrench[:3]
        torque_world = contact_frame.T @ local_wrench[3:]
        if robot_geom == geom1:
            force_world = -force_world
            torque_world = -torque_world

        contact_pos = np.array(contact.pos, dtype=np.float64).reshape(3)
        normal_world = contact_frame[0].copy()

        # Contact-point Jacobians: body-origin COM velocities are NOT used here.
        mujoco.mj_jac(model, data, jacp_robot, None, contact_pos, robot_body)
        robot_velocity = jacp_robot @ qvel
        velocity_leg = jacp_robot @ qvel_leg
        velocity_wheel = jacp_robot @ qvel_wheel
        velocity_base = jacp_robot @ qvel_base

        mujoco.mj_jac(model, data, jacp_terrain, None, contact_pos, terrain_body)
        terrain_velocity = jacp_terrain @ qvel

        relative_velocity = robot_velocity - terrain_velocity
        tangential_relative = (
            relative_velocity - float(np.dot(relative_velocity, normal_world)) * normal_world
        )
        residual = robot_velocity - (velocity_leg + velocity_wheel + velocity_base)

        wheel_index = int(wheel_index)
        wheel_forces[wheel_index] += force_world
        contact_counts[wheel_index] += 1
        total_force += force_world
        total_moment += np.cross(contact_pos - reference, force_world)
        total_moment += torque_world

        contacts.append(
            {
                "contact_index": int(contact_id),
                "wheel_index": wheel_index,
                "robot_body_id": robot_body,
                "terrain_body_id": terrain_body,
                "robot_geom_id": int(robot_geom),
                "terrain_geom_id": int(terrain_geom),
                "pos_world_m": [float(v) for v in contact_pos],
                "normal_world": [float(v) for v in normal_world],
                "force_world_n": [float(v) for v in force_world],
                "torque_world_nm": [float(v) for v in torque_world],
                "normal_force_n": float(local_wrench[0]),
                "tangential_force_magnitude_n": float(np.linalg.norm(local_wrench[1:3])),
                "robot_point_velocity_world_mps": [float(v) for v in robot_velocity],
                "terrain_point_velocity_world_mps": [float(v) for v in terrain_velocity],
                "relative_velocity_world_mps": [float(v) for v in relative_velocity],
                "tangential_relative_velocity_world_mps": [
                    float(v) for v in tangential_relative
                ],
                "tangential_relative_speed_mps": float(np.linalg.norm(tangential_relative)),
                "leg_dof_velocity_world_mps": [float(v) for v in velocity_leg],
                "leg_dof_tangent_speed_mps": _tangent_speed(velocity_leg, normal_world),
                "wheel_dof_velocity_world_mps": [float(v) for v in velocity_wheel],
                "wheel_dof_tangent_speed_mps": _tangent_speed(velocity_wheel, normal_world),
                "free_base_velocity_world_mps": [float(v) for v in velocity_base],
                "free_base_tangent_speed_mps": _tangent_speed(velocity_base, normal_world),
                "robot_point_tangent_speed_mps": _tangent_speed(robot_velocity, normal_world),
                "decomposition_residual_norm_mps": float(np.linalg.norm(residual)),
            }
        )

    return {
        "schema": TURN_CONTACT_SCHEMA,
        "measurement_time_s": float(data.time),
        "frame": "world",
        "sampling": "synchronized_endpoint_not_control_average",
        "reference_world_m": [float(v) for v in reference],
        "contacts": contacts,
        "contact_count_by_wheel": [int(c) for c in contact_counts],
        "wheel_force_world_n": [[float(v) for v in row] for row in wheel_forces],
        "total_wrench_world_6": [float(v) for v in np.concatenate((total_force, total_moment))],
    }
