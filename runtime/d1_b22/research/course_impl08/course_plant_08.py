"""Fresh synchronized D1 course plant with a compiled-geometry ground oracle.

Construction is deliberately explicit and must occur only inside a root-owned
counted model phase. Importing this module does not construct or step anything.
The underlying D1Plant owns the original robot, actuators, data, reset and
five-substep control integration; this class only validates the course world
geoms and exposes a pure query over their compiled primitive attributes.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from course_ground_08 import CourseGroundHit, CourseGroundMap

from wheel_legged_control.d1.model import ActuatorChannel, D1Plant
from wheel_legged_control.d1.training_terrain import TrainingGroundReference

COURSE_PLANT_SCHEMA = "d1-synchronized-compiled-course-plant-v1"


class CoursePlant(D1Plant):
    """One actual course model/data pair, never a substituted SingleStep model."""

    def __init__(self, *, control_dt: float = 0.01,
                 actuator_channel: ActuatorChannel | None = None) -> None:
        super().__init__(
            control_dt=control_dt, arena="course", sampling_mode="synchronized",
            actuator_channel=actuator_channel,
        )
        if (self.control_dt != 0.01 or self.physics_steps != 5
                or float(self.model.opt.timestep) != 0.002):
            raise RuntimeError("course integration timing differs from the D1 contract")
        self._ground_rows = self._compiled_ground_rows()
        self._ground_map = CourseGroundMap(self._ground_rows)
        if set(self._ground_map.geom_ids) != set(self.terrain_geom_ids):
            raise RuntimeError("course world collision identities differ from plant terrain set")

    def _compiled_ground_rows(self) -> tuple[dict[str, Any], ...]:
        model = self.model
        rows = []
        for gid in range(model.ngeom):
            body_id = int(model.geom_bodyid[gid])
            colliding = bool(model.geom_contype[gid] or model.geom_conaffinity[gid])
            if body_id != 0 or not colliding:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if not name or not (name == "floor" or name.startswith("terrain_")):
                raise RuntimeError("unexpected colliding world geom in course model")
            geom_type = int(model.geom_type[gid])
            kind = (
                "plane" if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE) else
                "box" if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX) else None
            )
            if kind is None:
                raise RuntimeError("course contains a nonprimitive world collision geom")
            rows.append({
                "geom_id": gid, "name": name, "body_id": body_id,
                "type": kind, "collision": colliding,
                "position_m": model.geom_pos[gid].tolist(),
                "size_m": model.geom_size[gid].tolist(),
                "quaternion_wxyz": model.geom_quat[gid].tolist(),
                "contype": int(model.geom_contype[gid]),
                "conaffinity": int(model.geom_conaffinity[gid]),
                "margin_m": float(model.geom_margin[gid]),
            })
        if len(rows) != 92:  # floor + 63 rough + 3 ramp + 9 stairs + 13 bumps + 3 jump
            raise RuntimeError("compiled course terrain geom count differs from source layout")
        return tuple(rows)

    def course_ground_hit(self, x: float, y: float) -> CourseGroundHit:
        return self._ground_map.query(x, y)

    @property
    def ground_map(self) -> CourseGroundMap:
        """The compiled primitive map consumed by the shared 99D encoder."""
        return self._ground_map

    def locomotion_ground_reference(self, x: float, y: float) -> TrainingGroundReference:
        hit = self.course_ground_hit(x, y)
        return TrainingGroundReference(hit.height_m, hit.pitch_rad, hit.roll_rad)

    @property
    def collision_terrain_metadata(self) -> dict[str, Any]:
        return {
            "schema": COURSE_PLANT_SCHEMA, "arena": "course",
            "world_collision_geoms": list(self._ground_rows),
            "joint_qpos_addresses": np.asarray(self.qpos_addresses).tolist(),
            "joint_dof_addresses": np.asarray(self.dof_addresses).tolist(),
            "actuator_ids": np.asarray(self.actuator_ids).tolist(),
            "model_nq": int(self.model.nq), "model_nv": int(self.model.nv),
            "model_nu": int(self.model.nu),
            "control_dt_s": self.control_dt,
            "native_dt_s": float(self.model.opt.timestep),
            "ground_query": "highest_vertical_hit_of_compiled_world_plane_or_box",
            "ground_source": "oracle_compiled_collision_geometry_not_runtime_sensor",
        }
