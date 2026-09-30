"""One fresh real plane/15mm-box model; unchanged inherited D1 dynamics."""
from __future__ import annotations
from scripts.d1_single_step_plant import D1SingleStepPlant
from course_ground_08 import CourseGroundMap
from wheel_legged_control.d1.training_terrain import TrainingGroundReference
from geometry24 import validate_geometry24


class SingleStepPlant24(D1SingleStepPlant):
    def __init__(self, *, actuator_channel):
        super().__init__(obstacle_enabled=True, control_dt=.01,
                         actuator_channel=actuator_channel)
        self._ground_rows = []
        for gid in sorted(self.terrain_geom_ids):
            name = 'floor' if gid == self.floor_geom_id else 'terrain_single_15mm_box'
            self._ground_rows.append(dict(geom_id=int(gid), name=name, body_id=0,
                type='plane' if gid == self.floor_geom_id else 'box', collision=True,
                position_m=self.model.geom_pos[gid].tolist(),
                size_m=self.model.geom_size[gid].tolist(),
                quaternion_wxyz=self.model.geom_quat[gid].tolist(),
                contype=int(self.model.geom_contype[gid]),
                conaffinity=int(self.model.geom_conaffinity[gid]),
                margin_m=float(self.model.geom_margin[gid])))
        self._ground_map = CourseGroundMap(self._ground_rows)
        validate_geometry24(self.collision_terrain_metadata)

    @property
    def ground_map(self):
        return self._ground_map

    def course_ground_hit(self, x, y):
        return self._ground_map.query(x, y)

    def locomotion_ground_reference(self, x, y):
        hit = self.course_ground_hit(x, y)
        return TrainingGroundReference(hit.height_m, hit.pitch_rad, hit.roll_rad)

    @property
    def collision_terrain_metadata(self):
        self.validate_single_step()
        return dict(schema='d1-actual-plane-single15mm-oracle-24-v1', arena='single_step',
            world_collision_geoms=self._ground_rows,
            joint_qpos_addresses=self.qpos_addresses.tolist(),
            joint_dof_addresses=self.dof_addresses.tolist(),
            actuator_ids=self.actuator_ids.tolist(),
            model_nq=int(self.model.nq), model_nv=int(self.model.nv),
            model_nu=int(self.model.nu), control_dt_s=.01, native_dt_s=.002,
            ground_query='highest_vertical_hit_of_compiled_world_plane_or_box',
            ground_source='oracle_compiled_collision_geometry_not_runtime_sensor')
