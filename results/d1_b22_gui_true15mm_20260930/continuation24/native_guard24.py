"""Exact plane + 15 mm box binding; inherited native sampling/archival unchanged."""
from __future__ import annotations

import numpy as np
from archive13.atomic_archive_13 import AtomicCourseNativeGuard


class SingleStepNativeGuard24(AtomicCourseNativeGuard):
    """Changes only terrain admission, never dynamics or force computation."""

    def bind(self, plant):
        if self.plant is not None or self.attempted:
            raise RuntimeError('single-step native guard may bind only once')
        rows = tuple(plant.collision_terrain_metadata['world_collision_geoms'])
        expected_names = {'floor', 'terrain_single_15mm_box'}
        if len(rows) != 2 or {row['name'] for row in rows} != expected_names:
            raise ValueError('C24 requires exactly the actual plane and one 15 mm box')
        model = plant.model
        terrain = {}
        for row in rows:
            gid, name = row['geom_id'], row['name']
            if type(gid) is not int or not 0 <= gid < int(model.ngeom) or gid in terrain:
                raise ValueError('invalid/duplicate compiled terrain ID')
            family = 'floor' if name == 'floor' else 'step'
            if (int(model.geom_bodyid[gid]) != 0 or row['body_id'] != 0
                    or row['collision'] is not True
                    or row['type'] != ('plane' if family == 'floor' else 'box')
                    or int(model.geom_type[gid]) != (0 if family == 'floor' else 6)
                    or not (model.geom_contype[gid] or model.geom_conaffinity[gid])):
                raise ValueError('terrain metadata differs from actual native collision geom')
            for actual, saved in ((model.geom_pos[gid],row['position_m']),
                                  (model.geom_size[gid],row['size_m']),
                                  (model.geom_quat[gid],row['quaternion_wxyz'])):
                if not np.array_equal(np.asarray(actual),np.asarray(saved)):
                    raise ValueError('compiled geom changed since terrain binding')
            if family == 'step' and not (
                    np.allclose(model.geom_pos[gid],(-3.1,0.,.0075),rtol=0,atol=1e-14)
                    and np.array_equal(model.geom_size[gid],np.array((.18,.62,.0075)))
                    and np.array_equal(model.geom_quat[gid],np.array((1.,0.,0.,0.)))):
                raise ValueError('C24 actual 15 mm box dimensions/orientation differ')
            if family == 'floor' and not (
                    np.array_equal(model.geom_pos[gid],np.zeros(3))
                    and np.array_equal(model.geom_quat[gid],np.array((1.,0.,0.,0.)))):
                raise ValueError('C24 floor must be horizontal at world z=0')
            terrain[gid] = (name,family)
        actual_world = {gid for gid in range(int(model.ngeom))
                        if int(model.geom_bodyid[gid]) == 0
                        and (model.geom_contype[gid] or model.geom_conaffinity[gid])}
        if set(terrain) != actual_world or set(terrain) != set(plant.terrain_geom_ids):
            raise ValueError('hidden/unbound native world collision geometry')
        if (type(plant.physics_steps) is not int or plant.physics_steps != 5
                or float(plant.control_dt) != .01 or float(model.opt.timestep) != .002):
            raise ValueError('C24 native cadence differs')
        self._identity = (int(model._address),int(plant.data._address))
        self._terrain = terrain
        self.plant = plant
