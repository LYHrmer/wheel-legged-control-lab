"""Compare scene-only grid annotations to MuJoCo rays on the real D1 terrain."""
import json
import hashlib
from pathlib import Path
import time

import mujoco
import numpy as np

from scripts.d1_terrain_display import append_terrain_grid, grid_segments
from scripts import d1_terrain_display
from wheel_legged_control.d1.locomotion_terrain import add_locomotion_terrain, locomotion_terrain_configs

spec = mujoco.MjSpec.from_string('<mujoco><worldbody><geom name="floor" type="plane" size="0 0 .1"/></worldbody></mujoco>')
config = locomotion_terrain_configs('development')[0]
add_locomotion_terrain(spec, config)
model = spec.compile()
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
hid = model.geom_dataid[gid]
rows, cols = int(model.hfield_nrow[hid]), int(model.hfield_ncol[hid])
address = int(model.hfield_adr[hid])
heights = model.hfield_data[address:address + rows * cols].reshape(rows, cols)
before = {name: getattr(model, name).copy() for name in ('hfield_data', 'geom_pos', 'geom_quat')}
q_before, v_before = data.qpos.copy(), data.qvel.copy()
segments = grid_segments(heights, model.hfield_size[hid, :2], model.hfield_size[hid, 2], model.geom_pos[gid], (2., .25))
errors = []
for segment in segments:
    for point in (segment[0], segment.mean(axis=0), segment[1]):
        distance = mujoco.mj_rayHfield(model, data, gid, np.asarray((point[0], point[1], 2.)), np.asarray((0., 0., -1.)), None)
        if distance < 0:
            raise AssertionError('grid annotation falls outside collision surface')
        errors.append(abs(float(point[2]) - .002 - (2. - distance)))
assert max(errors, default=0.) < 1e-7
scene = mujoco.MjvScene(model, maxgeom=2000)
mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scene)
native = scene.ngeom
start = time.perf_counter()
for _ in range(20):
    scene.ngeom = native
    count = append_terrain_grid(model, scene, (2., .25))
elapsed = time.perf_counter() - start
assert scene.ngeom == native + count and count <= 1500
assert all(np.array_equal(value, getattr(model, name)) for name, value in before.items())
assert np.array_equal(q_before, data.qpos) and np.array_equal(v_before, data.qvel)
report = {'status': 'passed', 'hfield_shape': [rows, cols], 'segments': len(segments),
          'ray_checks': len(errors), 'max_height_error_m': max(errors, default=0.),
          'append_mean_ms': elapsed / 20 * 1000, 'model_and_physics_data_unchanged': True,
          'display_source_sha256': hashlib.sha256(Path(d1_terrain_display.__file__).read_bytes()).hexdigest(),
          'scope': 'Real compiled D1 terrain and MuJoCo rays; no graphical window or physics step.'}
out = Path(__file__).parent / 'real_heightfield_audit.json'
out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
