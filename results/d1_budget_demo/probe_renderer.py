"""Time five recorded states without running physics or policy inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir()
start = time.monotonic()
model = mujoco.MjModel.from_binary_path(str(args.run / "model.mjb"))
model.vis.global_.offwidth, model.vis.global_.offheight = 960, 540
data, camera = mujoco.MjData(model), mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_FREE
camera.distance, camera.azimuth, camera.elevation = 2.7, 130.0, -25.0
records = np.load(args.run / "states.npz")
indices = np.linspace(0, len(records["qpos"]) - 1, 5, dtype=int).tolist()
durations = []
with mujoco.Renderer(model, height=540, width=960) as renderer:
    ready = time.monotonic()
    for index in indices:
        tick = time.monotonic()
        data.qpos[:], data.qvel[:] = records["qpos"][index], records["qvel"][index]
        mujoco.mj_forward(model, data)
        camera.lookat[:] = data.qpos[:3] + (0.2, 0, -0.1)
        renderer.update_scene(data, camera=camera)
        picture = renderer.render()
        durations.append(time.monotonic() - tick)
        Image.fromarray(picture).save(args.output / f"state{index}.png")
report = {"physics_steps": 0, "state_indices": indices,
    "setup_seconds": ready - start, "frame_seconds": durations,
    "wall_seconds": time.monotonic() - start,
    "LP_NUM_THREADS": os.environ.get("LP_NUM_THREADS"),
    "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with (args.output / "timing.json").open("x") as stream:
    json.dump(report, stream, indent=2)
    stream.write("\n")
print(json.dumps(report, indent=2))
