"""Development jump measurement; uses the unchanged course controller."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from scripts.d1_side_step import SideStepController
from wheel_legged_control.d1.interactive import D1InteractiveSimulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sim = D1InteractiveSimulation(arena='flat')
    geom = SideStepController(sim.plant)
    data = mujoco.MjData(sim.plant.model)
    rows = []
    for tick in range(600):
        if tick == 200:
            sim.teleop.handle_key(32)
        status = sim.step()
        mujoco.mj_copyData(data, sim.plant.model, sim.plant.data)
        mujoco.mj_forward(sim.plant.model, data)
        points, extent = geom._geometry(data)
        bottoms = data.xpos[geom.bodies, 2] - extent
        # Exact vertical cylinder support extent, distinct from body rise.
        row = dict(time_s=float(data.time), phase=status.jump_phase,
                   height_m=float(data.qpos[2]), vertical_speed_mps=float(data.qvel[2]),
                   wheel_bottoms_m=bottoms.tolist(),
                   requested_vertical_force_n=status.vertical_feedforward_force_n,
                   upward_support_n=sim.teleop.controller.low_level.last_breakdown.support_force_n.tolist(),
                   qpos=data.qpos.tolist(), qvel=data.qvel.tolist(),
                   torque=sim.teleop._last_torque.tolist(),
                   roll_pitch=sim.plant.base_rpy[:2].tolist(),
                   nonwheel_contacts=int(sim.plant.undesired_ground_contacts))
        rows.append(row)
    peak = max(rows[200:400], key=lambda r: min(r['wheel_bottoms_m']))
    summary = dict(
        controller='unchanged D1InteractiveSimulation flat legacy LQR/VMC',
        jump_request_time_s=2.,
        max_all_wheel_clearance_m=min(peak['wheel_bottoms_m']),
        clearance_peak_time_s=peak['time_s'],
        max_base_rise_m=max(r['height_m'] for r in rows[200:400])-rows[199]['height_m'],
        final_roll_pitch_rad=rows[-1]['roll_pitch'],
        max_tilt_rad=max(max(abs(v) for v in r['roll_pitch']) for r in rows),
        nonwheel_contact_steps=sum(r['nonwheel_contacts'] > 0 for r in rows),
        total_physical_transitions=len(rows),
    )
    (args.output/'trace.json').write_text(json.dumps(rows, allow_nan=False)+'\n')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
