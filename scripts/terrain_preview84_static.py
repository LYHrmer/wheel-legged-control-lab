"""Exercise the actual eight-ray sensor without integrating dynamics or training."""

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from wheel_legged_control.d1.terrain_preview import (
    PreviewChannel,
    RangeFrame,
    query_frame,
    read_ranges,
)


def capture(grade_deg, pitch_deg=0.0, roll_deg=0.0, *, occlude=False, unseen=False):
    # This test scene provides independent rigid geometry with a flat region under the rig.
    # Its body pose is physically the same across layouts. The controller receives ranges.
    alpha = math.radians(grade_deg)
    start, half, thickness = 0.30, 1.50, 0.05
    cx = start + half * math.cos(alpha) + thickness * math.sin(alpha)
    cz = half * math.sin(alpha) - thickness * math.cos(alpha)
    obstruction = '<geom type="box" pos="0.225 0 0.5" size="0.006 0.5 0.45"/>' if occlude else ''
    distant = '<geom type="box" pos="5 0 1" size="0.5 1 1"/>' if unseen else ''
    pitch, roll = math.radians(pitch_deg), math.radians(roll_deg)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    quat = f"{cp * cr} {cp * sr} {sp * cr} {-sp * sr}"
    xml = f'''<mujoco><compiler angle="radian"/><worldbody>
      <geom type="box" pos="-1.7 0 -0.05" size="2 1 0.05"/>
      <geom type="box" pos="{cx} 0 {cz}" size="{half} 1 {thickness}"
            quat="{math.cos(alpha / 2)} 0 {-math.sin(alpha / 2)} 0"/>
      {obstruction}{distant}<body name="sensor_mount" pos="0 0 0.455" quat="{quat}"/>
    </worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    body = model.body("sensor_mount").id
    mujoco.mj_forward(model, data)
    raw = read_ranges(model, data, body)
    return RangeFrame.from_ranges(0, 0.0, 0.04, 0.0, raw, pitch, roll)


def inspect(frame, now=0.06):
    reading = query_frame(frame, now, 0.0, 0.24, 0.35)
    delta = PreviewChannel().correction(
        reading, "on", support_pitch_rad=0.0, baseline_force_n=0.0,
        pitch_gain_n_per_rad=-93618.0, force_limit_n=180.0, dt_s=0.01,
    )
    return reading, delta


def check():
    geometries = []
    for pitch, roll in ((0, 0), (-2, 2), (2, -2)):
        for slope in (0, 4, -4):
            frame = capture(slope, pitch, roll)
            reading, delta = inspect(frame)
            estimate = math.degrees(reading.slope_rad) if reading.valid else None
            passed = estimate is not None and abs(estimate - slope) <= 1.0
            if slope:
                passed = passed and delta * slope < 0
            geometries.append({"grade_deg": slope, "body_pitch_deg": pitch, "body_roll_deg": roll,
                               "estimate_deg": estimate, "command_delta_rad": delta,
                               "reading": reading.as_dict(), "pass": passed})
    frame = capture(4)
    faults = {}
    for name, sample, now in (("occluded", capture(4, occlude=True), .06),
                              ("expired", frame, .26), ("future", frame, -.01)):
        reading, delta = inspect(sample, now)
        faults[name] = {"reason": reading.reason, "command_delta_rad": delta,
                        "pass": not reading.valid and delta == 0.0}
    distant = capture(4, unseen=True)
    unchanged = (np.array_equal(frame.ranges, distant.ranges, equal_nan=True)
                 and inspect(frame)[1] == inspect(distant)[1])
    return {"schema": "terrain-preview84-static-v1", "integration_steps": 0,
            "geometries": geometries, "faults": faults, "unseen_invariant": unchanged,
            "passed": all(r["pass"] for r in geometries) and all(v["pass"] for v in faults.values())
            and unchanged}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Reserve the result before any simulation call. Partial output remains exclusive.
    with args.output.open("x", encoding="utf-8") as out:
        result = check()
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    raise SystemExit(0 if result["passed"] else 2)
