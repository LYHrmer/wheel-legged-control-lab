"""Render recorded mobility qpos/qvel as a GIF, without stepping or controlling.

The default offscreen backend is EGL; if unavailable, retry with MUJOCO_GL=osmesa.
No dependencies or drivers are installed. The scene is rebuilt from this checkout
using the recorded arena; the rollout does not archive the original model bytes.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _read(directory: Path, scenario: str, seed: int) -> tuple[dict, dict, list[dict], dict]:
    summary_path, csv_path = directory / "summary.json", directory / "steps.csv.gz"
    summary_bytes, csv_bytes = summary_path.read_bytes(), csv_path.read_bytes()
    metadata = json.loads(summary_bytes)
    artifact = metadata["artifacts"]["steps"]
    csv_hash = hashlib.sha256(csv_bytes).hexdigest()
    if artifact["filename"] != csv_path.name or artifact["sha256"] != csv_hash:
        raise ValueError("recorded CSV filename/hash does not match summary.json")
    matches = [item for item in metadata["episodes"]
               if item["scenario"] == scenario and item["seed"] == seed]
    if len(matches) != 1:
        raise ValueError("scenario and seed must match exactly one recorded episode")
    with gzip.open(csv_path, "rt", encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["scenario"] == scenario and int(row["seed"]) == seed]
    if not rows or len(rows) != matches[0]["attempted_steps"]:
        raise ValueError("recorded rows are missing or disagree with episode step count")
    if [int(row["step"]) for row in rows] != list(range(len(rows))):
        raise ValueError("recorded steps must be contiguous and ordered; cannot trim missing steps")
    hashes = {"summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
              "steps_sha256": csv_hash}
    return metadata, matches[0], rows, hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new GIF and adjacent JSON")
    args = parser.parse_args()
    sidecar = args.output.with_suffix(".json")
    if args.output.suffix.lower() != ".gif" or args.output.exists() or sidecar.exists():
        parser.error("--output must name a new .gif; its adjacent .json must also be new")
    metadata, episode, rows, hashes = _read(args.rollout, args.scenario, args.seed)
    protocol = metadata["protocol"]
    if protocol.get("domain_randomization") is not False:
        parser.error("scene reconstruction currently requires domain_randomization=false")
    if not np.isclose(protocol["control_dt_s"], .01):
        parser.error("recording must use 0.01 s control steps for this 20 fps renderer")
    arena = protocol["arena_by_scenario"][args.scenario]
    planned_seconds = float(protocol["seconds_by_scenario"][args.scenario])
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from wheel_legged_control.d1.model import D1Plant
    from wheel_legged_control.provenance import capture_git_provenance

    plant = D1Plant(control_dt=.01, arena=arena)
    samples = [(rows[0], "input")]
    samples.extend((row, "output") for row in rows
                   if row["applied"].lower() == "true" and (int(row["step"]) + 1) % 5 == 0)
    final_prefix = "output" if rows[-1]["applied"].lower() == "true" else "input"
    if samples[-1][0] is not rows[-1] or samples[-1][1] != final_prefix:
        samples.append((rows[-1], final_prefix))
    applied_count = sum(row["applied"].lower() == "true" for row in rows)
    last_time = float(rows[-1][f"{final_prefix}_time_s"])
    passed = bool(
        episode.get("passed") is True and episode.get("completed") is True
        and applied_count == episode["applied_steps"] == round(planned_seconds / .01)
        and np.isclose(last_time, planned_seconds)
        and all(row["applied"].lower() == "true" and row["status"] == "solved" for row in rows)
    )
    outcome = "PASSED" if passed else "FAILED / PARTIAL"
    width, height = 640, 360
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
        heading_font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = heading_font = ImageFont.load_default()
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    turning = args.scenario.startswith("turn_")
    camera.distance = 2.8 if turning else 3.4
    camera.azimuth = 135.
    camera.elevation = -45. if turning else -24.
    frames, frame_times, trail = [], [], []
    renderer = None
    try:
        renderer = mujoco.Renderer(plant.model, width=width, height=height)
        for row, prefix in samples:
            qpos = np.asarray([float(row[f"{prefix}_qpos_{i}"]) for i in range(plant.model.nq)])
            qvel = np.asarray([float(row[f"{prefix}_qvel_{i}"]) for i in range(plant.model.nv)])
            if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
                raise ValueError("recorded state is nonfinite; cannot honestly render this frame")
            plant.data.qpos[:], plant.data.qvel[:] = qpos, qvel
            plant.data.time = float(row[f"{prefix}_time_s"])
            # Recompute geometry only; never call plant.step, mj_step or a controller.
            mujoco.mj_forward(plant.model, plant.data)
            camera.lookat[:] = plant.base_position + np.asarray((.0 if turning else .35, 0., -.12))
            renderer.update_scene(plant.data, camera=camera)
            if turning:
                trail.append(np.r_[qpos[:2], .015])
                for start, end in pairwise(trail):
                    if np.linalg.norm(end - start) < 1e-5:
                        continue
                    scene = renderer.scene
                    if scene.ngeom >= scene.maxgeom:
                        raise RuntimeError("render scene lacks space for the recorded trajectory")
                    geom = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                                       np.zeros(3), np.eye(3).ravel(), np.asarray((.95, .65, .15, 1.)))
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .004, start, end)
                    scene.ngeom += 1
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, width, 65), fill=(19, 29, 42))
            draw.rectangle((0, height - 23, width, height), fill=(19, 29, 42))
            draw.text((9, 5), f"D1 | {args.scenario} | seed {args.seed} | {outcome}",
                      font=heading_font, fill=(155, 228, 189) if passed else (255, 168, 143))
            draw.text((9, 27), f"t={plant.data.time:5.2f}s  "
                      f"cmd v={float(row['command_forward_velocity_mps']):+.2f}m/s  "
                      f"yaw rate={float(row['command_yaw_rate_rps']):+.2f}rad/s  QP={row['status']}",
                      font=font, fill="white")
            draw.text((9, 45), f"actual speed={np.linalg.norm(qvel[:2]):.3f}m/s  "
                      f"yaw={np.rad2deg(plant.base_rpy[2]):+.1f}deg  "
                      f"episode stop: {episode['stop_reason']}", font=font, fill="white")
            draw.text((9, height - 18), "Recorded simulation, not live control | "
                      f"warm_start={str(protocol['warm_start']).lower()} | {protocol['state_source']}",
                      font=font, fill="white")
            frames.append(frame)
            frame_times.append(float(plant.data.time))
    except Exception as error:
        raise RuntimeError(
            f"Rendering failed with MUJOCO_GL={os.environ.get('MUJOCO_GL')}: {error}. "
            "If EGL is unavailable, retry with MUJOCO_GL=osmesa; no dependencies were changed."
        ) from error
    finally:
        if renderer is not None:
            renderer.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        frames[0].save(stream, format="GIF", save_all=True, append_images=frames[1:],
                       duration=50, loop=0, optimize=False, disposal=2)
    report = {
        "rollout_directory": str(args.rollout.resolve()), **hashes,
        "rollout_source": metadata["source"], "scenario": args.scenario, "seed": args.seed,
        "arena": arena, "episode_passed": passed, "episode_stop_reason": episode["stop_reason"],
        "scene": {"factory": "D1Plant", "arena": arena, "domain_randomization": False},
        "renderer_source": capture_git_provenance(Path(__file__).resolve().parents[1]),
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "frames": len(frames), "width": width, "height": height, "frame_duration_ms": 50,
        "frame_simulation_times_s": frame_times, "playback_duration_s": len(frames) * .05,
        "simulation_duration_s": last_time - float(rows[0]["input_time_s"]),
        "planned_simulation_duration_s": planned_seconds,
        "sampling": "initial state, each fifth control endpoint, and final recorded state/event",
        "backend": os.environ["MUJOCO_GL"],
        "gif_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "notes": ["Recorded simulation, not live control or keyboard interaction.",
                  "Failure endpoints are retained; the final frame is held for another 50 ms.",
                  "Scene uses the current checkout; the rollout did not archive original model bytes."],
    }
    with sidecar.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "sidecar": str(sidecar),
                      "frames": len(frames), "episode_passed": passed}))


if __name__ == "__main__":
    main()
