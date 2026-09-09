"""Render the complete recorded D1 simulation, never substituting a baseline."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compiled_model_bytes(directory: Path, manifest: dict) -> bytes:
    artifact = manifest.get("compiled_model")
    if artifact is None:
        raw = (directory / "model.mjb").read_bytes()
        expected = manifest["sha256"]["model.mjb"]
    else:
        expected = artifact["uncompressed_sha256"]
        if artifact["path"] != f"../models/{expected}.mjb.gz":
            raise ValueError("unexpected shared model location")
        compressed = (directory / artifact["path"]).read_bytes()
        if hashlib.sha256(compressed).hexdigest() != artifact["sha256"]:
            raise ValueError("compressed model hash mismatch")
        raw = gzip.decompress(compressed)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("decompressed model hash mismatch")
    return raw


def read_recording(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text())
    required = {"protocol.json", "summary.json", "states.npz", "telemetry.csv"}
    if "compiled_model" not in manifest:
        required.add("model.mjb")
    if not required.issubset(manifest["sha256"]):
        raise ValueError("recording manifest omits required artifacts")
    for name in required:
        if sha256(directory / name) != manifest["sha256"][name]:
            raise ValueError(f"recording hash mismatch: {name}")
    protocol = json.loads((directory / "protocol.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    with (directory / "telemetry.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with np.load(directory / "states.npz", allow_pickle=False) as data:
        states = {key: data[key].copy() for key in ("time_s", "qpos", "qvel")}
    if not rows or len(rows) + 1 != len(states["time_s"]):
        raise ValueError("recording state/telemetry lengths differ")
    if [int(row["step"]) for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("recorded steps must be complete, contiguous and ordered")
    if any(not np.isfinite(array).all() for array in states.values()):
        raise ValueError("nonfinite recording cannot be rendered honestly")
    expected = np.arange(len(rows) + 1) * protocol["control_dt_s"]
    if not np.allclose(states["time_s"], expected, atol=1e-10, rtol=0):
        raise ValueError("recording timestamps omit or repeat simulation states")
    if not np.allclose([float(row["time_s"]) for row in rows], expected[1:]):
        raise ValueError("telemetry and state timestamps differ")
    if (
        protocol["compiled_model_sha256"]
        != hashlib.sha256(compiled_model_bytes(directory, manifest)).hexdigest()
    ):
        raise ValueError("compiled model differs from protocol")
    if protocol["mode"] not in ("zero", "policy"):
        raise ValueError("unknown rollout mode")
    if (protocol["mode"] == "policy") != bool(protocol["checkpoint_sha256"]):
        raise ValueError("policy labeling requires a recorded checkpoint hash")
    return protocol, summary, rows, states, manifest


def frame_indices(nsteps: int, dt: float, fps: int) -> list[int]:
    stride = round(1 / (dt * fps))
    if stride < 1 or not np.isclose(stride * dt * fps, 1):
        raise ValueError("fps must sample an integral number of control steps")
    indices = list(range(0, nsteps + 1, stride))
    if indices[-1] != nsteps:
        indices.append(nsteps)
    return indices


def render(directory: Path, output: Path, *, fps: int = 20) -> dict:
    sidecar, cover = output.with_suffix(".json"), output.with_suffix(".png")
    if output.suffix != ".mp4" or any(path.exists() for path in (output, sidecar, cover)):
        raise ValueError("output must be a new .mp4, with new .json and .png companions")
    protocol, summary, rows, states, manifest = read_recording(directory)
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    with TemporaryDirectory(prefix="d1-continuous-render-") as temporary:
        path = Path(temporary) / "model.mjb"
        path.write_bytes(compiled_model_bytes(directory, manifest))
        model = mujoco.MjModel.from_binary_path(str(path))
    data = mujoco.MjData(model)
    if states["qpos"].shape != (len(rows) + 1, model.nq):
        raise ValueError("recorded qpos shape differs from archived compiled model")
    if states["qvel"].shape != (len(rows) + 1, model.nv):
        raise ValueError("recorded qvel shape differs from archived compiled model")
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor < 0:
        raise ValueError("compiled model has no floor")
    model.geom_matid[floor] = -1
    model.geom_rgba[floor] = (0.69, 0.73, 0.76, 1)
    model.light_ambient[:] = 0.25
    model.light_diffuse[:] = 0.55
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.45
    model.vis.headlight.specular[:] = 0.05
    width, height = 960, 540
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = 2.7, 130.0, -25.0
    font = ImageFont.truetype("DejaVuSans.ttf", 16)
    heading = ImageFont.truetype("DejaVuSans.ttf", 20)
    indices = frame_indices(len(rows), protocol["control_dt_s"], fps)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    checkpoint = protocol["checkpoint_sha256"]
    mode = "PPO recorded rollout" if checkpoint else "ZERO residual baseline (no PPO)"
    model_id = (checkpoint or protocol["compiled_model_sha256"])[:12]
    # This inset is sampled from the archived collision surface, not an
    # illustrative drawing with exaggerated obstacle parameters.
    hfield = int(model.geom_dataid[floor])
    address = int(model.hfield_adr[hfield])
    ncol = int(model.hfield_ncol[hfield])
    road_x = np.linspace(-model.hfield_size[hfield, 0], model.hfield_size[hfield, 0], ncol)
    road_z = model.hfield_data[address : address + ncol] * model.hfield_size[hfield, 2]
    road_z += model.geom_pos[floor, 2]
    renderer = None
    try:
        renderer = mujoco.Renderer(model, width=width, height=height)
        for index in indices:
            row = rows[max(0, index - 1)]
            data.qpos[:] = states["qpos"][index]
            data.qvel[:] = states["qvel"][index]
            data.time = float(states["time_s"][index])
            mujoco.mj_forward(model, data)  # Geometry only; no physics stepping.
            camera.lookat[:] = data.qpos[:3] + np.asarray((0.2, 0.0, -0.10))
            renderer.update_scene(data, camera=camera)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, width, 168), fill=(20, 30, 43))
            draw.text((14, 7), f"D1 | {mode} | model SHA {model_id}", font=heading, fill="white")
            draw.text(
                (14, 35),
                f"t={data.time:5.2f}/{protocol['config']['duration_s']:.0f}s | "
                f"{row['stage']} | terrain: {row['terrain_section']}",
                font=font,
                fill="white",
            )
            draw.text(
                (14, 58),
                f"cmd v={float(row['command_velocity_mps']):+.2f} m/s "
                f"yaw={float(row['command_yaw_rate_rps']):+.2f} rad/s | "
                f"truth vx={float(row['velocity_mps']):+.2f} "
                f"vy={float(row['lateral_velocity_mps']):+.2f} "
                f"yaw={float(row['yaw_rate_rps']):+.2f}",
                font=font,
                fill="white",
            )
            draw.text(
                (14, 80),
                f"residual Fx={float(row['residual_longitudinal_n']):+.2f} N "
                f"Fz={float(row['residual_vertical_n']):+.2f} N | "
                f"clearance={float(row['clearance_m']):.3f} m | "
                f"{protocol['config']['ground_reference_mode']} ground",
                font=font,
                fill="white",
            )
            draw.text(
                (14, 106),
                protocol.get("observation_schema", "unrecorded observation schema"),
                font=font,
                fill="white",
            )
            draw.text(
                (14, 126),
                protocol.get("control_schema", "unrecorded control schema"),
                font=font,
                fill="white",
            )
            draw.text(
                (14, 147),
                f"Truth position: cached {model.opt.timestep * 1000:g} ms before displayed qpos",
                font=font,
                fill=(208, 218, 227),
            )
            draw.rectangle((0, height - 80, width, height), fill=(20, 30, 43))
            draw.text(
                (14, height - 76),
                "Static road: A=5 mm, wavelength=0.8 m, ramps +/-4 deg | "
                "profile below uses a separate vertical scale (metres)",
                font=font,
                fill="white",
            )
            pixels = [
                (int(80 + (x + 6) / 12 * 720), int(height - 22 - z * 400))
                for x, z in zip(road_x, road_z, strict=True)
            ]
            draw.line(pixels, fill=(115, 201, 235), width=2)
            marker = int(80 + (float(data.qpos[0]) + 6) / 12 * 720)
            draw.line((marker, height - 55, marker, height - 15), fill=(245, 191, 71), width=2)
            draw.text((12, height - 26), "0.00", font=font, fill="white")
            draw.text((12, height - 58), "0.08", font=font, fill="white")
            draw.text((810, height - 40), f"x={data.qpos[0]:+.2f} m", font=font, fill="white")
            if index == len(rows):
                outcome = "COMPLETE" if summary["completed"] else "FAILED / PARTIAL"
                draw.text(
                    (14, 174),
                    f"{outcome} | quality_pass={summary['quality_pass']}",
                    font=heading,
                    fill=(185, 55, 30),
                )
            if index == indices[len(indices) // 2]:
                frame.save(cover)
            encoder.stdin.write(np.asarray(frame).tobytes())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("ffmpeg failed; partial output is retained for diagnosis")
    finally:
        if renderer is not None:
            renderer.close()
        if encoder.poll() is None:
            encoder.stdin.close()
            encoder.terminate()
            encoder.wait()
    report = {
        "schema": "d1-continuous-recorded-video-v1",
        "input_manifest_sha256": sha256(directory / "manifest.json"),
        "input_sha256": manifest["sha256"],
        "compiled_model": manifest.get("compiled_model"),
        "renderer_sha256": sha256(Path(__file__)),
        "checkpoint_sha256": checkpoint,
        "observation_schema": protocol.get("observation_schema"),
        "control_schema": protocol.get("control_schema"),
        "mode": protocol["mode"],
        "completed": summary["completed"],
        "quality_pass": summary["quality_pass"],
        "frames": len(indices),
        "fps": fps,
        "simulation_duration_s": float(states["time_s"][-1]),
        "playback_duration_s": len(indices) / fps,
        "frame_state_indices": indices,
        "video_sha256": sha256(output),
        "cover_sha256": sha256(cover),
        "scene": "archived compiled model; only lighting/material/camera changed",
        "state_source": "actual complete recorded simulation; initial, periodic and terminal states",
        "physics_steps_in_renderer": 0,
        "physics_timestep_s": float(model.opt.timestep),
        "truth_position_phase": "cached final-substep kinematics; qpos recording is post-integration",
    }
    with sidecar.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    arguments = parser.parse_args()
    print(json.dumps(render(arguments.run, arguments.output, fps=arguments.fps), indent=2))
