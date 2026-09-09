"""Render archived synchronous D1 states; never simulate or infer a replacement.

Manifest/clock checks and offscreen drawing were drafted with Claude Opus.
Local integration corrects partial-run duration, torque field names and the
extra terminal frame's playback duration. Hashes establish consistency only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REQUIRED = {"protocol.json", "summary.json", "states.npz", "telemetry.csv", "model.mjb"}
WIDTH, HEIGHT = 960, 540


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def read_recording(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    _require(
        isinstance(manifest, dict) and REQUIRED <= manifest.keys(),
        "manifest omits required artifacts",
    )
    for name, expected in manifest.items():
        _require(
            isinstance(name, str)
            and name not in ("", ".", "..")
            and "/" not in name
            and "\\" not in name,
            "manifest names must be safe basenames",
        )
        path = directory / name
        _require(path.is_file() and not path.is_symlink(), f"missing or symlinked artifact: {name}")
        _require(
            isinstance(expected, str) and sha256(path) == expected,
            f"artifact hash mismatch: {name}",
        )
    protocol = json.loads((directory / "protocol.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    _require(protocol.get("schema") == "d1-locomotion-rollout-v1", "unknown recording schema")
    _require(
        protocol.get("compiled_model_sha256") == manifest["model.mjb"],
        "compiled model hash differs",
    )
    _require(summary.get("source_unchanged") is True, "recorded source changed during simulation")
    mode, checkpoint = protocol.get("policy"), protocol.get("model_sha256")
    _require(mode in ("zero_residual", "trusted_local_ppo"), "unknown recorded policy mode")
    _require(
        (mode == "zero_residual" and checkpoint is None)
        or (
            mode == "trusted_local_ppo"
            and isinstance(checkpoint, str)
            and len(checkpoint) == 64
            and all(c in "0123456789abcdef" for c in checkpoint)
        ),
        "policy/hash labeling mismatch",
    )
    episode = protocol["episode"]
    _require(
        all(isinstance(episode.get(k), str) and episode[k] for k in ("baseline", "source_schema")),
        "missing source/baseline",
    )
    _require(isinstance(protocol.get("observation_schema"), str), "missing observation schema")
    dt, duration = episode["control_dt_s"], episode["duration_s"]
    _require(
        not isinstance(dt, bool) and dt == 0.01 and np.isfinite(duration) and duration > 0,
        "invalid physical clock",
    )
    n = summary["steps"]
    _require(type(n) is int and n > 0, "recording must contain at least one completed interval")
    _require(type(summary.get("completed")) is bool, "completed must be a boolean")
    _require(
        n * dt <= duration + 1e-8 and np.isclose(summary["duration_s"], n * dt, atol=1e-9, rtol=0),
        "recorded duration mismatch",
    )
    if summary["completed"]:
        _require(
            np.isclose(n * dt, duration, atol=1e-9, rtol=0)
            and summary["stop_reason"] == "time_limit",
            "completion differs from time limit",
        )
    with np.load(directory / "states.npz", allow_pickle=False) as archive:
        states = {
            key: archive[key].copy() for key in ("qpos", "qvel", "time_s", "actuator_applied_nm")
        }
    for name, shape in (
        ("qpos", (n + 1, 23)),
        ("qvel", (n + 1, 22)),
        ("time_s", (n + 1,)),
        ("actuator_applied_nm", (n, 5, 16)),
    ):
        _require(
            states[name].shape == shape and np.isfinite(states[name]).all(),
            f"invalid recorded {name}",
        )
    _require(
        np.allclose(states["time_s"], np.arange(n + 1) * dt, atol=1e-9, rtol=0),
        "state clock mismatch",
    )
    with (directory / "telemetry.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    _require(len(rows) == n, "telemetry length mismatch")
    numeric = (
        "decision_time_s",
        "time_s",
        "command_vx_mps",
        "command_yaw_rps",
        "command_clearance_m",
        "velocity_mps",
        "yaw_rate_rps",
        "clearance_m",
        "x_m",
        "y_m",
        "z_m",
    )
    for index, row in enumerate(rows):
        _require(int(row["tick"]) == index, "telemetry ticks are not contiguous")
        _require(np.isfinite([float(row[k]) for k in numeric]).all(), "nonfinite telemetry")
        _require(
            np.isclose(float(row["decision_time_s"]), index * dt, atol=1e-9, rtol=0)
            and np.isclose(float(row["time_s"]), (index + 1) * dt, atol=1e-9, rtol=0),
            "telemetry clock mismatch",
        )
        _require(
            np.allclose(
                [float(row[k]) for k in ("x_m", "y_m", "z_m")],
                states["qpos"][index + 1, :3],
                atol=1e-9,
                rtol=0,
            ),
            "telemetry pose differs from synchronous state",
        )
    return protocol, summary, states, rows, manifest


def frame_indices(n, dt, fps):
    _require(
        type(n) is int and n > 0 and type(fps) is int and fps > 0,
        "positive integer steps/fps required",
    )
    stride = round(1 / (dt * fps))
    _require(
        stride >= 1 and np.isclose(stride * dt * fps, 1, atol=1e-12, rtol=0),
        "fps must use an integral state stride",
    )
    indices = list(range(0, n + 1, stride))
    if indices[-1] != n:
        indices.append(n)
    # Initial plus off-grid terminal can add almost two playback frames. Keep
    # the true terminal and report durations; never assert one-frame overhead.
    return indices


def render(directory, output, fps=20):
    directory, output = Path(directory), Path(output)
    sidecar, cover = output.with_suffix(".json"), output.with_suffix(".png")
    _require(
        output.suffix == ".mp4" and not any(p.exists() for p in (output, sidecar, cover)),
        "outputs must be new mp4/json/png paths",
    )
    protocol, summary, states, rows, manifest = read_recording(directory)
    indices = frame_indices(summary["steps"], protocol["episode"]["control_dt_s"], fps)
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    model = mujoco.MjModel.from_binary_path(str(directory / "model.mjb"))
    _require((model.nq, model.nv, model.nu) == (23, 22, 16), "compiled model dimensions differ")
    model.vis.global_.offwidth, model.vis.global_.offheight = WIDTH, HEIGHT
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.45
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    _require(floor >= 0, "archived model has no floor")
    model.geom_matid[floor] = -1
    model.geom_rgba[floor] = (0.68, 0.73, 0.77, 1)
    data, camera = mujoco.MjData(model), mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = 2.7, 130.0, -25.0
    font = ImageFont.truetype("DejaVuSans.ttf", 16)
    output.parent.mkdir(parents=True, exist_ok=True)
    renderer, encoder = None, None
    try:
        renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
        encoder = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-n",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{WIDTH}x{HEIGHT}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-threads",
                "1",
                "-c:v",
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
        for frame, index in enumerate(indices):
            data.qpos[:], data.qvel[:], data.time = (
                states["qpos"][index],
                states["qvel"][index],
                states["time_s"][index],
            )
            mujoco.mj_forward(model, data)  # Render geometry only; never mj_step.
            camera.lookat[:] = data.qpos[:3] + (0.2, 0, -0.1)
            renderer.update_scene(data, camera=camera)
            picture = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(picture)
            draw.rectangle((0, 0, WIDTH, 130), fill=(20, 30, 43))
            row = rows[max(0, index - 1)]
            lines = [
                f"D1 | {protocol['policy']} | {protocol['episode']['baseline']} | terrain {protocol['episode']['terrain']['layout']}",
                f"{protocol['episode']['source_schema']} | synchronous 10 ms control / 2 ms physics",
                f"t={data.time:.2f}/{protocol['episode']['duration_s']:.2f}s | cmd vx {float(row['command_vx_mps']):+.2f} m/s, yaw {float(row['command_yaw_rps']):+.2f} rad/s, clearance {float(row['command_clearance_m']):.3f} m",
                "Reset pose (no completed control interval yet)"
                if index == 0
                else f"truth vx {float(row['velocity_mps']):+.2f} m/s, yaw {float(row['yaw_rate_rps']):+.2f} rad/s, clearance {float(row['clearance_m']):.3f} m",
                f"policy SHA {(protocol['model_sha256'] or 'none')[:12]} | recorded replay, no resimulation",
            ]
            for y, line in enumerate(lines):
                draw.text((12, 6 + y * 24), line, font=font, fill="white")
            if index == summary["steps"]:
                outcome = (
                    "COMPLETE (not a quality claim)" if summary["completed"] else "FAILED / PARTIAL"
                )
                draw.rectangle((0, HEIGHT - 32, WIDTH, HEIGHT), fill=(20, 30, 43))
                draw.text(
                    (12, HEIGHT - 26),
                    f"{outcome} | {summary['stop_reason']}",
                    font=font,
                    fill="white",
                )
            if frame == len(indices) // 2:
                picture.save(cover)
            encoder.stdin.write(np.asarray(picture).tobytes())
        encoder.stdin.close()
        _require(encoder.wait(timeout=60) == 0, "ffmpeg failed; partial video retained")
    finally:
        if renderer is not None:
            renderer.close()
        if encoder is not None and encoder.poll() is None:
            encoder.terminate()
            try:
                encoder.wait(timeout=10)
            except subprocess.TimeoutExpired:
                encoder.kill()
                encoder.wait(timeout=10)
    report = {
        "schema": "d1-locomotion-recorded-video-v1",
        "input_manifest_sha256": sha256(directory / "manifest.json"),
        "input_sha256": manifest,
        "compiled_model_sha256": protocol["compiled_model_sha256"],
        "policy_sha256": protocol["model_sha256"],
        "renderer_sha256": sha256(Path(__file__)),
        "mode": protocol["policy"],
        "completed": summary["completed"],
        "stop_reason": summary["stop_reason"],
        "source_schema": protocol["episode"]["source_schema"],
        "frames": len(indices),
        "fps": fps,
        "frame_state_indices": indices,
        "simulation_duration_s": float(states["time_s"][-1]),
        "playback_duration_s": len(indices) / fps,
        "video_sha256": sha256(output),
        "cover_sha256": sha256(cover),
        "physics_steps_in_renderer": 0,
        "state_source": "recorded synchronized qpos/qvel; initial and terminal included",
    }
    sidecar.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(render(args.run, args.output, args.fps), indent=2))
