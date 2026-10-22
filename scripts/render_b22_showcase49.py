"""Render the complete C31 B22 recordings, without policy inference or stepping.

The frozen launch entry is deliberately not used: its environment has drifted.
Only archived poses are shown. Compiling the same robot/course geometry for
drawing is not a new closed-loop execution or performance qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "results/b22_suite31_20261018_01"
PROFILES = ("flat_1p6", "yaw_1p2")


def identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def checked_json(path: Path, expected: dict | None = None) -> dict:
    if expected is not None:
        require(identity(path) == expected, f"changed source: {path}")
    return json.loads(path.read_text())


def verify_source(run: Path, profile: str) -> tuple[dict, dict, dict]:
    require(profile in PROFILES, "unregistered profile")
    suite = checked_json(run / "report.json")
    require(suite["status"] == "COMPLETE" and suite["failure"] is None
            and suite["finite_suite_task_qualification_passed"] is True,
            "requires the entire COMPLETE qualified C31 suite, never a partial candidate")
    stage = profile + "_B"
    bound = next(item["report"] for item in suite["stages"] if item["name"] == stage)
    readback = checked_json(run / f"{stage}_readback.json", bound)
    require(readback["status"] == "COMPLETE", "readback incomplete")
    case = readback["cases"][0]
    require(case["completed_controls"] == 1600 and case["record"]["record_valid"]
            and case["record"]["full_horizon"] and case["safety"]["passed"]
            and case["final_window"]["passed"], "case not complete and safe")
    require({"episode_0/states.npz", "command_identity.json", "construction_receipt.json"}
            <= readback["artifacts"].keys(), "required artifact identities missing")
    for name, expected in readback["artifacts"].items():
        path = run / stage / name
        require(path.resolve().is_relative_to((run / stage).resolve()), "unsafe artifact path")
        require(identity(path) == expected, f"changed archived artifact: {name}")
    release = checked_json(run / "release.json", readback["release_identity"])
    # The rendering model uses this source tree and its redistributable assets.
    source_inputs = {name: value for name, value in release["inputs"].items()
                     if name.startswith(str(ROOT / "src") + "/")}
    require(source_inputs, "missing robot/model source closure")
    for name, expected in source_inputs.items():
        require(identity(Path(name)) == expected, f"render model source changed: {name}")
    command = checked_json(run / stage / "command_identity.json")
    require(command["actor"] == "B" and command["profile"] == profile
            and command["control_limit"] == 1600, "actor/profile mismatch")
    construction = checked_json(run / stage / "construction_receipt.json")
    return readback, construction, source_inputs


def check_geometry(model, construction: dict, mujoco, np) -> None:
    binding = construction["geometry_binding"]
    require((model.nq, model.nv, model.nu) == (23, 22, 16), "model dimensions changed")
    for key in ("actuator_gear", "actuator_trnid", "actuator_trntype", "body_mass",
                "geom_bodyid", "geom_conaffinity", "geom_contype", "geom_size", "geom_type"):
        require(np.array_equal(getattr(model, key), binding[key]), f"geometry binding differs: {key}")
    require(np.array_equal(model.jnt_qposadr[binding["joint_ids"]], binding["joint_qpos_addresses"])
            and np.array_equal(model.jnt_dofadr[binding["joint_ids"]], binding["joint_dof_addresses"]),
            "recorded joint layout differs")
    require(np.array_equal(model.body_ipos[binding["base_body_id"]], binding["base_body_ipos"]),
            "base COM differs")
    for row in construction["compiled_geometry"]["world_collision_geoms"]:
        gid = row["geom_id"]
        require(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) == row["name"],
                "course geometry name differs")
        for field, recorded in (("geom_pos", "position_m"), ("geom_size", "size_m"),
                                ("geom_quat", "quaternion_wxyz")):
            require(np.array_equal(getattr(model, field)[gid], row[recorded]),
                    f"course geometry differs: {field}")


def render(run: Path, profile: str, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "b22-recorded-showcase49-v1", "status": "INCOMPLETE",
              "profile": profile, "actor": "frozen B22 + C18, gain 1.0",
              "source_execution": str(run), "physics_step_calls": 0,
              "policy_calls": 0, "model_compilations": 0, "forward_calls": 0,
              "compiler_internal_steps_instrumented": False,
              "new_performance_evidence": False, "error": None}
    def interrupted(signum, _frame):
        raise RuntimeError(f"render interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        readback, construction, source_inputs = verify_source(run, profile)
        report["source_suite_report"] = identity(run / "report.json")
        report["source_readback"] = identity(run / f"{profile}_B_readback.json")
        report["render_model_sources"] = source_inputs
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
        sys.path.insert(0, str(ROOT / "src"))
        import mujoco
        import numpy as np
        from PIL import Image

        from wheel_legged_control.d1.model import build_d1_model

        require(mujoco.__version__ == "3.12.0", "render requires archived MuJoCo 3.12.0")
        def no_step(*_args, **_kwargs):
            raise RuntimeError("physics stepping forbidden in recorded renderer")
        mujoco.mj_step = no_step
        mujoco.mj_step1 = no_step
        mujoco.mj_step2 = no_step
        with np.load(run / f"{profile}_B/episode_0/states.npz", allow_pickle=False) as saved:
            qpos, qvel, times = (saved[key].copy() for key in ("qpos", "qvel", "time"))
        require(qpos.shape == (1601, 23) and qvel.shape == (1601, 22)
                and times.shape == (1601,), "incomplete state arrays")
        require(np.isfinite(qpos).all() and np.isfinite(qvel).all()
                and np.allclose(times, np.arange(1601) * .01, atol=1e-9, rtol=0),
                "invalid state clock/data")
        model = build_d1_model(arena="course")
        report["model_compilations"] += 1
        check_geometry(model, construction, mujoco, np)
        report["compiled_geometry_matches_archive"] = True
        data, camera = mujoco.MjData(model), mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance, camera.azimuth, camera.elevation = 1.8, 125., -24.
        model.vis.global_.offwidth, model.vis.global_.offheight = 512, 288
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        model.geom_matid[floor] = -1
        model.geom_rgba[floor] = (.72, .77, .82, 1.)
        model.vis.headlight.ambient[:] = .5
        report["visual_only_changes"] = {"camera_distance_m": 1.8,
            "floor_color": [.72, .77, .82], "world_grid_spacing_m": .5,
            "grid_changes_physics": False}
        scene_options = mujoco.MjvOption()
        scene_options.geomgroup[3] = 0  # Hide collision proxies, show original visual meshes.
        pngs = output / "frames"
        pngs.mkdir()
        indices = list(range(0, 1600, 5)) + [1600]
        with mujoco.Renderer(model, height=288, width=512) as renderer:
            for frame, index in enumerate(indices):
                data.qpos[:], data.qvel[:], data.time = qpos[index], qvel[index], times[index]
                mujoco.mj_forward(model, data)
                report["forward_calls"] += 1
                camera.lookat[:] = data.qpos[:3] + np.array([.15, 0, -.1])
                renderer.update_scene(data, camera=camera, scene_option=scene_options)
                # World-fixed drawing aids make displacement visible in a follow camera.
                # These are mjvScene primitives, never contacts or model geoms.
                for axis in (0, 1):
                    center = round(data.qpos[axis] * 2) / 2
                    for offset in range(-8, 9):
                        position = data.qpos[:3].copy()
                        position[2] = .002
                        position[axis] = center + .5 * offset
                        size = np.array([4., 4., .0005])
                        size[axis] = .002
                        geom = renderer.scene.geoms[renderer.scene.ngeom]
                        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX, size,
                                           position, np.eye(3).reshape(-1),
                                           np.array([.49, .55, .61, 1.], dtype=np.float32))
                        renderer.scene.ngeom += 1
                Image.fromarray(renderer.render()).save(pngs / f"{frame:04d}.png")
        with (output / "png_manifest.json").open("x") as stream:
            json.dump({p.name: identity(p) for p in sorted(pngs.glob("*.png"))}, stream, indent=2)
        gif = output / f"b22_{profile}.gif"
        # A single global palette and ordered dithering avoid temporal speckle.
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n",
                   "-threads", "1", "-filter_threads", "1", "-filter_complex_threads", "1",
                   "-framerate", "20", "-i", str(pngs / "%04d.png"),
                   "-filter_complex", "[0:v]split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
                   "-loop", "0", "-threads", "1", str(gif)]
        subprocess.run(command, check=True, timeout=120)
        require(gif.stat().st_size <= 5_000_000, "GIF exceeds the 5 MB presentation limit")
        with Image.open(gif) as check:
            duration_ms = 0
            for index in range(check.n_frames):
                check.seek(index)
                duration_ms += check.info["duration"]
            report["gif_frames"] = check.n_frames
        require(duration_ms == 16050, "GIF duration differs from the 20 fps recorded sequence")
        require(all(identity(Path(name)) == value for name, value in source_inputs.items()),
                "render source changed while drawing")
        report.update(status="COMPLETE", source_controls=1600, png_frames=len(indices), fps=20,
                      simulation_duration_s=16., playback_duration_s=duration_ms / 1000,
                      gif=identity(gif), frame_state_indices=indices,
                      png_manifest=identity(output / "png_manifest.json"),
                      source_states=readback["artifacts"]["episode_0/states.npz"],
                      renderer=identity(Path(__file__)))
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        with (output / "report.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.run, args.profile, args.output), indent=2))
