"""Development-only delay contrasts over the unchanged D1 environment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import logging
import os
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from delay_metrics import delay_steps, summarize

from wheel_legged_control.d1.actuator_channel import ActuatorChannelConfig
from wheel_legged_control.d1.locomotion_commands import (
    schedule_to_dict,
    validation_command_schedule,
)
from wheel_legged_control.d1.locomotion_env import LOCOMOTION_SPAWN_POSITION_M, D1LocomotionEnv
from wheel_legged_control.d1.locomotion_terrain import locomotion_terrain_configs
from wheel_legged_control.d1.model import (
    JOINT_POSITION_HIGH,
    JOINT_POSITION_LOW,
    JOINT_TORQUE_LIMIT,
    JOINT_VELOCITY_LIMIT,
)
from wheel_legged_control.d1.sensor_estimation import D1SensorNoise
from wheel_legged_control.d1.state_estimation import (
    D1EstimatorImpairments,
    D1MujocoTruthStateSource,
)
from wheel_legged_control.d1.state_provider import D1StateProviderConfig
from wheel_legged_control.d1.wheel_leg_controller import D1WheelLegControlConfig

HERE = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)
GAINS = D1WheelLegControlConfig(0.55, 1.5, 4, 1, 0.25)
NOISE = D1SensorNoise(0.002, 0.03, 0.0005, 0.005)
PROFILES = [
    {"name": f"fusion_noise_m{m}_a{a}", "source": "fusion_noise", "measurement_ms": m, "actuator_ms": a}
    for m, a in ((0, 0), (10, 0), (20, 0), (30, 0), (0, 10), (0, 20), (0, 30))
] + [
    {"name": f"fusion_ideal_m{m}_a0", "source": "fusion_ideal", "measurement_ms": m, "actuator_ms": 0}
    for m in (0, 30)
] + [
    {"name": f"oracle_m0_a{a}", "source": "oracle", "measurement_ms": 0, "actuator_ms": a}
    for a in (0, 30)
] + [{"name": "truth_delayed_m30_a0", "source": "truth_delayed", "measurement_ms": 30, "actuator_ms": 0}]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def provider(profile):
    delay = delay_steps(profile["measurement_ms"], "measurement")
    if profile["source"].startswith("fusion"):
        return D1StateProviderConfig(
            "imu_encoder_fusion", sensor_delay_steps=delay,
            sensor_noise=NOISE if profile["source"] == "fusion_noise" else D1SensorNoise(),
            initial_position_m=LOCOMOTION_SPAWN_POSITION_M, initial_rpy_rad=(0, 0, 0),
        )
    if profile["source"] == "truth_delayed":
        return D1StateProviderConfig("truth_impairment", impairments=D1EstimatorImpairments(delay_steps=delay))
    return D1StateProviderConfig("oracle")


def snapshot(root, output):
    files = list((root / "src").rglob("*.py")) + [root / "pyproject.toml"]
    files += [p for p in (root / "src/wheel_legged_control/d1/assets").rglob("*")
              if p.is_file() and "__pycache__" not in p.parts]
    files += [HERE / name for name in ("run_delay_channel_study.py", "delay_metrics.py", "analyze_delay_channel_study.py")]
    manifest = {}
    for source in sorted(set(files)):
        relative = Path("candidate") / source.name if source.parent == HERE else Path("repo") / source.relative_to(root)
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        target.write_bytes(data)
        manifest[str(relative)] = hashlib.sha256(data).hexdigest()
    write_json(output / "source_sha256.json", manifest)
    return manifest


def verify_source(root, manifest):
    changed = []
    for relative, expected in manifest.items():
        path = Path(relative)
        source = HERE / path.name if path.parts[0] == "candidate" else root / Path(*path.parts[1:])
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            changed.append(relative)
    return {"consistent": not changed, "checked_files": len(manifest), "changed_files": changed}


def error_fields(state, truth, prefix):
    rpy = (state.base_rpy - truth.base_rpy + np.pi) % (2 * np.pi) - np.pi
    return {
        f"{prefix}_position_error_norm_m": float(np.linalg.norm(state.base_position - truth.base_position)),
        f"{prefix}_velocity_error_norm_mps": float(np.linalg.norm(state.base_linear_velocity_world - truth.base_linear_velocity_world)),
        f"{prefix}_pitch_error_rad": float(rpy[1]),
        f"{prefix}_roll_error_rad": float(rpy[0]),
        f"{prefix}_angular_velocity_error_norm_rps": float(np.linalg.norm(state.base_angular_velocity_body - truth.base_angular_velocity_body)),
    }


def run_case(profile, seed, duration, directory, manifest):
    directory.mkdir(exist_ok=False)
    schedule = validation_command_schedule("development", 60.0)
    env = D1LocomotionEnv(
        episode_seconds=duration, terrain=locomotion_terrain_configs("development")[0],
        provider_config=provider(profile), command_source=schedule.cmd_at,
        wheel_leg_control=GAINS,
        actuator_config=ActuatorChannelConfig(
            delay_steps=delay_steps(profile["actuator_ms"], "actuator"),
            torque_limit_nm=tuple(JOINT_TORQUE_LIMIT),
        ),
    )
    rows, traces, qpos, qvel, states, controller_traces = [], [], [], [], [], []
    started, info, error = perf_counter(), None, None
    try:
        _, reset = env.reset(seed=seed)
        write_json(directory / "episode.json", {
            **reset["episode_metadata"], "episode_seed": seed, "profile": profile,
            "command_schedule": schedule_to_dict(schedule), "source_manifest": manifest,
            "evaluation_scope": "development v1 road0; fixed 60 s command schedule prefix",
        })
        truth_source = D1MujocoTruthStateSource(env.plant)
        truth_history = [truth_source.reset()]
        qpos.append(env.plant.data.qpos.copy())
        qvel.append(env.plant.data.qvel.copy())
        while True:
            _, _, terminated, truncated, info = env.step(np.zeros(8))
            transition = env.last_transition
            state, truth = transition.state, transition.truth
            truth_history.append(truth)
            index = round(state.measurement_time_s / env.plant.control_dt)
            aligned = truth_history[index]
            if not np.isclose(aligned.control_time_s, state.measurement_time_s, atol=1e-9, rtol=0):
                raise AssertionError("measurement time does not identify a recorded truth sample")
            torque = env.plant.last_control_interval_actuator_traces
            low = env.loop.controller.controller.last_result
            before = transition.decision.context.state
            saturated = np.clip(low.requested_torque_nm, -JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)
            position_guard = ((before.joint_position >= JOINT_POSITION_HIGH) & (saturated > 0)) | ((before.joint_position <= JOINT_POSITION_LOW) & (saturated < 0))
            velocity_guard = (np.abs(before.joint_velocity) >= JOINT_VELOCITY_LIMIT) & (saturated * before.joint_velocity > 0)
            controller_traces.append([low.requested_torque_nm, low.torque_nm, low.joint_target_rad,
                                      low.leg_pd_nm, low.support_nm, low.wheel_nm,
                                      before.joint_position, before.joint_velocity])
            traces.append([[getattr(t, f) for f in ("requested_nm", "limited_nm", "delayed_nm", "applied_nm", "joint_velocity_rps")] for t in torque])
            rows.append({
                **info["metrics"],
                **{f"command_{k}": v for k, v in info["command"].items()},
                "control_time_s": state.control_time_s,
                "measurement_time_s": state.measurement_time_s,
                "decision_time_s": transition.decision.context.state.control_time_s,
                "decision_measurement_time_s": transition.decision.context.state.measurement_time_s,
                "truth_roll_rad": float(truth.base_rpy[0]),
                "truth_pitch_rad": float(truth.base_rpy[1]),
                "clearance_below_0p22": int(info["metrics"]["clearance_m"] < 0.22),
                "attitude_above_0p85": int(np.max(np.abs(truth.base_rpy[:2])) > 0.85),
                "body_contact_detected": int(info["metrics"]["undesired_ground_contacts"] > 0),
                "truth_wheel_contacts": truth.wheel_ground_contacts,
                "estimate_wheel_contacts": state.wheel_ground_contacts,
                "requested_torque_abs_max_nm": float(np.max(np.abs([t.requested_nm for t in torque]))),
                "applied_torque_abs_max_nm": float(np.max(np.abs([t.applied_nm for t in torque]))),
                "controller_torque_clipped_fraction": float(np.mean(low.requested_torque_nm != saturated)),
                "controller_torque_changed_fraction": float(np.mean(low.torque_limited)),
                "controller_position_guard_fraction": float(np.mean(position_guard)),
                "controller_velocity_guard_fraction": float(np.mean(velocity_guard)),
                "controller_joint_target_limited_fraction": float(np.mean(low.joint_target_rate_limited)),
                "controller_preclip_abs_max_nm": float(np.max(np.abs(low.requested_torque_nm))),
                "nonflat_now": int(info["terrain_exposure"]["nonflat_now"]),
                **error_fields(state, truth, "current"),
                **error_fields(state, aligned, "measurement_aligned"),
            })
            qpos.append(env.plant.data.qpos.copy())
            qvel.append(env.plant.data.qvel.copy())
            states.append(np.concatenate((state.base_position, state.base_rpy, state.base_linear_velocity_world, state.base_angular_velocity_body)))
            if terminated or truncated:
                break
    except Exception as exc:
        LOGGER.exception("D1 delay case %s failed; partial artifacts are retained", directory.name)
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        env.close()
        if rows:
            with (directory / "metrics.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
        np.savez_compressed(directory / "trajectory.npz", qpos=np.asarray(qpos), qvel=np.asarray(qvel),
                            estimated_states=np.asarray(states), actuator_trace=np.asarray(traces),
                            controller_trace=np.asarray(controller_traces),
                            controller_fields=np.asarray(("requested_torque_nm", "torque_nm", "joint_target_rad", "leg_pd_nm", "support_nm", "wheel_nm", "measured_joint_position", "measured_joint_velocity")),
                            actuator_fields=np.asarray(("requested_nm", "limited_nm", "delayed_nm", "applied_nm", "joint_velocity_rps")))
    reason = "exception" if error else info["terminal_reason"]
    result = summarize(rows, duration, reason) if rows else {"completed": False, "executed_steps": 0, "terminal_reason": reason, "full_horizon_tracking": None}
    result.update(profile=profile, episode_seed=seed, error=error, wall_seconds=perf_counter() - started,
                  terrain_exposure=info["terrain_exposure"] if info else None)
    write_json(directory / "result.json", result)
    print(json.dumps({"case": directory.name, **result}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=HERE.parent)
    parser.add_argument("--duration", type=float, choices=(3.0, 60.0), default=60.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29])
    args = parser.parse_args()
    if args.seeds != ([17] if args.duration == 3.0 else [17, 29]):
        parser.error("fixed protocol: 3 s requires [17]; 60 s requires [17, 29]")
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(key) != "1":
            parser.error(f"set {key}=1 before running")
    root = args.repo_root.resolve()
    if not Path(inspect.getfile(D1LocomotionEnv)).resolve().is_relative_to(root / "src"):
        parser.error("--repo-root must match the imported D1 environment source repository")
    args.output.mkdir(exist_ok=False)
    manifest = snapshot(root, args.output)
    write_json(args.output / "protocol.json", {
        "schema": "d1-delay-channel-diagnostic-v1", "duration_s": args.duration,
        "schedule_duration_s": 60.0, "seeds": args.seeds, "profiles": PROFILES,
        "terrain_suite": "v1", "split": "development", "roads": [0],
        "controller": asdict(GAINS), "workers": 1, "python": platform.python_version(),
        "versions": {name: version(name) for name in ("mujoco", "numpy", "gymnasium")},
        "zero_residual": True, "actuator_time_constant_s": 0.0, "actuator_gain": 1.0,
        "measurement_dt_s": 0.01, "physics_dt_s": 0.002,
        "trace_layout": "control tick, physics substep, field, actuator axis; actuator trace before physics integration",
        "caveats": ["One previously used development road; no holdout or generalization claim.",
                    "Oracle/fusion ablation changes both state and ground reference source.",
                    "Full-horizon tracking is null on every early termination.",
                    "3 s smoke uses first 3 s of the fixed 60 s schedule, not a rescaled task."]})
    write_json(args.output / "process.json", {"pid": os.getpid()})
    results = []
    for profile in PROFILES:
        for seed in args.seeds:
            directory = args.output / f"{profile['name']}_road0_seed{seed}"
            results.append(run_case(profile, seed, args.duration, directory, manifest))
            write_json(args.output / "results.json", results)
    consistency = verify_source(root, manifest)
    write_json(args.output / "source_consistency.json", consistency)
    if any(r["error"] for r in results):
        raise SystemExit("All cases were retained, but at least one encountered an exception.")
    if not consistency["consistent"]:
        raise SystemExit("Source files changed during the study; results must not be accepted.")
    write_json(args.output / "finished.json", {
        "complete": True, "case_count": len(results),
        "expected_case_count": len(PROFILES) * len(args.seeds),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    main()
