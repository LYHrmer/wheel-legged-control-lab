"""Bounded 2x2 probe of arena geometry and domain setup for the delayed stand case.

One shared D1ControlLoop, one controller adapter and one fusion provider are used
for every case; only the plant arena (infinite plane vs all-zero locomotion
heightfield) and the presence of a neutral ``set_domain()`` before ``loop.reset``
change. Sensors feed control; truth is recorded and used for termination only.
This is a measurement harness: it records the contrast and never claims a cause.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import tarfile
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from wheel_legged_control.d1.actuator_channel import ActuatorChannel, ActuatorChannelConfig
from wheel_legged_control.d1.control_loop import (
    D1ControlLoop,
    D1MotionCommand,
    D1WheelLegControllerAdapter,
)
from wheel_legged_control.d1.control_primitives import terrain_normal_to_rpy
from wheel_legged_control.d1.locomotion_terrain import (
    LOCOMOTION_MAP_HALF_SIZE_M,
    D1LocomotionTerrainConfig,
)
from wheel_legged_control.d1.model import JOINT_TORQUE_LIMIT, JOINT_VELOCITY_LIMIT, D1Plant
from wheel_legged_control.d1.sensor_estimation import D1SensorNoise
from wheel_legged_control.d1.state_provider import D1StateProviderConfig, build_d1_state_provider
from wheel_legged_control.d1.training_terrain import TrainingGroundReference
from wheel_legged_control.d1.wheel_leg_controller import (
    D1WheelLegControlConfig,
    D1WheelLegController,
)

ROOT = Path(__file__).resolve().parents[1]
# Every parameter below is the locomotion task's own synthetic setting, not an
# identified D1 value; the seed is the measurement seed full env seed 17 draws.
SPAWN_POSITION_M = (-3.8, 0.0, 0.455)
SAFE_HALF_SIZE_M = (5.3, 2.3)
MEASUREMENT_SEED = 1814784298
DURATION_S = 4.0
SENSOR_DELAY_STEPS = 2
CONTROL_GAINS = D1WheelLegControlConfig(wheel_kp=0.55, wheel_ki=1.5, yaw_feedback_gain=4.0)
SENSOR_NOISE = D1SensorNoise(
    gyro_std_rad_s=0.002,
    accelerometer_std_m_s2=0.03,
    encoder_position_std_rad=0.0005,
    encoder_velocity_std_rad_s=0.005,
)
ACTUATOR_CONFIG = ActuatorChannelConfig(torque_limit_nm=tuple(JOINT_TORQUE_LIMIT))
# name: (flat locomotion heightfield instead of the plane, neutral set_domain call)
CASES = {
    "plane_bare": (False, False),
    "plane_set_domain": (False, True),
    "heightfield_bare": (True, False),
    "heightfield_set_domain": (True, True),
}
FLAT_HEIGHT_TOLERANCE_M = 0.001
FLAT_SLOPE_TOLERANCE_RAD = np.deg2rad(0.25)
SOURCE_FILES = (
    "scripts/probe_d1_delay_parity.py",
    "src/wheel_legged_control/d1/actuator_channel.py",
    "src/wheel_legged_control/d1/control_context.py",
    "src/wheel_legged_control/d1/control_loop.py",
    "src/wheel_legged_control/d1/control_primitives.py",
    "src/wheel_legged_control/d1/controllers.py",
    "src/wheel_legged_control/d1/locomotion_env.py",
    "src/wheel_legged_control/d1/locomotion_terrain.py",
    "src/wheel_legged_control/d1/model.py",
    "src/wheel_legged_control/d1/sensor_estimation.py",
    "src/wheel_legged_control/d1/state_estimation.py",
    "src/wheel_legged_control/d1/state_provider.py",
    "src/wheel_legged_control/d1/training_terrain.py",
    "src/wheel_legged_control/d1/wheel_leg_controller.py",
)


def build_case(*, heightfield: bool, noise: bool = True) -> tuple[D1Plant, D1ControlLoop]:
    """Same loop/provider/adapter for both arenas; only the geometry differs."""
    plant = D1Plant(
        sampling_mode="synchronized",
        locomotion_terrain=D1LocomotionTerrainConfig() if heightfield else None,
        actuator_channel=ActuatorChannel(ACTUATOR_CONFIG),
    )
    provider = build_d1_state_provider(
        plant,
        D1StateProviderConfig(
            kind="imu_encoder_fusion",
            sensor_noise=SENSOR_NOISE if noise else D1SensorNoise(),
            sensor_delay_steps=SENSOR_DELAY_STEPS,
            initial_position_m=SPAWN_POSITION_M,
            initial_rpy_rad=(0.0, 0.0, 0.0),
        ),
    )
    adapter = D1WheelLegControllerAdapter(
        D1WheelLegController(control_dt=plant.control_dt, **asdict(CONTROL_GAINS))
    )
    return plant, D1ControlLoop(plant, provider, adapter)


def truth_ground(plant: D1Plant, x: float, y: float) -> TrainingGroundReference:
    """Oracle geometry for logging/termination only, never for control."""
    if plant.locomotion_terrain is None:
        return TrainingGroundReference(0.0, 0.0, 0.0)
    half = np.asarray(LOCOMOTION_MAP_HALF_SIZE_M)
    return plant.locomotion_ground_reference(*np.clip((x, y), -half, half))


def observe(
    plant: D1Plant,
    step: int,
    sensor_ground,
    state,
    *,
    rejected: str | None = None,
    decision=None,
) -> dict:
    """Record one tick from sensor publication plus oracle truth."""
    position, rpy = plant.base_position, plant.base_rpy
    ground = truth_ground(plant, float(position[0]), float(position[1]))
    references = [ground] + [
        truth_ground(plant, float(point[0]), float(point[1]))
        for point in plant.measurement_data.xpos[list(plant.wheel_body_ids_by_leg)]
    ]
    roll, pitch = terrain_normal_to_rpy(
        -np.tan(sensor_ground.pitch_rad), np.tan(sensor_ground.roll_rad), float(state.base_rpy[2])
    )
    if decision is not None:
        # Actual pre-step reference; the publication below is post-step.
        roll, pitch = decision.world_command.roll_rad, decision.world_command.pitch_rad
    traces = plant.last_control_interval_actuator_traces
    clearance = float(position[2]) - ground.height_m
    fallen = (
        clearance < 0.22
        or float(np.max(np.abs(rpy[:2]))) > 0.85
        or plant.undesired_ground_contacts > 0
    )
    outside = bool(np.any(np.abs(position[:2]) >= np.asarray(SAFE_HALF_SIZE_M)))
    return {
        "step": step + 1,
        # A rejected prepare consumes no physics interval, so the clock, the
        # publication and the actuator traces still describe the previous tick.
        "tick_completed": int(rejected is None),
        "time_s": float(plant.data.time),
        "control_time_s": state.control_time_s,
        "measurement_time_s": state.measurement_time_s,
        "measurement_age_s": state.age_s,
        "x_m": float(position[0]),
        "y_m": float(position[1]),
        "z_m": float(position[2]),
        "roll_rad": float(rpy[0]),
        "pitch_rad": float(rpy[1]),
        "yaw_rad": float(rpy[2]),
        "clearance_m": clearance,
        "truth_ground_height_m": ground.height_m,
        "truth_ground_slope_rad": max(abs(ground.roll_rad), abs(ground.pitch_rad)),
        "nonflat_now": int(
            any(
                abs(g.height_m) > FLAT_HEIGHT_TOLERANCE_M
                or max(abs(g.roll_rad), abs(g.pitch_rad)) > FLAT_SLOPE_TOLERANCE_RAD
                for g in references
            )
        ),
        "sensor_ground_height_m": sensor_ground.height_m,
        "sensor_ground_roll_rad": sensor_ground.roll_rad,
        "sensor_ground_pitch_rad": sensor_ground.pitch_rad,
        "command_was_applied": int(decision is not None and rejected is None),
        "world_command_roll_rad": float(roll),
        "world_command_pitch_rad": float(pitch),
        "estimated_wheel_contacts": int(np.count_nonzero(state.wheel_contact)),
        "truth_wheel_ground_contacts": plant.wheel_ground_contacts,
        "undesired_ground_contacts": plant.undesired_ground_contacts,
        "max_joint_velocity_rad_s": float(np.max(np.abs(plant.joint_velocity))),
        "joint_velocity_rated_fraction": float(
            np.max(np.abs(plant.joint_velocity) / JOINT_VELOCITY_LIMIT)
        ),
        "mechanical_activity_w": float(
            np.mean([np.sum(np.abs(t.applied_nm * t.joint_velocity_rps)) for t in traces])
            if traces
            else 0.0
        ),
        "torque_rated_fraction": float(
            np.max([np.max(np.abs(t.applied_nm) / JOINT_TORQUE_LIMIT) for t in traces])
            if traces
            else 0.0
        ),
        "termination_reason": f"prepared_command_rejected: {rejected}"
        if rejected is not None
        else "fall_or_body_contact"
        if fallen
        else "map_boundary"
        if outside
        else "",
    }


def run_case(
    name: str, *, duration_s: float = DURATION_S, noise: bool = True, seed: int = MEASUREMENT_SEED
) -> tuple[list[dict], dict]:
    """Stand for ``duration_s`` under a zero action, stopping at the first failure."""
    if (
        not np.isfinite(duration_s)
        or duration_s < 0.01
        or not np.isclose(duration_s / 0.01, round(duration_s / 0.01), atol=1e-8, rtol=0)
    ):
        raise ValueError("duration must be a positive integer number of control ticks")
    heightfield, domain = CASES[name]
    plant, loop = build_case(heightfield=heightfield, noise=noise)
    expected_steps = round(duration_s / plant.control_dt)
    rows: list[dict] = []
    started = perf_counter()
    try:
        if domain:
            # mj_setConst rewrites data, not the model arrays, so reset follows.
            plant.set_domain(
                base_mass_scale=1.0,
                damping_scale=1.0,
                friction_scale=1.0,
                actuator_strength_scale=1.0,
            )
        loop.reset(seed=seed, base_position=SPAWN_POSITION_M)
        for step in range(expected_steps):
            sensor_ground, state = loop.provider.ground_reference(), loop.provider.read()
            try:
                decision = loop.prepare(D1MotionCommand())
            except ValueError as error:
                rows.append(observe(plant, step, sensor_ground, state, rejected=str(error)))
                break
            loop.step(np.zeros(8))
            rows.append(
                observe(
                    plant,
                    step,
                    loop.provider.ground_reference(),
                    loop.provider.read(),
                    decision=decision,
                )
            )
            if rows[-1]["termination_reason"]:
                break
    finally:
        wall_clock_s = perf_counter() - started
        del loop, plant
        gc.collect()
    failed = [r for r in rows if r["termination_reason"]]
    summary = {
        "case": name,
        "arena": "flat_locomotion_heightfield" if heightfield else "infinite_plane",
        "set_domain_before_reset": domain,
        "sensor_noise": noise,
        "measurement_seed": seed,
        "duration_s": duration_s,
        "expected_steps": expected_steps,
        "recorded_rows": len(rows),
        "executed_steps": sum(r["tick_completed"] for r in rows),
        "completed": len(rows) == expected_steps and not failed,
        "termination_reason": failed[0]["termination_reason"] if failed else "duration_reached",
        "first_failure_time_s": failed[0]["time_s"] if failed else None,
        "first_failure_step": failed[0]["step"] if failed else None,
        "final_time_s": rows[-1]["time_s"],
        "all_values_finite": bool(
            np.isfinite([v for r in rows for v in r.values() if isinstance(v, (int, float))]).all()
        ),
        "wall_clock_s": wall_clock_s,
        "measurement_age_values_s": sorted({round(r["measurement_age_s"], 12) for r in rows}),
        "control_clock_consistent": all(
            np.isclose(r["control_time_s"], r["time_s"], rtol=0, atol=1e-10)
            and np.isclose(
                r["control_time_s"],
                0.01 * (r["step"] - 1 + r["tick_completed"]),
                rtol=0,
                atol=1e-10,
            )
            for r in rows
        ),
        "min_clearance_m": min(r["clearance_m"] for r in rows),
        "max_abs_roll_pitch_rad": max(max(abs(r["roll_rad"]), abs(r["pitch_rad"])) for r in rows),
        "max_joint_velocity_rated_fraction": max(r["joint_velocity_rated_fraction"] for r in rows),
        "max_mechanical_activity_w": max(r["mechanical_activity_w"] for r in rows),
        "max_abs_world_command_roll_pitch_rad": max(
            max(abs(r["world_command_roll_rad"]), abs(r["world_command_pitch_rad"])) for r in rows
        ),
        "max_abs_truth_ground_height_m": max(abs(r["truth_ground_height_m"]) for r in rows),
        "max_truth_ground_slope_rad": max(r["truth_ground_slope_rad"] for r in rows),
        "nonflat_steps": sum(r["nonflat_now"] for r in rows),
        "min_truth_wheel_ground_contacts": min(r["truth_wheel_ground_contacts"] for r in rows),
        "displacement_m": float(
            np.hypot(rows[-1]["x_m"] - SPAWN_POSITION_M[0], rows[-1]["y_m"] - SPAWN_POSITION_M[1])
        ),
    }
    lost = [r for r in rows if r["truth_wheel_ground_contacts"] == 0]
    summary["first_zero_contact_time_s"] = lost[0]["time_s"] if lost else None
    nonflat = [r for r in rows if r["nonflat_now"]]
    summary["first_nonflat_time_s"] = nonflat[0]["time_s"] if nonflat else None
    return rows, summary


def protocol(cases: list[str], duration_s: float, noise: bool, seed: int) -> dict:
    return {
        "purpose": "isolate arena geometry and domain setup for the delayed stand failure",
        "scope": "synthetic simulation diagnostic; no hardware claim, no parameter search",
        "control_schema": D1ControlLoop.schema,
        "cases": {name: dict(zip(("heightfield", "set_domain"), CASES[name])) for name in cases},
        "command": {"kind": "D1MotionCommand()", **asdict(D1MotionCommand())},
        "action": "zeros(8) every tick",
        "duration_s": duration_s,
        "physics_dt_s": 0.002,
        "control_dt_s": 0.01,
        "sampling_mode": "synchronized",
        "spawn_position_m": SPAWN_POSITION_M,
        "safe_half_size_m": SAFE_HALF_SIZE_M,
        "measurement_seed": seed,
        "measurement_seed_origin": "second draw of full-env reset(seed=17)",
        "sensor_noise_enabled": noise,
        "provider": asdict(
            D1StateProviderConfig(
                kind="imu_encoder_fusion",
                sensor_noise=SENSOR_NOISE if noise else D1SensorNoise(),
                sensor_delay_steps=SENSOR_DELAY_STEPS,
                initial_position_m=SPAWN_POSITION_M,
                initial_rpy_rad=(0.0, 0.0, 0.0),
            )
        ),
        "actuator_channel": asdict(ACTUATOR_CONFIG),
        "controller": asdict(CONTROL_GAINS),
        "terrain": asdict(D1LocomotionTerrainConfig()),
        "termination": {
            "clearance_below_m": 0.22,
            "abs_roll_pitch_above_rad": 0.85,
            "undesired_ground_contacts_above": 0,
            "source": "oracle truth, logging and termination only",
        },
        "flatness_definition": {
            "height_tolerance_m": FLAT_HEIGHT_TOLERANCE_M,
            "slope_tolerance_rad": FLAT_SLOPE_TOLERANCE_RAD,
        },
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCE_FILES
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory; never reused")
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--duration", type=float, default=DURATION_S)
    parser.add_argument("--seed", type=int, default=MEASUREMENT_SEED)
    parser.add_argument("--zero-noise", action="store_true", help="run the noiseless contrast")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist; results are never overwritten")
    if (
        not np.isfinite(args.duration)
        or args.duration < 0.01
        or args.seed < 0
        or not np.isclose(args.duration / 0.01, round(args.duration / 0.01), atol=1e-8, rtol=0)
    ):
        parser.error(
            "duration must be an integer number of control ticks; seed must be nonnegative"
        )
    if len(args.cases) != len(set(args.cases)):
        parser.error("cases must not repeat")
    args.output.mkdir(parents=True)
    noise = not args.zero_noise
    (args.output / "protocol.json").write_text(
        json.dumps(protocol(args.cases, args.duration, noise, args.seed), indent=2) + "\n"
    )
    with tarfile.open(args.output / "source.tar.gz", "w:gz") as archive:
        for path in SOURCE_FILES:
            archive.add(ROOT / path, arcname=path)
    summaries = []
    for name in args.cases:
        rows, summary = run_case(name, duration_s=args.duration, noise=noise, seed=args.seed)
        with (args.output / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    (args.output / "summary.json").write_text(
        json.dumps({"cases": summaries, "protocol": "protocol.json"}, indent=2) + "\n"
    )
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(args.output.iterdir())
                if path.is_file()
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
