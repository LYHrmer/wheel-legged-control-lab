"""Development-only, single-variable attitude-reference probes on D1 bumps.

Changes live inside each process's controller wrapper. No production source,
reward, observation, reset pose, residual, gain or height target is modified.
This is a diagnosis, not a new trained policy or a production environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from wheel_legged_control.d1.terrain_env import D1TerrainResidualEnv

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("world_upright", "local_tangent", "support_plane")


def normal_to_rpy(slope_x: float, slope_y: float, yaw: float) -> tuple[float, float]:
    """Align the body z axis to [-dh/dx, -dh/dy, 1] with the current yaw.

    R = Rz(yaw) Ry(pitch) Rx(roll). A positive world-x slope therefore gives
    negative pitch at yaw=0. The coupled roll expression is exact, not atan(b).
    """

    forward = slope_x * math.cos(yaw) + slope_y * math.sin(yaw)
    lateral = -slope_x * math.sin(yaw) + slope_y * math.cos(yaw)
    pitch = -math.atan(forward)
    roll = math.atan2(lateral, math.sqrt(1.0 + forward * forward))
    return roll, pitch


def references(plant: Any, state: Any) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Return local and four-wheel projected-ground RPY, plus plane fit error.

    All four wheel-centre XY projections are used, regardless of whether a
    wheel is touching. Actual contacts/forces/contact-point flags are not read.
    """

    x, y = state.base_position[:2]
    ground = plant.training_ground_reference(float(x), float(y))
    local_slope_x = -math.tan(ground.pitch_rad)
    local = normal_to_rpy(local_slope_x, 0.0, float(state.base_rpy[2]))
    xy = np.asarray(state.foot_position[:, :2])
    heights = np.asarray(
        [plant.training_ground_reference(float(point[0]), float(point[1])).height_m for point in xy]
    )
    centered = xy - xy.mean(axis=0)
    design = np.column_stack((centered, np.ones(4)))
    coefficients, _, rank, _ = np.linalg.lstsq(design, heights, rcond=None)
    if rank != 3:
        raise ValueError("four wheel projections do not determine a support plane")
    support = normal_to_rpy(
        float(coefficients[0]), float(coefficients[1]), float(state.base_rpy[2])
    )
    fit_rmse = float(np.sqrt(np.mean(np.square(design @ coefficients - heights))))
    return local, support, fit_rmse


def development_cases() -> list[dict[str, Any]]:
    return [
        {
            "case_id": f"a{round(amplitude * 1000):02d}_w{round(wavelength * 10):02d}_p{phase_index}",
            "terrain": {
                "kind": "bumps",
                "amplitude_m": amplitude,
                "wavelength_m": wavelength,
                "phase_rad": phase,
            },
        }
        for amplitude in (0.005, 0.010)
        for wavelength in (0.8, 1.0, 1.2)
        for phase_index, phase in enumerate((0.0, math.pi / 2))
    ]


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _rate_metrics(values: np.ndarray, dt: float, prefix: str) -> dict[str, float]:
    # Do not count an artificial jump from zero to the first requested pitch.
    # Initial target offset is reported separately.
    rate = np.diff(values) / dt
    return {
        f"{prefix}_initial_rad": float(values[0]),
        f"{prefix}_rms_rad": _rms(values),
        f"{prefix}_rate_rms_rps": _rms(rate) if len(rate) else 0.0,
        f"{prefix}_rate_max_abs_rps": float(np.max(np.abs(rate))) if len(rate) else 0.0,
    }


def run_case(job: tuple[dict, float, int, int]) -> list[dict]:
    case, seconds, seed, repeats = job
    env = D1TerrainResidualEnv(
        baseline="lqr", randomize=False, episode_seconds=seconds, training_mode="flat"
    )
    original_compute = env.controller.compute
    results = []
    try:
        for repeat in range(repeats):
            for variant in VARIANTS:
                env.controller.compute = original_compute
                env.reset(
                    seed=seed,
                    options={
                        "terrain": case["terrain"],
                        "velocity_mps": 0.35,
                        "height_m": 0.455,
                    },
                )
                start_x = float(env.plant.base_position[0])
                rows = []
                reference_row: dict[str, float] = {}

                def compute(
                    command,
                    state,
                    residual_force_n=None,
                    *,
                    _variant=variant,
                    _reference_row=reference_row,
                    **kwargs,
                ):
                    local, support, plane_error = references(env.plant, state)
                    target = (command.roll_rad, command.pitch_rad)
                    if _variant == "local_tangent":
                        target = local
                    elif _variant == "support_plane":
                        target = support
                    _reference_row.update(
                        requested_roll_rad=float(target[0]),
                        requested_pitch_rad=float(target[1]),
                        local_pitch_rad=float(local[1]),
                        support_pitch_rad=float(support[1]),
                        support_roll_rad=float(support[0]),
                        plane_fit_rmse_m=plane_error,
                    )
                    # The attitude target is the sole intervention. Pass every
                    # remaining argument to the untouched original controller.
                    return original_compute(
                        replace(command, roll_rad=target[0], pitch_rad=target[1]),
                        state,
                        residual_force_n=residual_force_n,
                        **kwargs,
                    )

                env.controller.compute = compute
                for step in range(round(seconds / env.plant.control_dt)):
                    _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
                    rows.append(
                        {
                            "time_s": (step + 1) * env.plant.control_dt,
                            **reference_row,
                            "velocity_error_mps": float(info["velocity_error_mps"]),
                            "forward_velocity_mps": float(info["forward_velocity_mps"]),
                            "clearance_error_m": float(info["clearance_error_m"]),
                            "clearance_m": float(info["clearance_m"]),
                            "measured_pitch_rad": float(env.plant.base_rpy[1]),
                            "longitudinal_force_n": float(info["longitudinal_force_n"]),
                            "torque_saturation_fraction": float(info["torque_saturation_fraction"]),
                        }
                    )
                    if terminated or truncated:
                        break
                columns = {name: np.asarray([row[name] for row in rows]) for name in rows[0]}
                metrics = {
                    "case_id": case["case_id"],
                    "variant": variant,
                    "repeat": repeat,
                    "terrain": case["terrain"],
                    "duration_s": len(rows) * env.plant.control_dt,
                    "velocity_rmse_mps": _rms(columns["velocity_error_mps"]),
                    "clearance_rmse_m": _rms(columns["clearance_error_m"]),
                    "minimum_clearance_m": float(columns["clearance_m"].min()),
                    "progress_m": float(env.plant.base_position[0]) - start_x,
                    "tail_velocity_mps": float(columns["forward_velocity_mps"][-100:].mean()),
                    "mean_torque_saturation_fraction": float(
                        columns["torque_saturation_fraction"].mean()
                    ),
                    "max_plane_fit_rmse_m": float(columns["plane_fit_rmse_m"].max()),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "termination_reason": info["termination_reason"],
                }
                for field in ("requested_pitch", "local_pitch", "support_pitch"):
                    metrics.update(
                        _rate_metrics(columns[f"{field}_rad"], env.plant.control_dt, field)
                    )
                results.append({"metrics": metrics, "telemetry": rows})
    finally:
        env.controller.compute = original_compute
        env.close()
    return results


def _source_hashes() -> dict[str, str]:
    paths = [Path(__file__)]
    paths.extend(
        path
        for path in (ROOT / "src" / "wheel_legged_control").rglob("*")
        if path.is_file() and path.suffix.lower() in {".py", ".xml", ".urdf", ".stl"}
    )
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def _check_geometry() -> None:
    for slope_x, slope_y, yaw in ((0.1, 0.0, 0.0), (-0.1, 0.0, 0.0), (0.1, 0.05, 0.4)):
        roll, pitch = normal_to_rpy(slope_x, slope_y, yaw)
        heading_normal = np.asarray(
            (math.sin(pitch) * math.cos(roll), -math.sin(roll), math.cos(pitch) * math.cos(roll))
        )
        rotation = np.asarray(
            ((math.cos(yaw), -math.sin(yaw), 0), (math.sin(yaw), math.cos(yaw), 0), (0, 0, 1))
        )
        expected = np.asarray((-slope_x, -slope_y, 1.0))
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(rotation @ heading_normal, expected, atol=1e-12)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new diagnostic directory")
    parser.add_argument("--seconds", type=float, default=4.0, choices=(4.0, 6.0))
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--repeats", type=int, default=2, choices=(1, 2))
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2))
    args = parser.parse_args(argv)
    if args.seed < 0 or args.output.exists():
        raise SystemExit("seed must be nonnegative and output must not exist")
    _check_geometry()
    cases = development_cases()
    before = _source_hashes()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "scope": "development-only diagnostic; not policy selection or a production change",
        "intervention": "only controller command roll/pitch target is replaced in process",
        "variants": list(VARIANTS),
        "cases": cases,
        "episode_seconds": args.seconds,
        "seed": args.seed,
        "repeats": args.repeats,
        "command_velocity_mps": 0.35,
        "invariants": "same seed, spawn, height target, gains, zero residual, reward and observation",
        "support_reference": "least-squares oracle heights under all four wheel-centre XY projections; no contacts or filtering",
        "local_reference": "base collision-cell tangent; both nonzero variants convert world normal using current yaw",
        "pitch_rate": "difference of consecutive requested pitch divided by control_dt; first offset recorded separately",
        "source_sha256": before,
    }
    with (args.output / "protocol.json").open("x") as stream:
        json.dump(protocol, stream, indent=2, allow_nan=False)
    jobs = [(case, args.seconds, args.seed, args.repeats) for case in cases]
    if args.workers == 1:
        batches = list(map(run_case, jobs))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            batches = list(pool.map(run_case, jobs))
    results = [result for batch in batches for result in batch]
    metrics = [result["metrics"] for result in results]
    unchanged = before == _source_hashes()
    repeat_checks = []
    if args.repeats == 2:
        for case in cases:
            for variant in VARIANTS:
                pair = [
                    result
                    for result in results
                    if result["metrics"]["case_id"] == case["case_id"]
                    and result["metrics"]["variant"] == variant
                ]
                repeat_checks.append(
                    {
                        "case_id": case["case_id"],
                        "variant": variant,
                        "telemetry_identical": pair[0]["telemetry"] == pair[1]["telemetry"],
                    }
                )
    with (args.output / "summary.json").open("x") as stream:
        json.dump(
            {"source_unchanged": unchanged, "repeat_checks": repeat_checks, "metrics": metrics},
            stream,
            indent=2,
            allow_nan=False,
        )
    with (args.output / "telemetry.json").open("x") as stream:
        json.dump(results, stream, allow_nan=False)
    with (args.output / "manifest.json").open("x") as stream:
        json.dump(
            {
                "sha256": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(args.output.iterdir())
                    if path.name != "manifest.json"
                }
            },
            stream,
            indent=2,
        )
    for variant in VARIANTS:
        rows = [row for row in metrics if row["variant"] == variant and row["repeat"] == 0]
        print(
            json.dumps(
                {
                    "variant": variant,
                    "mean_case_velocity_rmse_mps": float(
                        np.mean([row["velocity_rmse_mps"] for row in rows])
                    ),
                    "mean_case_progress_m": float(np.mean([row["progress_m"] for row in rows])),
                    "terminated_cases": sum(row["terminated"] for row in rows),
                },
                allow_nan=False,
            ),
            flush=True,
        )
    print(
        f"Results: {args.output}; source unchanged: {unchanged}; repeated telemetry identical: "
        f"{all(row['telemetry_identical'] for row in repeat_checks)}",
        flush=True,
    )
    if not unchanged or any(not row["telemetry_identical"] for row in repeat_checks):
        raise SystemExit("diagnostic source freeze or repeatability check failed")


if __name__ == "__main__":
    main()
