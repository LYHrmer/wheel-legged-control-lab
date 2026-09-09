"""Measure physical/control sampling cost and short closed-loop consequences.

This is a development comparison, not a held-out RL or real-time guarantee.
Every repetition resets the same deterministic task; repetitions measure timing
variation, not additional independent terrain/noise samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import tarfile
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np

from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.hierarchical import D1LQRVMCController
from wheel_legged_control.d1.model import D1Plant
from wheel_legged_control.d1.sensor_estimation import D1SensorStateSource
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource


def run_case(mode: str, state_kind: str, duration_s: float, repeat: int) -> tuple[list, dict]:
    plant = D1Plant(sampling_mode=mode)
    controller = D1LQRVMCController(plant)
    source = (
        D1MujocoTruthStateSource(plant) if state_kind == "oracle" else D1SensorStateSource(plant)
    )
    state = source.reset(seed=17)
    truth_source = D1MujocoTruthStateSource(plant)
    truth_source.reset()
    rows = []
    # Model compilation and LQR identification are intentionally outside the
    # timed control loop. There is no unrecorded warmup physics/reset.
    for tick in range(round(duration_s / plant.control_dt)):
        command = D1Command(forward_velocity_mps=0.2 if tick * plant.control_dt >= 0.5 else 0.0)
        start = perf_counter()
        torque = controller.compute(command, state)
        controlled = perf_counter()
        plant.step(torque)
        stepped = perf_counter()
        state = source.read()
        read = perf_counter()
        truth = truth_source.read()
        # Post-integrated qpos is a phase-independent positional reference.
        rows.append(
            {
                "tick": tick + 1,
                "time_s": float(plant.data.time),
                "command_vx_mps": command.forward_velocity_mps,
                "published_vx_mps": float(truth.base_linear_velocity_body[0]),
                "height_error_m": float(plant.data.qpos[2] - command.base_height_m),
                "pitch_rad": float(truth.base_rpy[1]),
                "origin_phase_error_m": float(
                    np.max(np.abs(truth.base_position - plant.data.qpos[:3]))
                ),
                "controller_ms": 1000 * (controlled - start),
                "physics_and_sample_ms": 1000 * (stepped - controlled),
                "state_source_ms": 1000 * (read - stepped),
                "control_tick_ms": 1000 * (read - start),
            }
        )
        if truth.has_fallen():
            break
    tail = [row for row in rows if row["time_s"] > 1.0]
    summary = {
        "sampling_mode": mode,
        "state_source": state_kind,
        "repeat": repeat,
        "steps": len(rows),
        "completed": len(rows) == round(duration_s / plant.control_dt),
        "height_rmse_m": float(np.sqrt(np.mean([r["height_error_m"] ** 2 for r in rows]))),
        "tail_velocity_rmse_mps": float(
            np.sqrt(np.mean([(r["published_vx_mps"] - r["command_vx_mps"]) ** 2 for r in tail]))
        ),
        "max_origin_phase_error_m": max(r["origin_phase_error_m"] for r in rows),
    }
    # Skip the first tick only for timing percentiles, never for control quality.
    for field in ("controller_ms", "physics_and_sample_ms", "state_source_ms", "control_tick_ms"):
        values = [row[field] for row in rows[1:]]
        for label, quantile in (("median", 0.5), ("p95", 0.95), ("p99", 0.99), ("max", 1.0)):
            summary[f"{field}_{label}"] = float(np.quantile(values, quantile))
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not np.isfinite(args.duration) or not 2 <= args.duration <= 10:
        parser.error("duration must be finite and between 2 and 10 seconds")
    if not np.isclose(args.duration / 0.01, round(args.duration / 0.01)):
        parser.error("duration must be an integer number of 10 ms control ticks")
    if not 1 <= args.repeats <= 10:
        parser.error("repeats must be between 1 and 10")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "src/wheel_legged_control").rglob("*.py")) + [Path(__file__).resolve()]
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    with tarfile.open(args.output / "source.tar.gz", "w:gz") as archive:
        for source in sources:
            archive.add(source, arcname=source.relative_to(root))
    protocol = {
        "schema": "d1-runtime-sampling-development-v1",
        "duration_s": args.duration,
        "repeats": args.repeats,
        "noise_seed": 17,
        "state_noise": "zero",
        "physics_dt_s": 0.002,
        "control_dt_s": 0.01,
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "source_sha256": hashes,
        "notes": [
            "Same deterministic task; repetitions are timing repeats, not new samples.",
            "Control tick excludes CSV and audit truth capture; includes requested state source.",
            "Height uses post-integrated qpos; velocity is each mode's published COM/body velocity.",
            "Compare timing jointly with all three repetitions and CPU load, not only fastest run.",
        ],
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    summaries = []
    for repeat in range(args.repeats):
        for kind in ("oracle", "imu_encoder_fusion"):
            # Alternate order to avoid consistently favoring a later mode.
            modes = (
                ("legacy_mixed", "synchronized")
                if repeat % 2 == 0
                else ("synchronized", "legacy_mixed")
            )
            for mode in modes:
                rows, summary = run_case(mode, kind, args.duration, repeat)
                with (args.output / f"{kind}_{mode}_{repeat}.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    changed = [
        name
        for name, sha in hashes.items()
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != sha
    ]
    (args.output / "source_consistency.json").write_text(
        json.dumps({"changed_during_run": changed}, indent=2) + "\n"
    )
    if changed:
        raise RuntimeError(
            "Source changed during benchmark; retain this attempt but rerun before comparison"
        )


if __name__ == "__main__":
    main()
