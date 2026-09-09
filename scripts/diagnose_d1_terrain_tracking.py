"""Development-only probes for slope tracking; never edits trained artifacts.

Run from the project root. Counterfactual controller patches live only in this
process. This script reports their effects; it does not select a policy/model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from wheel_legged_control.d1.terrain_env import D1TerrainResidualEnv


def probe(variant: str, slope_deg: float, duration: float) -> dict:
    env = D1TerrainResidualEnv(randomize=False, episode_seconds=duration)
    try:
        options = {"terrain": {"kind": "ramp", "slope_deg": slope_deg}, "velocity_mps": 0.35}
        if variant == "spawn_aligned":
            options["initial_pitch"] = -float(np.deg2rad(slope_deg))
        env.reset(seed=31, options=options)
        original_compute = env.controller.compute
        mass = env.plant.nominal_total_mass_kg
        rows = []

        def compute(command, state, residual_force_n=None):
            ground = env._ground_reference()
            residual = np.array(residual_force_n, dtype=float, copy=True)
            if variant in ("gravity", "aligned_gravity"):
                residual[0] += -mass * 9.81 * np.sin(ground.pitch_rad)
            if variant in ("aligned", "aligned_gravity"):
                command = replace(command, pitch_rad=ground.pitch_rad)
            return original_compute(command, state, residual_force_n=residual)

        env.controller.compute = compute
        if variant == "unclamped_position":

            def advance(velocity_mps):
                env.controller._distance_reference_m += velocity_mps * env.plant.control_dt
                return env.controller._distance_reference_m

            env.controller._advance_position_reference = advance
        for _ in range(round(duration / env.plant.control_dt)):
            _, _, terminated, truncated, info = env.step(np.zeros(2))
            rows.append(
                (
                    info["forward_velocity_mps"],
                    info["velocity_error_mps"],
                    info["longitudinal_force_n"],
                    float(env.plant.base_rpy[1]),
                    info["torque_saturation_fraction"],
                    info["clearance_error_m"],
                )
            )
            if terminated or truncated:
                break
        values = np.asarray(rows)
        tail = values[-min(100, len(values)) :]
        return {
            "variant": variant,
            "slope_deg": slope_deg,
            "duration_s": len(rows) * env.plant.control_dt,
            "progress_m": float(env.plant.base_position[0]),
            "velocity_rmse_mps": float(np.sqrt(np.mean(values[:, 1] ** 2))),
            "tail_velocity_mps": float(tail[:, 0].mean()),
            "tail_force_n": float(tail[:, 2].mean()),
            "tail_pitch_rad": float(tail[:, 3].mean()),
            "clearance_rmse_m": float(np.sqrt(np.mean(values[:, 5] ** 2))),
            "torque_saturation": float(values[:, 4].mean()),
            "termination_reason": info["termination_reason"],
            "reference_gap_m": env.controller._distance_reference_m - env.controller._distance_m,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["original"],
        choices=(
            "original",
            "gravity",
            "aligned",
            "aligned_gravity",
            "unclamped_position",
            "spawn_aligned",
        ),
    )
    parser.add_argument("--slopes", type=float, nargs="+", default=[4.0])
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--require-forward", action="store_true")
    parser.add_argument("--output", type=Path, help="new directory for diagnostic evidence")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    results = [
        probe(variant, slope, args.seconds) for variant in args.variants for slope in args.slopes
    ]
    print(json.dumps(results, indent=2, allow_nan=False))
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
        root = Path(__file__).resolve().parents[1]
        sources = [Path(__file__), *(root / "src/wheel_legged_control").rglob("*.py")]
        payload = {
            "scope": "development-only counterfactual probes, not policy evaluation",
            "seed": 31,
            "target_velocity_mps": 0.35,
            "source_sha256": {
                str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(sources)
            },
            "results": results,
        }
        with (args.output / "probes.json").open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
    if args.require_forward and not all(r["progress_m"] > 0 for r in results):
        raise SystemExit("forward tracking failed: final progress is not positive")


if __name__ == "__main__":
    main()
