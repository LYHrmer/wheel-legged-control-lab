"""Audit fixed road geometry against MuJoCo rays; no robot/policy evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np

from wheel_legged_control.d1 import locomotion_terrain
from wheel_legged_control.d1.locomotion_terrain import (
    LOCOMOTION_TERRAIN_SCHEMA,
    add_locomotion_terrain,
    locomotion_ground_reference,
    locomotion_terrain_configs,
)


def audit_terrain() -> dict:
    records = []
    for split in ("train", "development", "holdout"):
        for index, config in enumerate(locomotion_terrain_configs(split)):
            spec = mujoco.MjSpec.from_string(
                '<mujoco><worldbody><geom name="floor" type="plane" size="0 0 .1"/>'
                "</worldbody></mujoco>"
            )
            started = perf_counter()
            add_locomotion_terrain(spec, config)
            model = spec.compile()
            build_ms = (perf_counter() - started) * 1000.0
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            errors, normal_errors = [], []
            # This RNG chooses audit points only; it never defines a road or split.
            rng = np.random.default_rng(20260909)
            for x, y in rng.uniform((-5.99, -2.99), (5.99, 2.99), (256, 2)):
                reference = locomotion_ground_reference(model, float(x), float(y))
                normal = np.zeros(3)
                distance = mujoco.mj_rayHfield(
                    model,
                    data,
                    floor_id,
                    np.asarray((x, y, 2.0)),
                    np.asarray((0.0, 0.0, -1.0)),
                    normal,
                )
                if distance < 0.0:
                    raise AssertionError(f"ray missed {split}/{index} at {(x, y)}")
                expected = np.asarray(
                    (
                        np.tan(reference.pitch_rad),
                        -np.tan(reference.roll_rad),
                        1.0,
                    )
                )
                expected /= np.linalg.norm(expected)
                errors.append(abs(reference.height_m - (2.0 - distance)))
                normal_errors.append(float(np.max(np.abs(normal - expected))))
            heights = model.geom_pos[floor_id, 2] + model.hfield_size[
                0, 2
            ] * model.hfield_data.astype(np.float64).reshape(121, 241)
            # Each cell has two slopes, not a bilinear-interpolated gradient.
            sx0, sy0 = np.diff(heights[:-1], axis=1) / 0.05, np.diff(heights[:, 1:], axis=0) / 0.05
            sx1, sy1 = np.diff(heights[1:], axis=1) / 0.05, np.diff(heights[:, :-1], axis=0) / 0.05
            max_slope = max(np.hypot(sx0, sy0).max(), np.hypot(sx1, sy1).max())
            collision_sha = hashlib.sha256(
                model.hfield_data.tobytes() + model.hfield_size.tobytes() + model.geom_pos.tobytes()
            ).hexdigest()
            records.append(
                {
                    "split": split,
                    "index": index,
                    "config": json.loads(config.to_json()),
                    "collision_sha256": collision_sha,
                    "height_range_m": [float(heights.min()), float(heights.max())],
                    "max_triangle_incline_deg": float(np.rad2deg(np.arctan(max_slope))),
                    "spawn_height_m": locomotion_ground_reference(model, -3.8, 0.0).height_m,
                    "ray_count": len(errors),
                    "max_ray_height_error_m": max(errors),
                    "max_ray_normal_component_error": max(normal_errors),
                    "build_ms": build_ms,
                }
            )
    passed = all(
        row["max_ray_height_error_m"] < 2e-12
        and row["max_ray_normal_component_error"] < 2e-12
        and abs(row["spawn_height_m"]) < 1e-7
        for row in records
    )
    return {
        "schema": LOCOMOTION_TERRAIN_SCHEMA,
        "mujoco_version": mujoco.__version__,
        "terrain_source_sha256": hashlib.sha256(
            Path(locomotion_terrain.__file__).read_bytes()
        ).hexdigest(),
        "scope": "collision_geometry_only; no robot traversal or policy quality claim",
        "map_bounds_xy_m": [[-6.0, 6.0], [-3.0, 3.0]],
        "grid_shape_yx": [121, 241],
        "grid_spacing_m": 0.05,
        "heightfield_bytes_per_map": 121 * 241 * 4,
        "flat_spawn_strip_x_m": [-6.0, -2.4],
        "step_bevel_m": 0.05,
        "passed": passed,
        "roads": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/d1_v3_terrain/geometry_audit.json")
    )
    args = parser.parse_args()
    report = audit_terrain()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"output": str(args.output), "passed": report["passed"], "roads": len(report["roads"])}
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
