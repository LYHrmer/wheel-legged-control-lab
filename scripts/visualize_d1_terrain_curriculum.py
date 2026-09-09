"""Plot a completed, paired terrain experiment without loading its policies.

Use --render-scenes only when a working MuJoCo OpenGL backend is available
(for example MUJOCO_GL=egl). Those images are zero-residual re-simulations,
not recordings of the trained policies. Default plots need only NumPy/Matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "zero_residual": "LQR + VMC (zero residual)",
    "flat_rl": "Flat PPO",
    "mixed_rl": "Mixed-terrain PPO",
    "curriculum_rl": "Curriculum PPO",
}
COLORS = {
    "zero_residual": "#59636e",
    "flat_rl": "#2366a0",
    "mixed_rl": "#32866b",
    "curriculum_rl": "#d27828",
}
TRACE_FIELDS = (
    "time_s",
    "forward_velocity_mps",
    "command_velocity_mps",
    "velocity_error_mps",
    "clearance_m",
    "command_clearance_m",
    "action_longitudinal",
    "action_vertical",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _read_csv(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields) or not set(required).issubset(fields):
            raise ValueError(f"missing or duplicate CSV columns: {path}")
        rows = list(reader)
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"empty or malformed CSV: {path}")
    return rows


def _finite_float(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"nonfinite {name}")
    return number


def protocol_profile(protocol: dict[str, Any]) -> str:
    """Reject an incompatible controller/observation interpretation before plotting."""

    version = protocol.get("schema_version")
    if version == 1:
        if protocol.get("profile", "legacy-v1") != "legacy-v1":
            raise ValueError("schema 1 requires the legacy-v1 profile")
        if protocol.get("observation_schema", "d1-terrain-oracle-v1") != "d1-terrain-oracle-v1":
            raise ValueError("legacy protocol has an incompatible observation schema")
        return "legacy-v1"
    if version != 2:
        raise ValueError("unsupported terrain experiment schema")
    expected = {
        "profile": "tracking-v2",
        "observation_schema": "d1-terrain-tracking-oracle-v2",
        "reward_schema": "d1-terrain-tracking-v2",
        "control_schema": "d1-lqr-vmc-local-tangent-v2",
    }
    if any(protocol.get(name) != value for name, value in expected.items()):
        raise ValueError("tracking-v2 protocol has incompatible control/observation/reward schemas")
    return "tracking-v2"


def protocol_conditions(protocol: dict[str, Any]) -> tuple[str, ...]:
    """Use declared training conditions, including strict missing-mixed checks."""

    profile = protocol_profile(protocol)
    conditions = protocol.get("training_conditions", [])
    allowed = ("flat", "curriculum") if profile == "legacy-v1" else ("flat", "mixed", "curriculum")
    if (
        not isinstance(conditions, list)
        or not conditions
        or any(condition not in allowed for condition in conditions)
        or len(set(conditions)) != len(conditions)
    ):
        raise ValueError(
            "training conditions must be unique supported names for the protocol profile"
        )
    return tuple(f"{condition}_rl" for condition in conditions)


def replay_environment(protocol: dict[str, Any]) -> type:
    """Lazy selection prevents a v1 world-upright replay being labelled as v2."""

    if protocol_profile(protocol) == "tracking-v2":
        from wheel_legged_control.d1.terrain_tracking_env import D1TerrainTrackingEnv

        return D1TerrainTrackingEnv
    from wheel_legged_control.d1.terrain_env import D1TerrainResidualEnv

    return D1TerrainResidualEnv


def representative_cases(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Select the first flat, first bumps and first uphill case in protocol order.

    This function accepts no metrics. Scores, failures and plot appearance
    cannot affect selection. Traces always use training seed 0.
    """

    selected: dict[str, dict[str, Any]] = {}
    for case in protocol["cases"]:
        kind = case["terrain"]["kind"]
        key = "uphill" if kind == "ramp" and case["terrain"]["slope_deg"] > 0 else kind
        if key in ("flat", "bumps", "uphill") and key not in selected:
            selected[key] = case
    if set(selected) != {"flat", "bumps", "uphill"}:
        raise ValueError("protocol needs flat, bumps, and positive-slope representative cases")
    return selected


def load_experiment(run: Path) -> tuple[dict[str, Any], dict[tuple, dict], dict[str, str]]:
    """Validate complete pairing and recompute every RMSE from its telemetry."""

    run = run.resolve()
    protocol_path = run / "protocol.json"
    metrics_path = run / "metrics.csv"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    conditions = protocol_conditions(protocol)
    seeds = protocol.get("training_seeds", [])
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise ValueError("plots require a paired training protocol with unique seeds")
    if 0 not in seeds:
        raise ValueError("fixed representative traces require training seed 0")
    cases = protocol.get("cases", [])
    case_by_id = {case["case_id"]: case for case in cases}
    if not cases or len(case_by_id) != len(cases):
        raise ValueError("protocol case IDs must be nonempty and unique")
    representative_cases(protocol)
    scale = np.asarray(protocol["residual_scale_n"], dtype=float)
    if scale.shape != (2,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("protocol residual_scale_n must contain two finite positive forces")

    rows = _read_csv(
        metrics_path,
        (
            "condition",
            "training_seed",
            "case_id",
            "terrain_kind",
            "environment_seed",
            "telemetry_csv",
            "velocity_rmse_mps",
            "control_steps",
            "completed",
        ),
    )
    indexed: dict[tuple, dict] = {}
    inputs = {"protocol.json": _sha256(protocol_path), "metrics.csv": _sha256(metrics_path)}
    paths_seen: set[Path] = set()
    for row in rows:
        condition = row["condition"]
        seed = None if row["training_seed"] == "" else int(row["training_seed"])
        key = (condition, seed, row["case_id"])
        if key in indexed:
            raise ValueError(f"duplicate paired metric: {key}")
        case = case_by_id.get(row["case_id"])
        if (
            case is None
            or row["terrain_kind"] != case["terrain"]["kind"]
            or int(row["environment_seed"]) != case["environment_seed"]
        ):
            raise ValueError(f"case geometry/seed does not match protocol: {key}")
        telemetry_path = (run / row["telemetry_csv"]).resolve()
        if not telemetry_path.is_relative_to(run) or telemetry_path in paths_seen:
            raise ValueError("telemetry paths must be unique files inside the run directory")
        paths_seen.add(telemetry_path)
        telemetry = _read_csv(telemetry_path, TRACE_FIELDS)
        trace = {
            field: np.asarray([_finite_float(step[field], field) for step in telemetry])
            for field in TRACE_FIELDS
        }
        if condition == "zero_residual" and (
            np.any(trace["action_longitudinal"] != 0.0) or np.any(trace["action_vertical"] != 0.0)
        ):
            raise ValueError("zero-residual baseline log contains a nonzero residual action")
        if int(row["control_steps"]) != len(telemetry) or np.any(np.diff(trace["time_s"]) <= 0):
            raise ValueError(f"invalid telemetry length/time: {key}")
        if not np.allclose(
            trace["velocity_error_mps"],
            trace["forward_velocity_mps"] - trace["command_velocity_mps"],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(f"logged velocity error disagrees with command/state: {key}")
        rmse = float(np.sqrt(np.mean(np.square(trace["velocity_error_mps"]))))
        if not np.isclose(
            rmse, _finite_float(row["velocity_rmse_mps"], "RMSE"), rtol=1e-10, atol=1e-12
        ):
            raise ValueError(f"metrics RMSE disagrees with telemetry: {key}")
        if row["completed"] not in ("0", "1"):
            raise ValueError(f"completed must be 0 or 1: {key}")
        indexed[key] = {"rmse": rmse, "completed": row["completed"] == "1", "trace": trace}
        inputs[str(telemetry_path.relative_to(run))] = _sha256(telemetry_path)
    expected = {("zero_residual", None, case_id) for case_id in case_by_id}
    expected.update(
        (condition, seed, case_id)
        for condition in conditions
        for seed in seeds
        for case_id in case_by_id
    )
    if set(indexed) != expected:
        raise ValueError(
            f"incomplete pairing: missing {len(expected - set(indexed))}, "
            f"unexpected {len(set(indexed) - expected)} episodes"
        )
    return protocol, indexed, inputs


def paired_differences(protocol: dict, indexed: dict) -> list[dict[str, Any]]:
    rows = []
    for condition in protocol_conditions(protocol):
        for seed in protocol["training_seeds"]:
            for case in protocol["cases"]:
                baseline = indexed[("zero_residual", None, case["case_id"])]
                policy = indexed[(condition, seed, case["case_id"])]
                rows.append(
                    {
                        "condition": condition,
                        "training_seed": seed,
                        "case_id": case["case_id"],
                        "terrain_kind": case["terrain"]["kind"],
                        "baseline_rmse_mps": baseline["rmse"],
                        "policy_rmse_mps": policy["rmse"],
                        "rmse_difference_mps": policy["rmse"] - baseline["rmse"],
                        "both_completed": int(baseline["completed"] and policy["completed"]),
                    }
                )
    return rows


def _plot_differences(plt: Any, output: Path, protocol: dict, rows: list[dict]) -> list[dict]:
    conditions = protocol_conditions(protocol)
    seeds = protocol["training_seeds"]
    case_ids = [case["case_id"] for case in protocol["cases"]]
    fig, axes = plt.subplots(
        1,
        len(conditions),
        figsize=(6 * len(conditions), max(5, len(case_ids) * 0.30)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    for axis, condition in zip(axes, conditions, strict=True):
        for index, seed in enumerate(seeds):
            selected = [
                row
                for row in rows
                if row["condition"] == condition and row["training_seed"] == seed
            ]
            offsets = np.arange(len(case_ids)) + (index - (len(seeds) - 1) / 2) * 0.13
            color = f"C{index % 10}"
            axis.scatter(
                [row["rmse_difference_mps"] for row in selected],
                offsets,
                color=color,
                s=28,
                label=f"seed {seed}",
            )
            for row, offset in zip(selected, offsets, strict=True):
                if not row["both_completed"]:
                    axis.scatter(
                        row["rmse_difference_mps"], offset, marker="x", color="black", s=50
                    )
        axis.axvline(0, color="#777777", lw=1)
        axis.set(title=LABELS[condition], xlabel="Velocity RMSE minus zero residual [m/s]")
        axis.set_yticks(np.arange(len(case_ids)), case_ids)
        axis.grid(axis="x", alpha=0.2)
        axis.legend(fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("Paired cases: negative differences mean lower tracking error")
    fig.text(
        0.5,
        0.01,
        "Each episode uses its recorded duration. Black crosses: either paired episode did not complete.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    fig.savefig(output / "paired_case_differences.png", dpi=160)
    plt.close(fig)

    summaries = []
    kinds = [
        kind
        for kind in ("flat", "bumps", "ramp")
        if any(row["terrain_kind"] == kind for row in rows)
    ]
    fig, axis = plt.subplots(figsize=(8, 4.6))
    for condition_index, condition in enumerate(conditions):
        for kind_index, kind in enumerate(kinds):
            for seed_index, seed in enumerate(seeds):
                selected = [
                    row
                    for row in rows
                    if row["condition"] == condition
                    and row["training_seed"] == seed
                    and row["terrain_kind"] == kind
                ]
                mean_difference = float(np.mean([row["rmse_difference_mps"] for row in selected]))
                summaries.append(
                    {
                        "condition": condition,
                        "training_seed": seed,
                        "terrain_kind": kind,
                        "case_count": len(selected),
                        "mean_rmse_difference_mps": mean_difference,
                        "both_completed_case_count": sum(row["both_completed"] for row in selected),
                    }
                )
                offset = (condition_index - (len(conditions) - 1) / 2) * 0.22 + (
                    seed_index - (len(seeds) - 1) / 2
                ) * 0.035
                axis.scatter(
                    kind_index + offset,
                    mean_difference,
                    color=COLORS[condition],
                    s=45,
                    marker="o" if all(row["both_completed"] for row in selected) else "x",
                    label=LABELS[condition] if kind_index == 0 and seed_index == 0 else None,
                )
                axis.annotate(
                    str(seed),
                    (kind_index + offset, mean_difference),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=8,
                )
    axis.axhline(0, color="#777777", lw=1)
    axis.set(
        xticks=np.arange(len(kinds)),
        xticklabels=kinds,
        ylabel="Mean paired velocity RMSE difference [m/s]",
        title="One point per training seed and terrain family",
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    fig.text(
        0.5,
        0.01,
        "Equal-weight cases within each seed; labels are training seeds. No confidence interval. Cross: incomplete pair(s).",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output / "terrain_seed_differences.png", dpi=160)
    plt.close(fig)
    return summaries


def _plot_traces(plt: Any, output: Path, protocol: dict, indexed: dict, selected: dict) -> None:
    scales = protocol["residual_scale_n"]
    for key in ("bumps", "uphill"):
        case = selected[key]
        fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
        baseline_trace = indexed[("zero_residual", None, case["case_id"])]["trace"]
        for axis, target in zip(
            axes[:2], ("command_velocity_mps", "command_clearance_m"), strict=True
        ):
            axis.plot(
                baseline_trace["time_s"],
                baseline_trace[target],
                "--",
                color="black",
                lw=1,
                label="command",
            )
        for condition in ("zero_residual", *protocol_conditions(protocol)):
            episode = indexed[
                (condition, None if condition == "zero_residual" else 0, case["case_id"])
            ]
            trace = episode["trace"]
            values = (
                trace["forward_velocity_mps"],
                trace["clearance_m"],
                scales[0] * np.clip(trace["action_longitudinal"], -1, 1),
                scales[1] * np.clip(trace["action_vertical"], -1, 1),
            )
            for axis, value in zip(axes, values, strict=True):
                label = LABELS[condition] + (" (incomplete)" if not episode["completed"] else "")
                axis.plot(trace["time_s"], value, color=COLORS[condition], label=label, lw=1.3)
        for axis, label in zip(
            axes,
            (
                "Velocity [m/s]",
                "Clearance [m]",
                "Longitudinal residual [N]",
                "Vertical residual [N]",
            ),
            strict=True,
        ):
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
        axes[0].legend(fontsize=8, loc="best")
        axes[-1].set_xlabel("Simulation time [s]")
        fig.suptitle(f"{case['case_id']} | fixed training seed 0")
        fig.text(
            0.5,
            0.01,
            "Residuals are clipped action × declared force scale, not measured contact forces. Lines stop at episode end.",
            ha="center",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.035, 1, 0.97))
        fig.savefig(output / f"representative_{key}.png", dpi=160)
        plt.close(fig)


def _scene_profile(plant: Any, terrain: dict) -> tuple[np.ndarray, np.ndarray, str]:
    """Read a metre-wide section on either side of the base from collision data."""

    relative_x = np.linspace(-1.0, 1.0, 401)
    base_x, base_y = plant.base_position[:2]
    center_height = plant.training_ground_reference(float(base_x), float(base_y)).height_m
    height_mm = np.asarray(
        [
            1000.0
            * (
                plant.training_ground_reference(float(base_x + x), float(base_y)).height_m
                - center_height
            )
            for x in relative_x
        ]
    )
    if terrain["kind"] == "ramp":
        description = f"ramp {terrain['slope_deg']:+.1f} deg (+x uphill for positive angles)"
    elif terrain["kind"] == "bumps":
        description = (
            f"bumps: A = {1000 * terrain['amplitude_m']:.1f} mm, "
            f"wavelength = {terrain.get('wavelength_m', 0.8):.2f} m"
        )
    else:
        description = "flat: h(x) = 0"
    return relative_x, height_mm, description


def _scene_profile_limit_mm(selected: dict) -> float:
    """Use one common vertical scale, determined only by declared geometry."""

    limits = [10.0]
    for case in selected.values():
        terrain = case["terrain"]
        limits.extend(
            (
                2000.0 * terrain.get("amplitude_m", 0.0),
                1000.0 * abs(np.tan(np.deg2rad(terrain.get("slope_deg", 0.0)))),
            )
        )
    return float(10.0 * np.ceil(max(limits) / 10.0))


def _render_scenes(plt: Any, output: Path, protocol: dict, selected: dict) -> list[dict]:
    import mujoco

    environment = replay_environment(protocol)
    profile = protocol_profile(protocol)
    records = []
    profile_limit_mm = _scene_profile_limit_mm(selected)
    for key in ("flat", "bumps", "uphill"):
        case = selected[key]
        env = environment(
            baseline="lqr",
            episode_seconds=protocol["episode_seconds"],
            training_mode="flat",
            randomize=False,
        )
        renderer = None
        try:
            _, info = env.reset(
                seed=case["environment_seed"],
                options={name: case[name] for name in ("terrain", "velocity_mps", "height_m")},
            )
            # Fixed one-second endpoint, or the actual earlier episode ending.
            count = min(100, round(protocol["episode_seconds"] / protocol["control_dt_s"]))
            terminated = truncated = False
            for _ in range(count):
                _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
                if terminated or truncated:
                    break
            # Presentation-only changes happen after the final physics step.
            # Remove the dark floor material; no collision, inertia or pose data
            # changes. A fixed side camera and common profile scales make these
            # small surfaces comparable without magnifying their geometry.
            floor_id = env.plant.floor_geom_id
            env.plant.model.geom_matid[floor_id] = -1
            env.plant.model.geom_rgba[floor_id] = (0.69, 0.73, 0.76, 1.0)
            env.plant.model.light_ambient[:] = (0.25, 0.25, 0.25)
            env.plant.model.light_diffuse[:] = (0.55, 0.55, 0.55)
            env.plant.model.vis.headlight.ambient[:] = (0.45, 0.45, 0.45)
            env.plant.model.vis.headlight.diffuse[:] = (0.45, 0.45, 0.45)
            env.plant.model.vis.headlight.specular[:] = (0.05, 0.05, 0.05)
            relative_x, height_mm, description = _scene_profile(env.plant, case["terrain"])
            # Create only after this case's heightfield update: its GPU copy is fresh.
            renderer = mujoco.Renderer(env.plant.model, width=960, height=540)
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.distance, camera.azimuth, camera.elevation = 2.2, 95.0, -12.0
            camera.lookat[:] = env.plant.base_position + np.asarray((0.12, 0.0, -0.15))
            renderer.update_scene(env.plant.data, camera=camera)
            pixels = renderer.render()
            fig, (axis, profile_axis) = plt.subplots(
                2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}
            )
            axis.imshow(pixels)
            axis.axis("off")
            axis.set_title(
                f"Zero-residual baseline replay | {case['case_id']}\n"
                f"{description}\nt = {env.plant.data.time:.2f} s; {profile}; no trained policy loaded"
            )
            profile_axis.plot(relative_x, height_mm, color="#2366a0", lw=1.8)
            profile_axis.axhline(0, color="#8c959d", lw=0.8)
            profile_axis.axvline(0, color="#8c959d", lw=0.8, linestyle="--")
            profile_axis.set(
                xlim=(-1.0, 1.0),
                ylim=(-profile_limit_mm, profile_limit_mm),
                xticks=np.arange(-1.0, 1.01, 0.25),
                xlabel="World-x offset from base [m]",
                ylabel="h(x) - h(base) [mm]",
                title="Actual collision profile; shared scales across the three scenes",
            )
            profile_axis.grid(alpha=0.25)
            fig.tight_layout()
            filename = f"baseline_scene_{key}.png"
            fig.savefig(output / filename, dpi=140)
            plt.close(fig)
            records.append(
                {
                    "case_id": case["case_id"],
                    "file": filename,
                    "replayed_time_s": float(env.plant.data.time),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "termination_reason": info["termination_reason"],
                    "scope": "new zero-residual baseline simulation; not trained-policy footage",
                    "profile": profile,
                    "control_schema": getattr(env, "control_schema", None),
                    "observation_schema": env.observation_schema,
                    "reward_schema": env.reward_schema,
                    "terrain_description": description,
                    "camera": {"distance_m": 2.2, "azimuth_deg": 95.0, "elevation_deg": -12.0},
                    "profile_x_limits_m": [-1.0, 1.0],
                    "profile_relative_height_limits_mm": [-profile_limit_mm, profile_limit_mm],
                    "presentation": "lighter floor and lighting applied after simulation; collision geometry unchanged",
                }
            )
        finally:
            if renderer is not None:
                renderer.close()
            env.close()
    return records


def run_visualization(run: Path, output: Path, *, render_scenes: bool = False) -> Path:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    protocol, indexed, input_hashes = load_experiment(run)
    selected = representative_cases(protocol)
    rows = paired_differences(protocol, indexed)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=False)
    with (output / "paired_differences.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = _plot_differences(plt, output, protocol, rows)
    _plot_traces(plt, output, protocol, indexed, selected)
    scenes = _render_scenes(plt, output, protocol, selected) if render_scenes else []
    _write_json(
        output / "selection.json",
        {
            "training_seed": 0,
            "profile": protocol_profile(protocol),
            "training_conditions": list(protocol["training_conditions"]),
            "rule": "first flat, first bumps, first positive-slope ramp in protocol order; never selected by score",
            "case_ids": {key: case["case_id"] for key, case in selected.items()},
            "scene_replays": scenes,
        },
    )
    _write_json(
        output / "terrain_seed_summary.json",
        {
            "replication_unit": "training seed; equal-weight evaluation cases within terrain family",
            "baseline_replication": "one shared deterministic baseline per case, not new seed replicates",
            "uncertainty": "individual seed points only; no confidence intervals",
            "rows": summaries,
        },
    )
    source_paths = [Path(__file__)]
    if render_scenes:
        source_paths.extend(
            path
            for path in (ROOT / "src" / "wheel_legged_control").rglob("*")
            if path.is_file() and path.suffix.lower() in {".py", ".xml", ".urdf", ".stl"}
        )
    _write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "input_sha256": input_hashes,
            "source_sha256": {
                str(path.relative_to(ROOT)): _sha256(path) for path in sorted(source_paths)
            },
            "sha256": {
                str(path.relative_to(output)): _sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
            "excludes": ["manifest.json", "files added after visualization"],
        },
    )
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new visualization directory")
    parser.add_argument(
        "--render-scenes", action="store_true", help="render fixed zero-residual baseline replays"
    )
    args = parser.parse_args(argv)
    try:
        destination = run_visualization(args.run, args.output, render_scenes=args.render_scenes)
    except (ValueError, FileExistsError) as error:
        raise SystemExit(str(error)) from error
    print(f"Plots: {destination}")


if __name__ == "__main__":
    main()
