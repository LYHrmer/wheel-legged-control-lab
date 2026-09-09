"""Record one real continuous D1 rollout, including every simulated state."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from wheel_legged_control.d1.continuous_task import (
    STAGES,
    TASK_SCHEMA,
    ContinuousTaskConfig,
    D1ContinuousTask,
    summarize_task,
)

ROOT = Path(__file__).resolve().parents[1]
ACTION_REGISTRY = {
    "d1-terrain-residual-quarter-v1": {
        "shape": (2,),
        "scale_n": [11.25, 20.0],
        "mapping": "[ax,az]",
    },
    "d1-terrain-residual-longitudinal-only-v1": {
        "shape": (1,),
        "scale_n": [11.25],
        "mapping": "[ax,0]",
    },
}


def adapt_policy_action(action, schema: str) -> np.ndarray:
    if schema not in ACTION_REGISTRY:
        raise ValueError("unregistered checkpoint action schema")
    action = np.asarray(action, dtype=np.float64)
    if action.shape != ACTION_REGISTRY[schema]["shape"] or not np.isfinite(action).all():
        raise ValueError("policy action differs from its registered shape")
    if schema == "d1-terrain-residual-longitudinal-only-v1":
        return np.asarray((action[0], 0.0))
    return action.copy()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def store_compiled_model(model, directory: Path) -> dict:
    """Store one content-addressed, lossless model shared by sibling runs."""
    with TemporaryDirectory(prefix="d1-continuous-model-") as temporary:
        path = Path(temporary) / "model.mjb"
        mujoco.mj_saveModel(model, str(path), None)
        raw = path.read_bytes()
    content_sha = hashlib.sha256(raw).hexdigest()
    path = directory.parent / "models" / f"{content_sha}.mjb.gz"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        if hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest() != content_sha:
            raise ValueError("existing shared model has wrong decompressed content")
    else:
        with path.open("xb") as stream:
            stream.write(gzip.compress(raw, compresslevel=9, mtime=0))
    return {
        "path": f"../models/{path.name}",
        "sha256": sha256(path),
        "uncompressed_sha256": content_sha,
        "compression": "gzip",
    }


def validate_checkpoint(
    metadata: dict, model, task: D1ContinuousTask, path: Path, *, allow_v2_transfer: bool = False
) -> None:
    """Check declared semantics as well as dimensions, before any policy rollout."""

    for name in ("observation_schema", "control_schema"):
        if metadata.get(name) != getattr(task, name):
            if (
                allow_v2_transfer
                and task.config.ground_reference_mode == "oracle"
                and name == "control_schema"
                and metadata.get(name) == "d1-lqr-vmc-local-tangent-v2"
            ):
                continue
            raise ValueError(f"incompatible checkpoint {name}")
    if metadata.get("model_sha256") != sha256(path):
        raise ValueError("checkpoint SHA256 does not match its metadata")
    if model.observation_space.shape != task.observation_space.shape:
        raise ValueError("checkpoint observation shape differs")
    action = ACTION_REGISTRY.get(metadata.get("action_schema"))
    if action is None:
        raise ValueError("unregistered checkpoint action schema")
    if model.action_space.shape != action["shape"]:
        raise ValueError("checkpoint action shape differs from registered schema")
    scale = metadata.get("residual_force_scale_n")
    legacy_v2 = metadata.get("control_schema") == "d1-lqr-vmc-local-tangent-v2"
    if scale is None:
        if not (allow_v2_transfer and legacy_v2):
            raise ValueError(
                "checkpoint lacks residual_force_scale_n; known v2 needs explicit OOD permission"
            )
    elif scale != action["scale_n"]:
        raise ValueError("checkpoint residual force scale differs from registered schema")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path, help="new directory; never overwritten"
    )
    parser.add_argument("--mode", choices=("zero", "policy"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--yaw-controller", choices=("pi", "legacy-p"), default="pi")
    parser.add_argument("--yaw-kp", type=float, default=2.0)
    parser.add_argument("--yaw-ki", type=float, default=3.0)
    parser.add_argument("--state-source", choices=("oracle", "sensor"), default="oracle")
    parser.add_argument(
        "--observation-layout", choices=("legacy44", "command45"), default="legacy44"
    )
    parser.add_argument(
        "--allow-oracle-v2-task-transfer",
        action="store_true",
        help="explicit OOD transfer: old v2 layout, new yaw PI and long task",
    )
    return parser


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.mode == "policy" and args.checkpoint is None:
        raise ValueError("policy mode requires --checkpoint")
    if args.mode == "zero" and (args.checkpoint or args.metadata):
        raise ValueError("zero mode does not accept checkpoint files")
    config = ContinuousTaskConfig(
        duration_s=args.duration,
        ground_reference_mode="estimated" if args.state_source == "sensor" else "oracle",
        observation_layout=args.observation_layout,
        yaw_controller=args.yaw_controller,
        yaw_kp=args.yaw_kp,
        yaw_ki=args.yaw_ki,
    )
    factory = None
    if args.state_source == "sensor":
        from functools import partial

        from wheel_legged_control.d1.sensor_estimation import D1SensorStateSource

        factory = partial(
            D1SensorStateSource, initial_position=(-3.8, 0.0, 0.455), initial_rpy=(0.0, 0.0, 0.0)
        )
    task = D1ContinuousTask(config, state_source_factory=factory)
    model = None
    metadata = None
    if args.mode == "policy":
        import torch
        from stable_baselines3 import PPO

        torch.set_num_threads(1)
        metadata_path = args.metadata or args.checkpoint.with_name("metadata.json")
        metadata = json.loads(metadata_path.read_text())
        model = PPO.load(args.checkpoint, device="cpu")
        validate_checkpoint(
            metadata,
            model,
            task,
            args.checkpoint,
            allow_v2_transfer=args.allow_oracle_v2_task_transfer,
        )
    observation, _ = task.reset(seed=args.seed)
    args.output.mkdir(parents=True)
    # Exact compiled geometry/physics accompanies the states. A future source
    # change cannot silently alter the road in the renderer.
    compiled_model = store_compiled_model(task.plant.model, args.output)
    protocol = {
        "task_schema": TASK_SCHEMA,
        "config": asdict(config),
        "mode": args.mode,
        "seed": args.seed,
        "episode_reset_count": 1,
        "stages_45_second_template": STAGES,
        "road": task.road,
        "control_dt_s": task.plant.control_dt,
        "state_source": args.state_source,
        "observation_schema": task.observation_schema,
        "action_schema": task.action_schema,
        "residual_force_scale_n": [11.25, 20.0],
        "checkpoint_action_mapping": ACTION_REGISTRY[metadata["action_schema"]]["mapping"]
        if metadata
        else None,
        "control_schema": task.control_schema,
        "allow_oracle_v2_task_transfer": args.allow_oracle_v2_task_transfer,
        "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None,
        "checkpoint_metadata": metadata,
        "compiled_model_sha256": compiled_model["uncompressed_sha256"],
        "interpretation": (
            "actual deterministic policy rollout; stops, yaw and continuous road are task shifts "
            "from the old forward-only 4 s oracle-v2 training"
            if model is not None
            else "actual zero-residual LQR/VMC baseline rollout; no PPO policy"
        ),
        "state_recording": "actual post-step qpos/qvel plus initial state; no resimulation",
    }
    write_json(args.output / "protocol.json", protocol)
    rows = []
    qpos, qvel = task.plant.simulation_state()
    positions, velocities, times = [qpos.copy()], [qvel.copy()], [0.0]
    for _ in range(task.max_steps):
        action = np.zeros(2)
        if model is not None:
            action = adapt_policy_action(
                model.predict(observation, deterministic=True)[0], metadata["action_schema"]
            )
        observation, reward, terminated, truncated, info = task.step(action)
        rows.append({**info, "reward": reward})
        qpos, qvel = task.plant.simulation_state()
        positions.append(qpos.copy())
        velocities.append(qvel.copy())
        times.append(info["time_s"])
        if terminated or truncated:
            break
    with (args.output / "telemetry.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output / "states.npz", time_s=times, qpos=positions, qvel=velocities)
    summary = summarize_task(rows)
    write_json(args.output / "summary.json", summary)
    sources = [Path(__file__), ROOT / "scripts/render_d1_continuous_task.py"]
    sources.extend((ROOT / "src/wheel_legged_control/d1").glob("*.py"))
    write_json(
        args.output / "manifest.json",
        {
            "schema": "d1-continuous-rollout-artifact-v1",
            "compiled_model": compiled_model,
            "sha256": {
                path.name: sha256(path) for path in sorted(args.output.iterdir()) if path.is_file()
            },
            "source_sha256": {
                str(path.relative_to(ROOT)): sha256(path)
                for path in sorted(sources)
                if path.exists()
            },
        },
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), indent=2))
