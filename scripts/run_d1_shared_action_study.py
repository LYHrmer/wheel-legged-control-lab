"""Run the preregistered same-controller shared2/independent8 comparison.

This standard-library orchestrator does not choose checkpoints or score cases.
It runs one child at a time and stops on errors or source changes. Successful
commands are not evidence that tracking quality passed. --smoke runs a separate
two-model interface/reload check, never a shortened performance experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "scripts/run_d1_shared_action_study.py"
CHILD_SCRIPT = "scripts/run_d1_locomotion_experiment.py"
ACTION_MODES = ("shared2", "independent8")
TRAINING_SEEDS = (31000, 32000, 33000)
THREAD_ENV = {
    key: "1"
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    )
}


def build_runs(output: Path, *, smoke: bool = False) -> list[dict]:
    """Build the entire ordered protocol without reading models or writing files."""
    output = Path(output).resolve()
    seeds = (31001,) if smoke else TRAINING_SEEDS
    steps, duration = (512, "0.2") if smoke else (32768, "60.0")
    if steps % (4 * 128):
        raise ValueError("budget must contain whole 4-worker, 128-step rollouts")
    common = [
        "--baseline",
        "wheel_leg",
        "--source",
        "imu_encoder_fusion",
        "--history",
        "1",
        "--measurement-delay",
        "0",
        "--workers",
        "4",
        "--steps",
        str(steps),
        "--duration",
        duration,
        "--terrain-suite",
        "action_compare_v1",
        "--wheel-kp",
        "0.55",
        "--wheel-ki",
        "1.5",
        "--yaw-feedback-gain",
        "4",
        "--leg-feedback-scale",
        "1",
        "--attitude-feedback-scale",
        "0.25",
    ]

    def entry(name, mode, action_mode, seed, split=None, model_name=None):
        command = [
            sys.executable,
            str(ROOT / CHILD_SCRIPT),
            mode,
            "--output",
            str(output / name),
            *common,
            "--action-mode",
            action_mode,
            "--seed",
            str(seed),
        ]
        if mode == "evaluate":
            command.extend(["--split", split])
        if model_name is not None:
            command.extend(
                [
                    "--policy",
                    str(output / model_name / "checkpoint.zip"),
                    "--metadata",
                    str(output / model_name / "checkpoint.json"),
                ]
            )
        return {
            "name": name,
            "mode": mode,
            "action_mode": action_mode,
            "seed": seed,
            "split": split,
            "command": command,
        }

    models = [(mode, seed, f"{mode}_seed{seed}") for mode in ACTION_MODES for seed in seeds]
    runs = [entry("zero_development", "evaluate", "independent8", 31000, "development")]
    for mode, seed, name in models:
        runs.append(entry(name, "train", mode, seed))
        runs.append(entry(name + "_development", "evaluate", mode, seed, "development", name))
    runs.append(entry("zero_holdout", "evaluate", "independent8", 31000, "holdout"))
    for mode, seed, name in models:
        runs.append(entry(name + "_holdout", "evaluate", mode, seed, "holdout", name))
    return runs


def source_hashes() -> dict[str, str]:
    """Hash the complete source inventory; added/deleted files also change it."""
    paths = {ROOT / SCRIPT, ROOT / CHILD_SCRIPT, ROOT / "pyproject.toml"}
    source_entries = list((ROOT / "src").rglob("*"))
    if any(path.is_symlink() for path in source_entries):
        raise ValueError("source tree contains a symlink, including a directory alias")
    paths.update(path for path in source_entries if path.suffix == ".py")
    assets = ROOT / "src/wheel_legged_control/d1/assets"
    paths.update(path for path in assets.rglob("*") if path.is_file())
    if not (ROOT / "src").is_dir() or not assets.is_dir():
        raise ValueError("source/assets directory is missing")
    result = {}
    for path in sorted(paths):
        if not path.is_file() or any(parent.is_symlink() for parent in (path, *path.parents)):
            raise ValueError(f"source must be a regular non-symlink file: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result[str(path.relative_to(ROOT))] = digest.hexdigest()
    return result


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _output_path(output):
    raw = Path(output).absolute()
    if any(path.is_symlink() for path in (raw, *raw.parents)):
        raise ValueError("output must not be a symlink or have symlink parents")
    output = raw.resolve()
    if output.exists():
        raise ValueError("output exists; choose a new directory, including after a failed run")
    if any(output.is_relative_to(ROOT / part) for part in ("src", "scripts")):
        raise ValueError("output overlaps experiment source")
    return output


def run(output: Path, *, smoke: bool = False) -> dict:
    output = _output_path(output)
    runs = build_runs(output, smoke=smoke)
    before = source_hashes()  # Preflight before creating even a partial output.
    environment = os.environ.copy()
    environment.update(THREAD_ENV)
    python_paths = [ROOT / "src", ROOT]
    if (ROOT / ".local-deps").is_dir():
        python_paths.append(ROOT / ".local-deps")
    environment["PYTHONPATH"] = os.pathsep.join(map(str, python_paths))
    protocol = {
        "schema": "d1-shared-action-study-v1",
        "kind": "smoke" if smoke else "formal",
        "created_utc": _utc(),
        "training_seeds": [31001] if smoke else list(TRAINING_SEEDS),
        "action_modes": list(ACTION_MODES),
        "timesteps_per_model": 512 if smoke else 32768,
        "duration_s": 0.2 if smoke else 60.0,
        "workers": 4,
        "history": 1,
        "source": "imu_encoder_fusion",
        "measurement_delay_steps": 0,
        "delay_randomization": False,
        "terrain_suite": "action_compare_v1",
        "controller_parameters": {
            "wheel_kp": 0.55,
            "wheel_ki": 1.5,
            "yaw_feedback_gain": 4.0,
            "leg_feedback_scale": 1.0,
            "attitude_feedback_scale": 0.25,
        },
        "ppo_settings": "unchanged child runner defaults, recorded in every child protocol",
        "evaluation_reset_seeds": {"development": [1017, 1029], "holdout": [1617, 1629]},
        "evaluation_seed_note": "--seed links a model; actual evaluation reset seeds belong to the terrain suite",
        "evaluation_budget_note": "--steps records the associated training budget; evaluation is duration-limited and does not train",
        "selection_rule": "final fixed-budget checkpoints only; all development commands precede holdout",
        "zero_reference": "one independent8 zero policy per split; no trained replicate",
        "interpretation": "interface/reload check only"
        if smoke
        else "paired same-controller action-structure comparison",
        "working_directory": str(ROOT),
        "python_executable": sys.executable,
        "child_environment_overrides": {**THREAD_ENV, "PYTHONPATH": environment["PYTHONPATH"]},
        "runs": runs,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", protocol)
    write_json(
        output / "source.json",
        {
            "sha256": before,
            "scope": "all src Python files/assets plus child runner, orchestrator and pyproject; child runs own source archives",
        },
    )
    (output / "logs").mkdir()
    completed = 0
    started = time.perf_counter()
    try:
        with (output / "runs.jsonl").open("x", encoding="utf-8") as ledger:
            for item in runs:
                if source_hashes() != before:
                    raise RuntimeError("source changed before the next command; stopped")
                record = {
                    "name": item["name"],
                    "command": item["command"],
                    "started_utc": _utc(),
                    "log": f"logs/{item['name']}.log",
                }
                print(
                    f"[{completed + 1}/{len(runs)}] {item['name']} -> {record['log']}", flush=True
                )
                tick = time.perf_counter()
                try:
                    with (output / record["log"]).open("x", encoding="utf-8") as stream:
                        result = subprocess.run(
                            item["command"],
                            cwd=ROOT,
                            env=environment,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            check=False,
                        )
                    record["returncode"] = result.returncode
                except (Exception, KeyboardInterrupt) as exc:
                    record.update(returncode=None, error_type=type(exc).__name__, error=str(exc))
                    raise
                finally:
                    record.update(ended_utc=_utc(), elapsed_s=time.perf_counter() - tick)
                    ledger.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                    ledger.flush()
                if result.returncode:
                    raise RuntimeError(
                        f"{item['name']} exited {result.returncode}; inspect {record['log']}"
                    )
                if source_hashes() != before:
                    raise RuntimeError(f"source changed during {item['name']}; stopped")
                completed += 1
                print(
                    f"[{completed}/{len(runs)}] command exited 0 ({record['elapsed_s']:.1f}s)",
                    flush=True,
                )
        after = source_hashes()
        if after != before:
            raise RuntimeError("source changed at the final gate")
        write_json(output / "source_consistency.json", {"unchanged": True, "sha256": after})
    except (Exception, KeyboardInterrupt) as exc:
        try:
            after = source_hashes()
            consistency = {
                "unchanged": after == before,
                "sha256": after,
                "changed": sorted(
                    name for name in set(before) | set(after) if before.get(name) != after.get(name)
                ),
            }
        except (ValueError, OSError, KeyboardInterrupt) as source_error:
            consistency = {"unchanged": False, "error": str(source_error)}
        write_json(output / "source_consistency.json", consistency)
        write_json(
            output / "failure.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "commands_completed": completed,
                "planned_commands": len(runs),
                "elapsed_s": time.perf_counter() - started,
                "note": "partial outputs retained; no retry or overwrite",
            },
        )
        raise
    summary = {
        "status": "commands_completed",
        "kind": protocol["kind"],
        "commands_completed": completed,
        "planned_commands": len(runs),
        "elapsed_s": time.perf_counter() - started,
        "quality_claim": "none; inspect child metrics and independently audit trajectories",
    }
    write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    output = args.output or ROOT / "results" / (
        "d1_shared_action_smoke" if args.smoke else "d1_shared_action_study"
    )
    try:
        summary = run(output, smoke=args.smoke)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"shared-action study failed: {exc}\n")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    main()
