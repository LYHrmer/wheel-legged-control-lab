"""Run a fixed checkpoint-budget ladder on six continuous PPO training trajectories.

All training finishes before development evaluation; all development finishes
before holdout. Every preregistered checkpoint is evaluated, without best-model
selection. --smoke checks the interface only. --dry-run prints the complete
protocol without creating output, importing simulation libraries or loading models.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "scripts/run_d1_budget_study.py"
CHILD_SCRIPT = "scripts/run_d1_locomotion_experiment.py"
ACTION_MODES = ("shared2", "independent8")
TRAINING_SEEDS = (31000, 32000, 33000)
CHECKPOINT_BUDGETS = (32768, 65536, 131072, 262144)
PPO_SETTINGS = {
    "learning_rate": 3e-4,
    "n_steps": 128,
    "batch_size": 128,
    "n_epochs": 4,
    "gamma": math.exp(-0.01 / 2.0),
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "policy_kwargs": {"net_arch": [64, 64], "log_std_init": -2.0},
}
TERRAIN_FIELDS = (
    "layout",
    "slope_deg",
    "cross_slope_deg",
    "ripple_amplitude_m",
    "wavelength_x_m",
    "wavelength_y_m",
    "phase_x_rad",
    "phase_y_rad",
    "step_height_m",
)
TERRAIN_ROWS = {
    "train": (
        ("straight", 1.0, 0.5, 0.005, 0.6, 0.9, 0.0, 0.2, 0.005),
        ("straight", -1.5, -0.5, 0.01, 0.9, 1.2, 0.6, 0.8, 0.01),
        ("left_offset", 1.5, -1.0, 0.0075, 0.6, 1.2, 0.6, 0.2, 0.0075),
        ("left_offset", -1.0, 1.0, 0.005, 0.9, 0.9, 0.0, 0.8, 0.005),
    ),
    "development": (
        ("right_offset", 1.30, -0.70, 0.0065, 0.77, 1.07, 0.45, 0.65, 0.0065),
        ("right_offset", -1.30, 0.70, 0.0070, 0.87, 1.17, 1.05, 1.25, 0.0070),
    ),
    "holdout": (
        ("s_bend", 1.20, 0.55, 0.0080, 0.74, 1.04, 1.70, 1.90, 0.0080),
        ("diagonal", -1.20, -0.55, 0.0075, 0.84, 1.14, 2.30, 2.50, 0.0075),
    ),
}
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
LEDGER_FIELDS = (
    "sequence",
    "name",
    "mode",
    "action_mode",
    "seed",
    "budget",
    "split",
    "phase",
    "model_name",
    "checkpoint_id",
    "started_utc",
    "ended_utc",
    "elapsed_s",
    "returncode",
    "log",
    "error_type",
    "error",
    "command_json",
    "child_artifact_sha256_json",
)


def _output_path(output: Path) -> Path:
    raw = Path(output).absolute()
    if ".." in raw.parts or any(path.is_symlink() for path in (raw, *raw.parents)):
        raise ValueError("output must not contain parent traversal or symlink aliases")
    output = raw.resolve()
    if output.exists():
        raise ValueError("output exists; choose a new directory, including after a failed run")
    if any(output.is_relative_to(ROOT / part) for part in ("src", "scripts", "tests")):
        raise ValueError("output overlaps experiment source")
    return output


def build_runs(output: Path, *, smoke: bool = False) -> list[dict]:
    """Construct ordered argv; checkpoint files intentionally need not exist yet."""
    output = _output_path(output)
    seeds = (31001,) if smoke else TRAINING_SEEDS
    budgets = (512, 1024) if smoke else CHECKPOINT_BUDGETS
    duration = "0.2" if smoke else "60.0"
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
        "--duration",
        duration,
        "--terrain-suite",
        "budget_compare_v1",
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

    def entry(name, mode, action_mode, seed, budget, *, split=None, model_name=None):
        argv = [
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
            "--steps",
            str(budget or budgets[-1]),
        ]
        if mode == "train":
            argv.extend(["--checkpoint-budgets", *map(str, budgets)])
        else:
            argv.extend(["--split", split])
        if model_name is not None:
            checkpoint = output / model_name / "checkpoints" / f"budget{budget}"
            argv.extend(
                [
                    "--policy",
                    str(checkpoint / "checkpoint.zip"),
                    "--metadata",
                    str(checkpoint / "checkpoint.json"),
                ]
            )
        return {
            "name": name,
            "mode": mode,
            "action_mode": action_mode,
            "seed": seed,
            "budget": budget,
            "split": split,
            "command": argv,
            "model_name": name if mode == "train" else model_name,
            "checkpoint_id": f"{model_name}_budget{budget}" if model_name else None,
            "phase": "train" if mode == "train" else split,
        }

    models = [(mode, seed, f"{mode}_seed{seed}") for mode in ACTION_MODES for seed in seeds]
    runs = [entry("zero_development", "evaluate", "independent8", 31000, None, split="development")]
    runs.extend(entry(name, "train", mode, seed, budgets[-1]) for mode, seed, name in models)
    for split in ("development", "holdout"):
        if split == "holdout":
            runs.append(entry("zero_holdout", "evaluate", "independent8", 31000, None, split=split))
        runs.extend(
            entry(
                f"{name}_budget{budget}_{split}",
                "evaluate",
                mode,
                seed,
                budget,
                split=split,
                model_name=name,
            )
            for mode, seed, name in models
            for budget in budgets
        )
    return runs


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(THREAD_ENV)
    python_paths = [ROOT / "src", ROOT]
    if (ROOT / ".local-deps").is_dir():
        python_paths.append(ROOT / ".local-deps")
    environment["PYTHONPATH"] = os.pathsep.join(map(str, python_paths))
    return environment


def build_protocol(output: Path, *, smoke: bool = False) -> dict:
    runs = build_runs(output, smoke=smoke)
    seeds = [31001] if smoke else list(TRAINING_SEEDS)
    budgets = [512, 1024] if smoke else list(CHECKPOINT_BUDGETS)
    environment = _environment()
    evaluations = sum(item["mode"] == "evaluate" for item in runs)
    models = [
        {
            "model_id": item["name"],
            "action_mode": item["action_mode"],
            "seed": item["seed"],
            "training_run": item["name"],
        }
        for item in runs
        if item["mode"] == "train"
    ]
    checkpoints = [
        {
            "checkpoint_id": item["checkpoint_id"],
            "model_id": item["model_name"],
            "budget": item["budget"],
            "training_run": item["model_name"],
            "relative_zip": f"{item['model_name']}/checkpoints/budget{item['budget']}/checkpoint.zip",
            "relative_metadata": f"{item['model_name']}/checkpoints/budget{item['budget']}/checkpoint.json",
            "expected_update_index": item["budget"] // (4 * PPO_SETTINGS["n_steps"]) - 1,
        }
        for item in runs
        if item["split"] == "development" and item["checkpoint_id"]
    ]
    return {
        "schema": "d1-budget-study-v1",
        "kind": "smoke" if smoke else "formal",
        "training_seeds": seeds,
        "action_modes": list(ACTION_MODES),
        "checkpoint_budgets": budgets,
        "total_timesteps_per_model": budgets[-1],
        "training_mode": "single_continuous_learn",
        "n_steps": PPO_SETTINGS["n_steps"],
        "models": models,
        "checkpoints": checkpoints,
        "continuous_training_trajectories": len(seeds) * len(ACTION_MODES),
        "checkpoint_count": len(seeds) * len(ACTION_MODES) * len(budgets),
        "planned_commands": len(runs),
        "planned_evaluation_cases": evaluations * 4,
        "total_training_transitions": len(models) * budgets[-1],
        "expected_train_calls": len(models) * budgets[-1] // (4 * PPO_SETTINGS["n_steps"]),
        "duration_s": 0.2 if smoke else 60.0,
        "workers": 4,
        "history": 1,
        "source": "imu_encoder_fusion",
        "measurement_delay_steps": 0,
        "delay_randomization": False,
        "terrain_suite": "budget_compare_v1",
        "terrain_counts": {"train": 4, "development": 2, "holdout": 2},
        "evaluation_seeds": {"development": [1017, 1029], "holdout": [4617, 4629]},
        "terrains": {
            split: [
                {**dict(zip(TERRAIN_FIELDS, row)), "schema": "d1-fixed-2d-road-v1"} for row in rows
            ]
            for split, rows in TERRAIN_ROWS.items()
        },
        "terrain_scope": "training roads unchanged; action_compare_v1 development reused; new holdout parameter instances, not new layout families",
        "controller_parameters": {
            "wheel_kp": 0.55,
            "wheel_ki": 1.5,
            "yaw_feedback_gain": 4.0,
            "leg_feedback_scale": 1.0,
            "attitude_feedback_scale": 0.25,
        },
        "ppo_settings": copy.deepcopy(PPO_SETTINGS),
        "evaluation_seed_note": "--seed identifies a training trajectory; evaluation reset seeds belong to the terrain suite",
        "evaluation_budget_note": "evaluation --steps identifies its saved checkpoint budget, not additional training; zero has no training budget",
        "selection_rule": "evaluate every fixed checkpoint; all training then all development then holdout; no best checkpoint selection",
        "zero_reference": "one independent8 zero policy per split, not a trained replicate",
        "checkpoint_dependence": "budgets along one training trajectory are dependent checkpoints, not independent training runs",
        "interpretation": "interface/reload check only"
        if smoke
        else "paired budget ladder on continuous same-controller training trajectories",
        "working_directory": str(ROOT),
        "python_executable": sys.executable,
        "child_environment_overrides": {**THREAD_ENV, "PYTHONPATH": environment["PYTHONPATH"]},
        "runs": runs,
    }


def source_hashes() -> dict[str, str]:
    """Detect edits, additions and removals; exclude only Python runtime caches."""
    source = ROOT / "src"
    if not source.is_dir() or not (source / "wheel_legged_control/d1/assets").is_dir():
        raise ValueError("source/assets directory is missing")
    paths = {ROOT / SCRIPT, ROOT / CHILD_SCRIPT, ROOT / "pyproject.toml"}
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError("source tree contains a symlink")
        if "__pycache__" in path.relative_to(source).parts or path.suffix in (".pyc", ".pyo"):
            continue
        if not path.is_dir():
            paths.add(path)
    result = {}
    for path in sorted(paths):
        if not path.is_file() or any(parent.is_symlink() for parent in (path, *path.parents)):
            raise ValueError(f"source must be a regular non-symlink file: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result[str(path.relative_to(ROOT))] = digest.hexdigest()
    return result


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: Path) -> str:
    if not path.is_file() or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"artifact must be a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _archive_extras(output: Path, hashes: dict[str, str]) -> dict[str, str]:
    extras = {}
    for name, expected in hashes.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("source archive paths must be repository-relative without traversal")
        child_snapshots = (
            name in (CHILD_SCRIPT, "pyproject.toml")
            or (path.parts[0] == "src" and path.suffix == ".py")
            or (
                path.is_relative_to("src/wheel_legged_control/d1/assets")
                and path.suffix.lower() in (".xml", ".urdf", ".stl")
            )
        )
        if child_snapshots:
            continue
        if _file_sha256(ROOT / name) != expected:
            raise RuntimeError("source changed before archiving parent-only files")
        relative = "orchestrator_source.py" if name == SCRIPT else f"source_extras/{name}"
        target = output / relative
        if any(part.is_symlink() for part in (target, *target.parents)):
            raise ValueError("source archive output must not contain symlink aliases")
        target.parent.mkdir(parents=True, exist_ok=True)
        with (ROOT / name).open("rb") as source, target.open("xb") as destination:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                destination.write(block)
        if _file_sha256(target) != expected:
            raise RuntimeError("source changed while archiving parent-only files")
        extras[name] = relative
    return extras


def _child_artifacts(directory: Path, mode: str) -> dict[str, str]:
    names = (
        "protocol.json",
        "source.json",
        "source.tar.gz",
        "training.json" if mode == "train" else "evaluation.json",
    )
    hashes = {name: _file_sha256(directory / name) for name in names}
    source = json.loads((directory / "source.json").read_text(encoding="utf-8"))
    if source.get("archive_sha256") != hashes["source.tar.gz"]:
        raise ValueError("child source archive SHA256 differs from its source.json")
    return hashes


def _stop_child(process) -> None:
    previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # The leader may have exited while a forked worker still ignores TERM.
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            pass
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            raise RuntimeError("child did not stop within the cleanup deadline") from None
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _execute(command: list[str], environment: dict[str, str], stream) -> int:
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        code = process.wait()
    except BaseException:
        _stop_child(process)
        raise
    if code:
        _stop_child(process)
    return code


def _sigterm(signum, frame):
    raise KeyboardInterrupt(f"received signal {signum}")


def run(output: Path, *, smoke: bool = False, dry_run: bool = False) -> dict:
    output = _output_path(output)
    protocol = build_protocol(output, smoke=smoke)
    before = source_hashes()
    protocol["source_sha256"] = before
    if dry_run:
        return protocol
    environment = _environment()
    output.mkdir(parents=True, exist_ok=False)
    started, completed = time.perf_counter(), 0
    previous_sigterm = signal.signal(signal.SIGTERM, _sigterm)
    try:
        protocol["created_utc"] = _utc()
        write_json(output / "protocol.json", protocol)
        extras = _archive_extras(output, before)
        write_json(
            output / "source.json",
            {
                "sha256": before,
                "archived_extras": extras,
                "scope": "all src files except Python bytecode caches, child, orchestrator and pyproject; each child archives its own source",
            },
        )
        (output / "logs").mkdir()
        with (
            (output / "runs.jsonl").open("x", encoding="utf-8") as ledger,
            (output / "runs.csv").open("x", newline="", encoding="utf-8") as csv_stream,
        ):
            writer = csv.DictWriter(csv_stream, fieldnames=LEDGER_FIELDS)
            writer.writeheader()
            csv_stream.flush()
            for sequence, item in enumerate(protocol["runs"], 1):
                if source_hashes() != before:
                    raise RuntimeError("source changed before the next command; stopped")
                _output_path(output / item["name"])
                record = {
                    **item,
                    "sequence": sequence,
                    "started_utc": _utc(),
                    "log": f"logs/{item['name']}.log",
                }
                print(
                    f"[{completed + 1}/{len(protocol['runs'])}] {item['name']} -> {record['log']}",
                    flush=True,
                )
                tick = time.perf_counter()
                try:
                    with (output / record["log"]).open("x", encoding="utf-8") as stream:
                        record["returncode"] = _execute(item["command"], environment, stream)
                    if record["returncode"] == 0:
                        record["child_artifact_sha256"] = _child_artifacts(
                            output / item["name"], item["mode"]
                        )
                except BaseException as exc:
                    record.setdefault("returncode", None)
                    record.update(error_type=type(exc).__name__, error=str(exc))
                    raise
                finally:
                    record.update(ended_utc=_utc(), elapsed_s=time.perf_counter() - tick)
                    ledger.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                    ledger.flush()
                    writer.writerow(
                        {
                            **{key: record.get(key) for key in LEDGER_FIELDS},
                            "command_json": json.dumps(record["command"]),
                            "child_artifact_sha256_json": json.dumps(
                                record.get("child_artifact_sha256")
                            ),
                        }
                    )
                    csv_stream.flush()
                if record["returncode"]:
                    raise RuntimeError(
                        f"{item['name']} exited {record['returncode']}; inspect {record['log']}"
                    )
                if source_hashes() != before:
                    raise RuntimeError(f"source changed during {item['name']}; stopped")
                completed += 1
                print(
                    f"[{completed}/{len(protocol['runs'])}] command exited 0 ({record['elapsed_s']:.1f}s)",
                    flush=True,
                )
        after = source_hashes()
        if after != before:
            raise RuntimeError("source changed at the final gate")
        write_json(output / "source_consistency.json", {"unchanged": True, "sha256": after})
    except BaseException as exc:
        try:
            after = source_hashes()
            consistency = {
                "unchanged": after == before,
                "sha256": after,
                "changed": sorted(
                    name for name in set(before) | set(after) if before.get(name) != after.get(name)
                ),
            }
        except (OSError, ValueError) as source_error:
            consistency = {"unchanged": False, "error": str(source_error)}
        write_json(output / "source_consistency.json", consistency)
        write_json(
            output / "failure.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "commands_completed": completed,
                "planned_commands": len(protocol["runs"]),
                "elapsed_s": time.perf_counter() - started,
                "note": "partial outputs retained; no retry, resume, overwrite or quality claim",
            },
        )
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    summary = {
        "status": "commands_completed",
        "kind": protocol["kind"],
        "commands_completed": completed,
        "planned_commands": len(protocol["runs"]),
        "planned_evaluation_cases": protocol["planned_evaluation_cases"],
        "elapsed_s": time.perf_counter() - started,
        "quality_claim": "none; inspect child metrics and independently audit trajectories",
    }
    write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    output = args.output or ROOT / "results" / (
        "d1_budget_smoke" if args.smoke else "d1_budget_study"
    )
    try:
        result = run(output, smoke=args.smoke, dry_run=args.dry_run)
    except KeyboardInterrupt:
        parser.exit(130, "budget study interrupted; partial records retained\n")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"budget study failed: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()
