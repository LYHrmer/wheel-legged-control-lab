"""Exclusive root host for the single new world-upright 11-T training and heldout reservation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

W = Path(__file__).resolve().parent
R = Path("/home/lyh/wheel-legged-control-lab")
OUT = W / "training_run_01"
GO = W / "astra_short_rl_go_11.json"
RESERVATION = W / "short_rl_root_reservation_11.json"
CONTRACT = W / "next_short_training_contract_11.md"
CONTRACT_SHA = "e6e9d9d61d43d5eb3184c6c0cb31493f9f95e76cf4ab5632de02ce28135cd929"
SCHEMA = "d1-world-upright-short-rl16-training-heldout-11-v1"
SCORING = CONTRACT
MODEL_PREFLIGHT = W / "model_preflight_01/receipt.json"
BUDGET = {
    "control_limit": 85136,
    "normal_native_limit": 425680,
    "compiler_native_limit": 2,
    "wallclock_limit_s": 1800,
}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


e = module("short_rl11_historical_preflight", W.parent / "launch_course_e_08.py")
identity, save = e.identity, e.save


def preflight():
    records, paths = e.preflight()
    closed_path = W.parent / "upright11/reference_closed_manifest_01.json"
    closed = json.loads(closed_path.read_text())
    for name, expected in closed["files"].items():
        path = Path(closed["run"]) / name
        if identity(path) != expected:
            raise RuntimeError("closed 11-R record changed: " + str(path))
        records[str(path)] = expected
    readback_path = W.parent / "upright11/reference_readback_01.json"
    outcome = json.loads(readback_path.read_text())
    for name in ("execution_record_valid", "reference_baseline_passed",
                 "geometry_and_safety_passed", "full_1800_controls",
                 "all_three_ramp_geom_positive_wheel_load"):
        if outcome.get(name) is not True:
            raise RuntimeError("world-upright baseline prerequisite failed: " + name)
    model = json.loads(MODEL_PREFLIGHT.read_text())
    if model.get("passed") is not True or model.get("synthetic_only") is not True:
        raise RuntimeError(
            "real PPO model interface has not passed synthetic preflight"
        )
    if model.get("robot_control_steps") != 0 or model.get("native_steps") != 0:
        raise RuntimeError("synthetic preflight unexpectedly used physics")
    if (
        model.get("blocked_engine_imports") != []
        or model.get("source_hashes_changed") != []
    ):
        raise RuntimeError("synthetic preflight imported engine or source drifted")
    if (model.get("actual_model_calls") != {"save": 1, "load": 1, "predict": 3}
            or model.get("training_receipt", {}).get("actual") != {
                "num_timesteps": 1024, "train_calls": 1, "epochs": 4,
                "optimizer_steps": 16, "rollouts": 1, "transitions": 1024,
            }
            or len(model.get("loader_rejections", [])) != 3):
        raise RuntimeError("new completed-budget interface counts/rejections differ")
    # This imports no saved synthetic checkpoint and never uses its learned weights.
    for name, expected in model["source_hashes_before"].items():
        if identity(Path(name)) != expected:
            raise RuntimeError("PPO preflight source/dependency changed: " + name)
        records[name] = expected
    # The CPU policy uses a CUDA-enabled Torch distribution. Its import loader
    # may map vendor ELFs even though learning never selects a CUDA device.
    for base in (
        Path("/home/lyh/.local/lib/python3.10/site-packages"),
        R / ".local-deps",
    ):
        vendor = base / "nvidia"
        if vendor.is_dir():
            for path in sorted(vendor.rglob("*")):
                if path.is_file() and (
                    path.suffix in (".py", ".so") or ".so." in path.name
                ):
                    records[str(path)] = identity(path)
    passed_rounds = []
    for number in (1, 2):
        path = W / f"pure_round_{number:02d}/receipt.json"
        if path.is_file() and json.loads(path.read_text()).get("all_passed") is True:
            passed_rounds.append(path)
    if len(passed_rounds) != 1:
        raise RuntimeError("new schedule/scorer pure qualification not uniquely passed")
    score_receipt = passed_rounds[0]
    score_test = json.loads(score_receipt.read_text())
    if not (
        score_test.get("all_passed") is True
        and 1 <= score_test.get("collected", 0) <= 8
        and score_test.get("robot_controls") == 0
        and score_test.get("native_calls") == 0
        and score_test.get("sources_changed") == []
        and score_test.get("blocked_engine_imports") == []
    ):
        raise RuntimeError("bounded new pure schedule/scorer tests have not passed")
    for name, expected in score_test["source_sha256"].items():
        if identity(Path(name))["sha256"] != expected:
            raise RuntimeError("tested new source changed: " + name)
        records[name] = identity(Path(name))
    extras = [
        Path(__file__).resolve(), CONTRACT, W / "short_seed_semantics_11.md",
        W / "run_short_rl_11.py",
        W / "run_short_rl16_training_11.py",
        W / "rl16_learning_11.py",
        W / "budget_spec_11.py",
        W / "short_curriculum_11.py",
        W / "short_episode_plan_11.py",
        W / "short_plan_recipe_11.py",
        W / "short_heldout_recipe_11.py",
        W / "short_heldout_11.py",
        W / "short_corridor_11.py",
        W / "rl16_heldout_score_11.py",
        W / "model_preflight_contract_11.md",
        W / "model_preflight_count_clarification_11.md",
        MODEL_PREFLIGHT, score_receipt, closed_path, readback_path,
        W.parent / "upright11/world_upright_course_11.py",
        W.parent / "upright11/run_world_upright_reference_11.py",
        W.parent / "upright11/verify_world_upright_reference_11.py",
        W.parent / "run_rl16_training_08.py",
        W.parent / "course_impl08/rl16_curriculum_08.py",
    ]
    for path in extras:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("missing regular new training input: " + str(path))
        records[str(path)] = identity(path)
    if identity(CONTRACT)["sha256"] != CONTRACT_SHA:
        raise RuntimeError("new short-training contract changed")
    return records, paths


def checked_go(records):
    go = json.loads(GO.read_text())
    if go.get("decision") != "GO" or go.get("contract_sha256") != CONTRACT_SHA:
        raise RuntimeError("missing exact T GO")
    for key, value in BUDGET.items():
        if go.get(key) != value:
            raise RuntimeError("T GO budget differs: " + key)
    required = [
        Path(__file__).resolve(), W / "run_short_rl_11.py",
        W / "run_short_rl16_training_11.py", W / "rl16_learning_11.py",
        W / "budget_spec_11.py", W / "short_curriculum_11.py",
        W / "short_episode_plan_11.py", W / "short_plan_recipe_11.py",
        W / "short_heldout_recipe_11.py", W / "short_heldout_11.py",
        W / "short_corridor_11.py",
        W / "rl16_heldout_score_11.py", CONTRACT, MODEL_PREFLIGHT,
        W.parent / "upright11/reference_readback_01.json",
    ]
    for path in required:
        if go.get("inputs", {}).get(str(path)) != identity(path):
            raise RuntimeError("T GO omits critical input: " + str(path))
    for name, expected in go["inputs"].items():
        if identity(Path(name)) != expected:
            raise RuntimeError("T GO source changed: " + name)
        records[name] = expected
    records[str(GO)] = identity(GO)
    return go


def cleanup_owned_worker(command):
    """Inspect the reserved session even after its rtk wrapper has exited."""
    report = {"live_owned_pids_before": [], "signals_sent": [], "no_orphans": False}
    child = json.loads((OUT / "child_pid.json").read_text())
    session = json.loads((OUT / "session.json").read_text())
    if child["argv"] != command or session["worker_argv"] != command:
        raise RuntimeError("cleanup argv differs from the reserved T worker")
    group = child["pid"]
    if type(group) is not int or group <= 1:
        raise RuntimeError("invalid T owned process group")
    allowed = (command, command[2:])

    def members():
        found = []
        for folder in Path("/proc").iterdir():
            if not folder.name.isdigit():
                continue
            try:
                stat = (folder / "stat").read_text().rsplit(")", 1)[1].split()
                if int(stat[2]) != group or stat[0] == "Z":
                    continue
                if int(stat[3]) != group:
                    raise RuntimeError(
                        "T cleanup refuses foreign session in reused group"
                    )
                raw = (folder / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
                argv = [item.decode() for item in raw]
                # Executable argv[0] can be absolute; every remaining argument is exact.
                expected = next(
                    (
                        row
                        for row in allowed
                        if argv[1:] == row[1:]
                        and Path(argv[0]).name == Path(row[0]).name
                    ),
                    None,
                )
                if expected is None:
                    raise RuntimeError("T cleanup refuses foreign argv in reused group")
                found.append(int(folder.name))
            except (FileNotFoundError, ProcessLookupError):
                continue
        return found

    report["live_owned_pids_before"] = members()
    for number, grace in ((signal.SIGTERM, 15), (signal.SIGKILL, 3)):
        if not members():
            break
        try:
            os.killpg(group, number)
        except ProcessLookupError:
            break
        report["signals_sent"].append(int(number))
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and members():
            time.sleep(0.05)
    report["no_orphans"] = not members()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    records, paths = preflight()
    if args.check:
        print(
            json.dumps(
                {
                    "source_preflight_passed": True,
                    "input_count": len(records),
                    "engine_imported": False,
                    "model_loaded": False,
                }
            )
        )
        return 0
    checked_go(records)
    if RESERVATION.exists() or OUT.exists():
        raise RuntimeError("T reservation already used; no restart or physical retry")
    command = [
        "rtk",
        "proxy",
        "/usr/bin/python3",
        "-B",
        str(W / "run_short_rl_11.py"),
        "--session",
        str(OUT / "session.json"),
    ]
    child_env, environment = e.q.helper.child_environment(R, paths)
    child_env["PYTHONPATH"] += os.pathsep + str(W.parent / "course_impl08") + os.pathsep + str(W.parent / "upright11") + os.pathsep + str(W) + os.pathsep + str(W.parent)
    environment["PYTHONPATH"] = child_env["PYTHONPATH"]
    documents = [
        CONTRACT, W / "short_seed_semantics_11.md", W / "model_preflight_contract_11.md",
        W / "model_preflight_count_clarification_11.md",
        W.parent / "upright11/next_reference_contract_11.md",
    ]
    # This imports only the frozen pure immutable count module.
    budget_module = module("budget_spec_11", W / "budget_spec_11.py")
    budget = budget_module.BudgetSpec(65536, 1024, 256, 4)
    session = {
        "schema": SCHEMA,
        "phase": "T11",
        "run_id": "world_upright_short_rl11_01",
        "wall_clock_utc": datetime.now(timezone.utc).isoformat(),
        **BUDGET,
        "training_control_limit": 65536,
        "heldout_control_limit": 19600,
        "segments": [65536] + [1600] * 10 + [1800] * 2,
        "seed": 88621,
        "ppo_seed": 88621,
        "command_seed": 88622,
        "measurement_seed": 88623,
        "budget_spec": budget.as_dict(),
        "budget_spec_sha256": budget.canonical_sha256(),
        "short_plan_source_sha256": identity(W / "short_plan_recipe_11.py")["sha256"],
        "short_plan_schema": "d1-world-upright-short1000-preregistered-plan-v1",
        "reference_definition": {
            "frame": "world", "roll_target_rad": 0.0, "pitch_target_rad": 0.0,
            "compiled_height_normal_geom_scan_unchanged": True,
        },
        "argv": command[4:],
        "worker_argv": command,
        "retry_permitted": False,
        "output_directory": str(OUT),
        "session_path": str(OUT / "session.json"),
        "source_hashes": records,
        "runtime_environment": environment,
        "contract_path": str(CONTRACT),
        "contract_sha256": CONTRACT_SHA,
        "contract_documents": {str(p): identity(p) for p in documents},
        "preregistered_scoring_path": str(SCORING),
        "dependency_hashes": {
            p: v
            for p, v in records.items()
            if "/site-packages/" in p or "/.local-deps/" in p
        },
        "go_sha256": identity(GO)["sha256"],
        **{
            key: str(paths[key])
            for key in ("library", "binding_module", "continuation_import_repair")
        },
    }
    save(
        RESERVATION,
        {
            "schema": SCHEMA,
            "argv": command,
            "output": str(OUT),
            **BUDGET,
            "host_pid": os.getpid(),
            "created_utc": session["wall_clock_utc"],
            "fully_reserved_even_on_crash": True,
            "automatic_retry": False,
            "go_sha256": session["go_sha256"],
        },
    )
    OUT.mkdir(exist_ok=False)
    save(OUT / "session.json", session)
    process = None
    failure = None
    cleanup = {"no_orphans": True, "worker_was_not_started": True}
    code = None
    started = time.monotonic()

    def interrupted(sig, _frame):
        raise KeyboardInterrupt("root T host interrupted by signal " + str(sig))

    previous = {
        sig: signal.signal(sig, interrupted)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    }
    try:
        with (OUT / "worker_stdout.log").open("xb") as stream:
            process = subprocess.Popen(
                command,
                cwd=R,
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            save(
                OUT / "child_pid.json",
                {
                    "pid": process.pid,
                    "argv": command,
                    "host_pid": os.getpid(),
                    "wallclock_limit_s": 1800,
                },
            )
            heartbeat = 0
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= 1800:
                    raise TimeoutError("host 1800s T watchdog expired")
                try:
                    code = process.wait(timeout=min(30.0, 1800 - elapsed))
                except subprocess.TimeoutExpired:
                    # Append independent host evidence even if the worker is stuck.
                    save(
                        OUT / f"host_heartbeat_{heartbeat:04d}.json",
                        {
                            "pid": process.pid,
                            "host_pid": os.getpid(),
                            "elapsed_wall_s": time.monotonic() - started,
                            "worker_running": process.poll() is None,
                            "automatic_retry": False,
                        },
                    )
                    heartbeat += 1
            code = process.returncode
    except BaseException as error:  # noqa: BLE001 -- preserve failure and stop only owned worker
        failure = {"type": type(error).__name__, "message": str(error)}
    finally:
        if process is not None:
            try:
                cleanup = cleanup_owned_worker(command)
                code = process.wait(timeout=5)
            except Exception as error:  # noqa: BLE001 -- report cleanup failure without signaling strangers
                cleanup = {
                    "no_orphans": False,
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
                code = process.poll()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        changed = [
            name
            for name, expected in records.items()
            if not Path(name).is_file() or identity(Path(name)) != expected
        ]
        save(
            OUT / "launcher_receipt.json",
            {
                "schema": SCHEMA,
                "exit_code": code,
                "failure": failure,
                "source_hash_mismatches": changed,
                "worker_exited": process is None or process.poll() is not None,
                "owned_worker_cleanup": cleanup,
                "elapsed_wall_s": time.monotonic() - started,
                "fully_reserved_budget_closed": True,
                "retry_permitted": False,
            },
        )
    return (
        0
        if code == 0 and failure is None and not changed and cleanup["no_orphans"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
