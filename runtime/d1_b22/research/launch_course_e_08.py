"""Root-only, exclusive headless E diagnostic; no implicit retry or training."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

W = Path(__file__).resolve().parent
R = Path("/home/lyh/wheel-legged-control-lab")
OUT = W / "course_e_run_01"
GO = W / "astra_course_e_go_08_02.json"
RESERVATION = W / "course_e_root_reservation_08.json"
CONTRACT = W / "next_rl_pilot_contract_08.md"
CONTRACT_SHA = "475273fd4517263606af7abd07008cdd6811f9992e6affacf6eb758a9ccbddd7"
SCHEMA = "d1-course-e-rl16-authority-speed-08-v1"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


q = module("course_e_historical_preflight", W / "launch_compact_08.py")
identity, save = q.identity, q.save


def preflight():
    records, paths = q.source_preflight()
    extras = [
        Path(__file__).resolve(),
        CONTRACT,
        W / "rl_schedule_geometry_addendum_08_01.md",
        W / "rl_semantics_and_terrain_addendum_08_02.md",
        W / "run_course_e_08.py",
        W / "run_course_pure_tests_08.py",
        W / "claude_residual16_provenance_08.json",
        W / "headless_progress_addendum_08_03.md",
        W / "compact_outcome_readback_08.json",
        W / "compact_closed_manifest_08.json",
        *(
            W / "course_impl08" / name
            for name in (
                "course_ground_08.py",
                "course_plant_08.py",
                "course_native_guard_08.py",
                "course_corridor_08.py",
                "full_drive_command_08.py",
                "full_drive_controller_08.py",
                "full_drive_env_08.py",
                "full_drive_loop_08.py",
                "full_drive_observation_08.py",
                "full_drive_schedule_08.py",
                "full_drive_servo_08.py",
                "residual16_math_08.py",
                "test_residual16_math_08.py",
                "test_course_seams_08.py",
            )
        ),
    ]
    for path in extras:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("missing regular E source: " + str(path))
        records[str(path)] = identity(path)
    if identity(CONTRACT)["sha256"] != CONTRACT_SHA:
        raise RuntimeError("E contract identity changed")
    closed = json.loads((W / "compact_closed_manifest_08.json").read_text())
    closed_root = Path(closed["run"])
    for name, expected in closed["files"].items():
        path = closed_root / name
        if identity(path) != expected:
            raise RuntimeError("closed Q record differs: " + str(path))
        records[str(path)] = expected
    outcome = json.loads((W / "compact_outcome_readback_08.json").read_text())
    if outcome.get("equivalence_passed") is not True:
        raise RuntimeError("Q numeric equivalence prerequisite failed")
    return records, paths


def checked_go(records):
    go = json.loads(GO.read_text())
    if go.get("decision") != "GO" or go.get("contract_sha256") != CONTRACT_SHA:
        raise RuntimeError("missing exact E GO")
    for key in (
        "control_limit",
        "normal_native_limit",
        "compiler_native_limit",
        "wallclock_limit_s",
    ):
        if (
            go.get(key)
            != {
                "control_limit": 2400,
                "normal_native_limit": 12000,
                "compiler_native_limit": 2,
                "wallclock_limit_s": 600,
            }[key]
        ):
            raise RuntimeError("E GO budget differs")
    for path in (Path(__file__).resolve(), W / "run_course_e_08.py", CONTRACT):
        if go.get("inputs", {}).get(str(path)) != identity(path):
            raise RuntimeError("E GO omitted critical source: " + str(path))
    for name, expected in go["inputs"].items():
        if identity(Path(name)) != expected:
            raise RuntimeError("E GO source drift: " + name)
        records[name] = expected
    pure = json.loads(Path(go["pure_receipt_path"]).read_text())
    if not (
        pure.get("all_passed")
        and pure.get("collected") == 16
        and pure.get("exit_code") == 0
        and pure.get("blocked_engine_imports") == []
        and pure.get("sources_changed_during_tests") == []
    ):
        raise RuntimeError("E pure integration gates not passed")
    records[str(GO)] = identity(GO)
    return go


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
                }
            )
        )
        return 0
    checked_go(records)
    if RESERVATION.exists() or OUT.exists():
        raise RuntimeError("E reserved already; physical retry forbidden")
    command = [
        "rtk",
        "proxy",
        "/usr/bin/python3",
        "-B",
        str(W / "run_course_e_08.py"),
        "--session",
        str(OUT / "session.json"),
    ]
    child_env, environment = q.helper.child_environment(R, paths)
    child_env["PYTHONPATH"] += os.pathsep + str(W / "course_impl08")
    environment["PYTHONPATH"] = child_env["PYTHONPATH"]
    session = {
        "schema": SCHEMA,
        "phase": "E",
        "control_limit": 2400,
        "normal_native_limit": 12000,
        "compiler_native_limit": 2,
        "wallclock_limit_s": 600,
        "segments": [800, 1600],
        "argv": command[4:],
        "worker_argv": command,
        "retry_permitted": False,
        "output_directory": str(OUT),
        "session_path": str(OUT / "session.json"),
        "source_hashes": records,
        "runtime_environment": environment,
        "contract_path": str(CONTRACT),
        "contract_sha256": CONTRACT_SHA,
        "seed": 88401,
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
            "control_limit": 2400,
            "normal_native_limit": 12000,
            "compiler_native_limit": 2,
            "go_sha256": session["go_sha256"],
            "automatic_retry": False,
        },
    )
    OUT.mkdir(exist_ok=False)
    save(OUT / "session.json", session)
    process = None
    failure = None
    code = None
    started = time.monotonic()

    def interrupted(_sig, _frame):
        raise KeyboardInterrupt("root E launcher interrupted")

    previous = signal.signal(signal.SIGTERM, interrupted)
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
            save(OUT / "child_pid.json", {"pid": process.pid, "argv": command})
            code = process.wait(timeout=600)
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    finally:
        if process is not None:
            q.helper.stop_owned(process, reason="08-E closure")
            code = process.returncode
        signal.signal(signal.SIGTERM, previous)
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
                "elapsed_wall_s": time.monotonic() - started,
                "retry_permitted": False,
            },
        )
    return 0 if code == 0 and failure is None and not changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
