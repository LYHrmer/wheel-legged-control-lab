"""Exclusive root host for the single 11-R world-upright whole-ramp baseline."""

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
OUT = W / "reference_run_01"
GO = W / "astra_reference_go_11.json"
RESERVATION = W / "reference_root_reservation_11.json"
CONTRACT = W / "next_reference_contract_11.md"
CONTRACT_SHA = "2f891044eac2c881f7240dc0da22a79ced22458c75f1df7ac6b103b7a01ae375"
SCHEMA = "d1-world-upright-reference-11-v1"
BUDGET = {
    "control_limit": 1800,
    "normal_native_limit": 9000,
    "compiler_native_limit": 2,
    "wallclock_limit_s": 180,
}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


e = module("upright11_historical_preflight", W.parent / "launch_course_e_08.py")
identity, save = e.identity, e.save


def preflight():
    records, paths = e.preflight()
    previous = W.parent / "terminal_fix10"
    readback = previous / "probe_readback_01.json"
    report10 = json.loads(readback.read_text())
    expected10 = {
        "execution_complete": True,
        "main_native_all_13_arrays_bitwise_equal_to_original": True,
        "first_696_saved_observation_action_torque_reward_equal": True,
        "failed_tick_returned_task_failure_terminal": True,
        "main_replay_controls": 697,
        "post_terminal_zero_controls": 8,
        "normal_native_returned": 3525,
        "compiler_native_returned": 2,
        "reset_same_model_data_and_seed": True,
    }
    if any(report10.get(key) != value for key, value in expected10.items()):
        raise RuntimeError("11-R requires the closed independent terminal10 proof")
    closed_path = previous / "probe_closed_manifest_01.json"
    closed = json.loads(closed_path.read_text())
    for name, expected in closed["files"].items():
        path = previous / "probe_run_01" / name
        if identity(path) != expected:
            raise RuntimeError("closed terminal10 record changed: " + name)
        records[str(path)] = expected
    reports = []
    for number in (1, 2):
        path = W / f"pure_round_{number:02d}/receipt.json"
        if path.is_file():
            report = json.loads(path.read_text())
            if report.get("all_passed") is True:
                reports.append((path, report))
    if len(reports) != 1:
        raise RuntimeError("11-R needs exactly one successful bounded pure round")
    path, report = reports[0]
    if (not 1 <= report["collected"] <= 8 or report["robot_controls"] != 0
            or report["native_calls"] != 0 or report["sources_changed"] != []
            or report["blocked_engine_imports"] != []):
        raise RuntimeError("11-R pure-test precondition failed")
    records[str(path)] = identity(path)
    for name, expected in report["source_sha256"].items():
        if identity(Path(name))["sha256"] != expected:
            raise RuntimeError("11-R tested source changed: " + name)
        records[name] = identity(Path(name))
    extras = (
        Path(__file__).resolve(), CONTRACT, W / "run_world_upright_reference_11.py",
        W / "world_upright_course_11.py", W / "run_pure_tests_11.py",
        W / "astra_world_upright_design_11.md", readback, closed_path,
        previous / "verify_terminal_probe_10.py",
    )
    for path in extras:
        records[str(path)] = identity(path)
    if identity(CONTRACT)["sha256"] != CONTRACT_SHA:
        raise RuntimeError("11-R bounded contract changed")
    return records, paths



def checked_go(records):
    go = json.loads(GO.read_text())
    if go.get("decision") != "GO" or go.get("contract_sha256") != CONTRACT_SHA:
        raise RuntimeError("missing exact upright11 source GO")
    for key, value in BUDGET.items():
        if go.get(key) != value:
            raise RuntimeError("11-R GO budget differs: " + key)
    required = (Path(__file__).resolve(), W / "run_world_upright_reference_11.py",
                W / "world_upright_course_11.py", CONTRACT,
                W.parent / "terminal_fix10/probe_readback_01.json",
                W / "run_pure_tests_11.py")
    for path in required:
        if go.get("inputs", {}).get(str(path)) != identity(path):
            raise RuntimeError("11-R GO omits critical input: " + str(path))
    for name, expected in go["inputs"].items():
        if identity(Path(name)) != expected:
            raise RuntimeError("11-R GO input drift: " + name)
        records[name] = expected
    records[str(GO)] = identity(GO)
    return go



def cleanup_owned_worker(command):
    """Inspect the reserved session even after its rtk wrapper has exited."""
    report = {"live_owned_pids_before": [], "signals_sent": [], "no_orphans": False}
    child = json.loads((OUT / "child_pid.json").read_text())
    session = json.loads((OUT / "session.json").read_text())
    if child["argv"] != command or session["worker_argv"] != command:
        raise RuntimeError("cleanup argv differs from the reserved 11-R worker")
    group = child["pid"]
    if type(group) is not int or group <= 1:
        raise RuntimeError("invalid 11-R owned process group")
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
                        "11-R cleanup refuses foreign session in reused group"
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
                    raise RuntimeError("11-R cleanup refuses foreign argv in reused group")
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
        raise RuntimeError("11-R reservation already used; no restart or physical retry")
    command = [
        "rtk",
        "proxy",
        "/usr/bin/python3",
        "-B",
        str(W / "run_world_upright_reference_11.py"),
        "--session",
        str(OUT / "session.json"),
    ]
    child_env, environment = e.q.helper.child_environment(R, paths)
    child_env["PYTHONPATH"] += os.pathsep + str(W.parent / "course_impl08") + os.pathsep + str(W)
    environment["PYTHONPATH"] = child_env["PYTHONPATH"]
    documents = [CONTRACT]
    session = {
        "schema": SCHEMA,
        "phase": "reference_baseline",
        "run_id": "world_upright_reference_11_01",
        "wall_clock_utc": datetime.now(timezone.utc).isoformat(),
        **BUDGET,
        "segments": [1800],
        "seed": 88611,
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
        raise KeyboardInterrupt("root 11-R host interrupted by signal " + str(sig))

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
                    "wallclock_limit_s": 180,
                },
            )
            heartbeat = 0
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= 180:
                    raise TimeoutError("host 180s 11-R watchdog expired")
                try:
                    code = process.wait(timeout=min(30.0, 180 - elapsed))
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
