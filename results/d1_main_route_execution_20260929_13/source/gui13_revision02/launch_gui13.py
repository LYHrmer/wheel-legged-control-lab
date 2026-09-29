"""Stdlib-only reservation and one-shot host for each GUI13 development arm."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from gui13_contract import (
    CHECKPOINT,
    HERE,
    MANIFEST,
    PARTIAL_CLOSED,
    READBACK,
    SCHEMA,
    TRAIN,
    TRAIN_HOST,
    W,
    check_plan,
    digest,
    read_json,
)

R = Path("/home/lyh/wheel-legged-control-lab")


def save_exclusive(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def input_paths() -> list[Path]:
    train_session = read_json(TRAIN / "session.json")
    paths = [Path(name) for name in train_session["source_hashes"]]
    paths.extend([Path(__file__), HERE / "prepare_gui13_plan.py",
                  HERE / "run_gui13.py", HERE / "gui13_contract.py",
                  HERE / "gui13_bridge.py", W / "gui12/run_world_upright_gui_12.py",
                  HERE / "phase_contract.md",
                  W / "gui12/latest_frame_mailbox_12.py",
                  W / "gui12/async_course_renderer_12.py", READBACK, PARTIAL_CLOSED,
                  TRAIN_HOST, MANIFEST,
                  CHECKPOINT / "final_metadata.json", TRAIN / "session.json",
                  W.parent / "gui_isolation/isolated_x11.py",
                  R / "docs/main_plan_20260929.md",
                  W / "continuation13/astra_plan_13.md",
                  W / "continuation13/archive_green_receipt_01.json",
                  W / "continuation13/gui_bridge_green_receipt_01.json",
                  W / "continuation13/reader_green_receipt_01.json"])
    paths.extend(CHECKPOINT / name for name in read_json(MANIFEST)["files"])
    paths.extend((W / "continuation13/archive13").glob("*.py"))
    readers = list((W / "continuation13/sol_verify13").glob("*.py"))
    if {path.name for path in readers} != {"verify_gui13.py", "test_verify_gui13.py"}:
        raise RuntimeError("GUI13 independent pair reader source is not yet frozen")
    paths.extend(readers)
    return sorted({path.resolve(strict=True) for path in paths})


def sources() -> dict:
    actual = {str(path): digest(path) for path in input_paths()}
    training = read_json(TRAIN / "session.json")
    for name, expected in training["source_hashes"].items():
        if actual.get(str(Path(name).resolve(strict=True))) != expected:
            raise RuntimeError("11-S frozen training source differs: " + name)
    return actual


@contextmanager
def deadline(limit: float, label: str):
    """Independent wall limit for pure hashing on the launcher main thread."""
    def expired(_number, _frame):
        raise TimeoutError(label)

    remaining = limit - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(label)
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def preflight(plan_path: Path, arm: str):
    plan = read_json(plan_path)
    check_plan(plan)
    frozen = plan.get("inputs")
    actual = sources()
    if not isinstance(frozen, dict) or frozen != actual:
        raise RuntimeError("GUI13 GO frozen source closure differs")
    spec = plan["arms"][arm]
    training = read_json(TRAIN / "session.json")
    if (training.get("library") not in actual
            or training.get("binding_module") not in actual
            or training.get("continuation_import_repair") not in actual):
        raise RuntimeError("GUI13 fixed engine/import origins missing")
    child_env = os.environ.copy()
    environment = dict(training["runtime_environment"])
    environment["PYTHONPATH"] += os.pathsep + str(HERE)
    environment["PYTHONPATH"] += os.pathsep + str(W / "gui12")
    environment["PYTHONPATH"] += os.pathsep + str(W / "continuation13")
    if arm == "gui":
        environment["DISPLAY"] = os.environ.get("DISPLAY")
        environment["XAUTHORITY"] = os.environ.get("XAUTHORITY")
    for key, value in environment.items():
        if value is None:
            child_env.pop(key, None)
        else:
            child_env[key] = value
    return plan, spec, actual, training, child_env, environment


def owned_members(group: int) -> list[tuple[int, list[str]]]:
    """Read only the fresh child process group; zombies have already exited."""
    found = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            status = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            if status[0] == "Z" or int(status[2]) != group:
                continue
            argv = [word.decode(errors="replace") for word in
                    (directory / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")]
            found.append((int(directory.name), argv))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return found


def cleanup_owned(process: subprocess.Popen | None, deadline: float) -> dict:
    if process is None:
        return {"worker_started": False, "no_orphans": True}
    while owned_members(process.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if owned_members(process.pid):
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=1)
    return {"worker_started": True, "exit_code": process.returncode,
            "worker_exited": process.poll() is not None,
            "no_orphans": not owned_members(process.pid)}


def run_arm(plan_path: Path, arm: str) -> int:
    started = float(os.environ.get("GUI13_PREFLIGHT_STARTED_MONOTONIC", time.monotonic()))
    with deadline(started + 240, "GUI13 parent preflight exceeded 240 s"):
        _plan, spec, frozen, training, child_env, environment = preflight(plan_path, arm)
    frozen = {**frozen, str(plan_path): digest(plan_path)}
    output = Path(spec["output_directory"])
    reservation = Path(spec["reservation_path"])
    if output.exists() or reservation.exists():
        raise FileExistsError("GUI13 arm already reserved; no retry")
    command = ["rtk", "proxy", "/usr/bin/python3", "-B", str(HERE / "run_gui13.py"),
               "--session", str(output / "session.json")]
    # Reservation precedes output and child construction, and is never refunded.
    save_exclusive(reservation, {
        "schema": SCHEMA, "arm": arm, "plan_sha256": digest(plan_path)["sha256"],
        "worker_argv": command, "control_limit": 600, "normal_native_limit": 3000,
        "compiler_native_limit": 2, "wallclock_hard_s": 120,
        "preflight_wall_s": 240, "postcheck_wall_s": 180, "outer_host_wall_s": 600,
        "retry_permitted": False, "fully_reserved_even_on_failure": True})
    output.mkdir(exist_ok=False)
    session = {
        **spec, "schema": SCHEMA, "argv": command[4:],
        "worker_argv": command, "source_hashes": frozen,
        "runtime_environment": environment, "evidence": _plan["evidence"],
        "library": training["library"],
        "binding_module": training["binding_module"],
        "continuation_import_repair": training["continuation_import_repair"],
        "plan_path": str(plan_path), "plan_sha256": digest(plan_path)["sha256"],
        "host_started_monotonic": started,
        "preflight_started_monotonic": started,
        "wall_clock_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_exclusive(output / "session.json", session)
    process = None
    failure = None
    execution_started = None
    execution_ended = None
    ready_record = None
    try:
        with (output / "worker_stdout.log").open("xb") as stream:
            process = subprocess.Popen(command, cwd=R, env=child_env,
                                       stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            save_exclusive(output / "child_pid.json", {"pid": process.pid,
                           "argv": command, "host_pid": os.getpid()})
            soft_sent = False
            cleanup_sent = False
            while process.poll() is None:
                if execution_started is None:
                    ready_path = output / "preflight_ready.json"
                    if ready_path.exists():
                        try:
                            candidate = read_json(ready_path)
                        except json.JSONDecodeError:
                            # The exclusive readiness record is still being written.
                            candidate = None
                        if candidate is not None:
                            members = {pid for pid, _argv in owned_members(process.pid)}
                            ready_ns = candidate.get("monotonic_ns")
                            if (candidate.get("pid") not in members
                                    or candidate.get("session_sha256") != digest(output / "session.json")["sha256"]
                                    or candidate.get("source_count") != len(frozen)
                                    or candidate.get("full_source_sha256_checked") is not True
                                    or candidate.get("engine_imported") is not False
                                    or candidate.get("model_loaded") is not False
                                    or type(ready_ns) is not int
                                    or not started <= ready_ns / 1e9 <= min(started + 240, time.monotonic())):
                                raise RuntimeError("GUI13 worker readiness identity differs")
                            ready_record = candidate
                            execution_started = ready_ns / 1e9
                    if execution_started is None:
                        if time.monotonic() >= started + 240:
                            raise TimeoutError("GUI13 worker preflight exceeded 240 s")
                        time.sleep(0.05)
                        continue
                elapsed = time.monotonic() - execution_started
                if elapsed >= 120:
                    raise TimeoutError("GUI13 hard 120s wall limit")
                if elapsed >= 95 and not soft_sent:
                    targets = [pid for pid, argv in owned_members(process.pid)
                               if str(HERE / "run_gui13.py") in argv]
                    for pid in targets:
                        os.kill(pid, signal.SIGTERM)
                    soft_sent = bool(targets)
                if elapsed >= 115 and not cleanup_sent:
                    os.killpg(process.pid, signal.SIGTERM)
                    cleanup_sent = True
                try:
                    process.wait(timeout=min(0.5, 120 - elapsed))
                except subprocess.TimeoutExpired:
                    pass
            execution_ended = time.monotonic()
            if execution_started is None:
                raise RuntimeError("GUI13 worker exited before readiness was acknowledged")
    except BaseException as error:  # noqa: BLE001 - always seal the reserved host budget
        failure = {"type": type(error).__name__, "message": str(error)}
    finally:
        cleanup_deadline = (execution_started + 120 if execution_started is not None
                            else min(started + 240, time.monotonic() + 5))
        cleanup = cleanup_owned(process, cleanup_deadline)
        post_started = time.monotonic()
        changed = []
        postcheck_complete = False
        try:
            with deadline(post_started + 180, "GUI13 pure postcheck exceeded 180 s"):
                changed = [name for name, expected in frozen.items()
                           if not Path(name).is_file() or digest(Path(name)) != expected]
            postcheck_complete = True
        except (OSError, TimeoutError) as error:
            failure = failure or {"type": type(error).__name__, "message": str(error)}
        receipt = {
            "schema": SCHEMA, "arm": arm, "failure": failure,
            "exit_code": None if process is None else process.returncode,
            "source_hash_mismatches": changed, "owned_worker_cleanup": cleanup,
            "elapsed_wall_s": time.monotonic() - started,
            "preflight_ready": ready_record,
            "preflight_elapsed_s": None if execution_started is None else execution_started - started,
            "execution_elapsed_s": None if execution_started is None or execution_ended is None
            else execution_ended - execution_started,
            "postcheck_elapsed_s": time.monotonic() - post_started,
            "full_source_postcheck_complete": postcheck_complete,
            "fully_reserved_budget_closed": True, "retry_permitted": False}
        save_exclusive(output / "launcher_receipt.json", receipt)
    return 0 if (failure is None and not changed and cleanup["no_orphans"]
                 and process is not None and process.returncode == 0) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--arm", choices=("headless", "gui"), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--isolated-child", action="store_true")
    args = parser.parse_args()
    plan = args.plan.resolve(strict=True)
    if args.check:
        with deadline(time.monotonic() + 240, "GUI13 pure check exceeded 240 s"):
            preflight(plan, args.arm)
        print(json.dumps({"source_preflight_passed": True,
                          "engine_imported": False, "model_loaded": False}))
        return 0
    if args.arm == "gui" and not args.isolated_child:
        preflight_started = float(os.environ.get("GUI13_PREFLIGHT_STARTED_MONOTONIC", time.monotonic()))
        os.environ["GUI13_PREFLIGHT_STARTED_MONOTONIC"] = str(preflight_started)
        with deadline(preflight_started + 240, "GUI13 outer preflight exceeded 240 s"):
            _, spec, _, _, _, _ = preflight(plan, args.arm)
        source = W.parent / "gui_isolation/isolated_x11.py"
        loader = importlib.util.spec_from_file_location("gui13_isolated_x11", source)
        if loader is None or loader.loader is None:
            raise RuntimeError("GUI13 isolated X11 source unavailable")
        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)
        command = ["rtk", "proxy", "/usr/bin/python3", "-B", str(Path(__file__).resolve()),
                   "--plan", str(plan), "--arm", "gui", "--run", "--isolated-child"]
        return module.run(command, Path(spec["output_directory"]))
    return run_arm(plan, args.arm)


if __name__ == "__main__":
    raise SystemExit(main())
