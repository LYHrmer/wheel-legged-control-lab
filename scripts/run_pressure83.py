"""Serial83 supervisor with byte reserve, PSI and reclaim-event admission.

Only verified descendants in the owned cgroup are signalled. It never changes
limits, stops unrelated programs, resumes work or retries a failed command.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from scripts.resource_guard42 import (
    _directory,
    _read,
    _scope,
    _self_scope,
    check_memory,
    snapshot_memory,
)
from scripts.resource_pressure83 import check_pressure, read_pressures
from scripts.run_bounded49 import write_json


def owned_members(scope, cgroup_root=Path("/sys/fs/cgroup")):
    """Include descendant groups; an actor may create its own process group."""
    if _self_scope(Path("/proc")) != scope:
        raise RuntimeError("cleanup supervisor left its registered cgroup")
    directory = _directory(cgroup_root, _scope(scope))
    before = directory.stat()
    files = [directory / "cgroup.procs", *directory.glob("**/*/cgroup.procs")]
    members = set()
    for path in files:
        relative = "/" + str(path.parent.relative_to(cgroup_root))
        _directory(cgroup_root, _scope(relative))
        for item in _read(path).splitlines():
            if not item.isascii() or not item.isdigit() or int(item) <= 0:
                raise ValueError("invalid owned process id")
            members.add(int(item))
    after = directory.stat()
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise RuntimeError("owned cgroup identity changed during cleanup")
    return members - {os.getpid()}


def signal_owned_pid(scope, pid, sig):
    """Pin the process with pidfd and recheck cgroup; PID reuse cannot escape."""
    if pid == os.getpid():
        raise ValueError("cannot signal supervisor")
    try:
        descriptor = os.pidfd_open(pid, 0)
    except ProcessLookupError:
        return False
    try:
        try:
            unified = [
                s[3:]
                for s in _read(Path("/proc") / str(pid) / "cgroup").splitlines()
                if s.startswith("0::")
            ]
        except (FileNotFoundError, ProcessLookupError):
            return False
        if len(unified) != 1 or not (unified[0] == scope or unified[0].startswith(scope + "/")):
            raise RuntimeError("refusing to signal process outside owned cgroup")
        try:
            signal.pidfd_send_signal(descriptor, sig, None, 0)
        except ProcessLookupError:
            return False
        return True
    finally:
        os.close(descriptor)


def close_owned_cgroup(process, scope, report):
    # Popen owns/reaps its direct child first. PR_SET_CHILD_SUBREAPER adopts
    # grandchildren after that exit, including actors in independent PGIDs.
    def reap():
        process.poll()
        if process.returncode is not None:
            while True:
                try:
                    pid, _ = os.waitpid(-1, os.WNOHANG)
                    if pid == 0:
                        break
                except ChildProcessError:
                    break

    for sig, seconds in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        deadline = time.monotonic() + seconds
        sent = set()
        while True:
            reap()
            members = owned_members(scope)
            if not members:
                break
            for pid in sorted(members - sent):
                if signal_owned_pid(scope, pid, sig):
                    report["signals"].append(
                        {
                            "pid": pid,
                            "signal": signal.Signals(sig).name,
                            "scope": "verified_owned_cgroup_pidfd",
                        }
                    )
                sent.add(pid)
            if time.monotonic() >= deadline:
                break
            time.sleep(0.1)
    reap()
    remaining = sorted(owned_members(scope))
    report["owned_cgroup_children_empty"] = not remaining
    report["remaining_owned_pids"] = remaining
    report["child_reaped"] = process.poll() is not None
    if remaining or not report["child_reaped"]:
        report.update(status="INCOMPLETE", cleanup_error="owned cgroup processes survived cleanup")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", required=True, type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 0 < args.timeout <= 3600:
        parser.error("command and timeout in (0,3600] required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "schema": "pressure-supervisor83-v1",
        "status": "INCOMPLETE",
        "command": command,
        "scope": args.scope,
        "timeout_s": args.timeout,
        "supervisor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pressure_policy_sha256": hashlib.sha256(
            (ROOT / "scripts/resource_pressure83.py").read_bytes()
        ).hexdigest(),
        "child_started": False,
        "returncode": None,
        "error": None,
        "signals": [],
        "minimum_effective_bytes": None,
        "peak_group_bytes": 0,
        "retry_permitted": False,
        "startup_full_avg10_strictly_below": 1.0,
        "runtime_full_avg10_strictly_below": 10.0,
        "maximum_full_avg10": {},
        "new_physics_steps_in_supervisor": 0,
    }
    process = None

    def stop(signum, _frame):
        raise InterruptedError("supervisor signal " + str(signum))

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def sample(startup):
        snap = snapshot_memory(args.scope)
        snap["elapsed_s"] = time.monotonic() - started
        snap["stage"] = "startup" if startup else "runtime"
        try:
            check_memory(snap)
            snap["pressures"] = read_pressures(snap)
        except BaseException as exc:
            snap["pressure_read_failure"] = type(exc).__name__ + ": " + str(exc)
            raise
        finally:
            with (args.output / "resources.jsonl").open("a") as stream:
                stream.write(json.dumps(snap, allow_nan=False) + "\n")
        available = snap["effective_available_bytes"]
        previous = report["minimum_effective_bytes"]
        report["minimum_effective_bytes"] = (
            available if previous is None else min(previous, available)
        )
        report["peak_group_bytes"] = max(
            report["peak_group_bytes"],
            snap["owned"]["current_bytes"],
            snap["owned"].get("peak_bytes") or 0,
        )
        report["final_memory_events"] = snap["owned"]["events"]
        for path, value in snap["pressures"].items():
            report["maximum_full_avg10"][path] = max(
                report["maximum_full_avg10"].get(path, 0.0), value["full"]["avg10"]
            )
        if available < 2 << 30:
            raise RuntimeError("effective memory below original2GiB reserve")
        # Owned reclaim/max events are irreversible for this fresh unit.
        check_pressure(snap, snap["pressures"], startup=False)
        if startup:
            return available >= 4 << 30 and all(
                v["full"]["avg10"] < 1.0 for v in snap["pressures"].values()
            )
        return True

    try:
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise RuntimeError("Linux pidfd cleanup required")
        if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")
        if os.getpriority(os.PRIO_PROCESS, 0) != 10:
            raise RuntimeError("Nice10 required")
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "LP_NUM_THREADS",
        ):
            if os.environ.get(key) != "1":
                raise RuntimeError(key + " must be1")
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        report["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        stable = None
        while True:
            admitted = sample(True)
            now = time.monotonic()
            stable = (stable or now) if admitted else None
            if stable is not None and now - stable >= 10:
                break
            if now - started >= 300:
                raise RuntimeError("4GiB/quiet-PSI admission not sustained10s within300s")
            time.sleep(1)
        with (
            (args.output / "stdout.log").open("xb") as out,
            (args.output / "stderr.log").open("xb") as err,
        ):
            process = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
            report.update(child_started=True, child_pid=process.pid)
            child_start = time.monotonic()
            while process.poll() is None:
                sample(False)
                if time.monotonic() - child_start >= args.timeout:
                    raise RuntimeError("child wall-time budget exhausted")
                time.sleep(0.25)
            report["returncode"] = process.wait()
            sample(False)
        if report["returncode"] != 0:
            raise RuntimeError("child exit " + str(report["returncode"]))
        report["status"] = "COMPLETE"
    except BaseException as exc:  # noqa: BLE001 - retain errors and prefixes.
        report["error"] = type(exc).__name__ + ": " + str(exc)
    finally:
        if process is not None:
            try:
                close_owned_cgroup(process, args.scope, report)
            except BaseException as exc:  # noqa: BLE001 - cleanup must remain auditable.
                report.update(
                    status="INCOMPLETE", cleanup_error=type(exc).__name__ + ": " + str(exc)
                )
        report["elapsed_s"] = time.monotonic() - started
        report["child_reaped"] = process is None or process.poll() is not None
        if process is not None:
            report["returncode_after_cleanup"] = process.returncode
        write_json(args.output / "report.json", report)
        print(json.dumps(report, allow_nan=False))
    return int(report["status"] != "COMPLETE")


if __name__ == "__main__":
    raise SystemExit(main())
