"""Run one new job inside an explicitly named, memory-limited systemd unit.

The caller creates the unit. This supervisor only signals its own new process
group; source experiments and failed output directories are never resumed.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from resource_guard42 import check_memory, snapshot_memory


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def close_owned_group(process, report):
    """Reap the leader and adopted descendants; wait on the whole group."""
    def exists():
        process.poll()
        if process.returncode is not None:
            while True:
                try:
                    pid, _ = os.waitpid(-process.pid, os.WNOHANG)
                    if pid == 0:
                        break
                except ChildProcessError:
                    break
        try:
            os.killpg(process.pid, 0)
            return True
        except ProcessLookupError:
            return False

    for sig, seconds in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        if not exists():
            break
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break  # The owned group exited after the existence check.
        report["signals"].append(signal.Signals(sig).name + "_owned_process_group")
        deadline = time.monotonic() + seconds
        while exists() and time.monotonic() < deadline:
            time.sleep(.1)
    report["owned_process_group_empty"] = not exists()
    report["child_reaped"] = process.poll() is not None
    if not report["owned_process_group_empty"]:
        report.update(status="INCOMPLETE", error="owned process group survived cleanup")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", required=True, type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.timeout <= 0:
        parser.error("a command and positive timeout are required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"schema": "bounded-control-job49-v1", "status": "INCOMPLETE",
              "command": command, "scope": args.scope, "timeout_s": args.timeout,
              "supervisor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "minimum_effective_bytes": None, "peak_group_bytes": 0,
              "child_started": False, "returncode": None, "error": None,
              "signals": [], "retry_permitted": False}
    process = None

    def interrupted(signum, _frame):
        raise RuntimeError(f"supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def sample():
        value = snapshot_memory(args.scope)
        value["elapsed_s"] = time.monotonic() - started
        with (args.output / "resources.jsonl").open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
        check_memory(value)
        available = value["effective_available_bytes"]
        previous = report["minimum_effective_bytes"]
        report["minimum_effective_bytes"] = available if previous is None else min(previous, available)
        report["peak_group_bytes"] = max(report["peak_group_bytes"],
                                         value["owned"]["current_bytes"],
                                         value["owned"].get("peak_bytes", 0))
        report["final_memory_events"] = value["owned"]["events"]
        if available < 2 << 30:
            raise RuntimeError("effective host/ancestor memory below 2 GiB reserve")
        return available

    try:
        if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")
        if os.getpriority(os.PRIO_PROCESS, 0) != 10:
            raise RuntimeError("supervisor must run at Nice=10")
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "LP_NUM_THREADS"):
            if os.environ.get(key) != "1":
                raise RuntimeError(f"{key} must equal 1")
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        report["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        stable_since = None
        while True:
            available = sample()
            now = time.monotonic()
            stable_since = (stable_since or now) if available >= 4 << 30 else None
            if stable_since is not None and now - stable_since >= 10:
                break
            if now - started >= 300:
                raise RuntimeError("4 GiB admission not sustained for 10 s within 300 s")
            time.sleep(2)
        with (args.output / "stdout.log").open("xb") as out, (args.output / "stderr.log").open("xb") as err:
            process = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
            report["child_started"] = True
            report["child_pid"] = process.pid
            while process.poll() is None:
                sample()
                if time.monotonic() - started >= args.timeout:
                    raise RuntimeError("job wall-time budget exhausted")
                time.sleep(1)
            report["returncode"] = process.wait()
            sample()
        if report["returncode"] != 0:
            raise RuntimeError(f"child exited {report['returncode']}")
        report["status"] = "COMPLETE"
    except BaseException as exc:  # noqa: BLE001 -- retain bounded-job interruption evidence
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if process is not None:
            try:
                close_owned_group(process, report)
            except BaseException as exc:  # noqa: BLE001 -- retain cleanup failures too
                report.update(status="INCOMPLETE", cleanup_error=f"{type(exc).__name__}: {exc}")
        report["elapsed_s"] = time.monotonic() - started
        report["child_reaped"] = process is None or process.poll() is not None
        write_json(args.output / "report.json", report)
        print(json.dumps(report, indent=2), flush=True)
    return int(report["status"] != "COMPLETE")


if __name__ == "__main__":
    raise SystemExit(main())
